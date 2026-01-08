# ---------------------------------------------------------------------------
# model_utils.py – shared business logic for both modules
# ---------------------------------------------------------------------------

from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import hashlib
import json
import pickle
import sqlite3
from datetime import datetime, date

import numpy as np
import pandas as pd
import pwlf

# Application paths
DATA_PATH = Path("data/marginal.csv")
MODEL_PATH = Path("models/2025_marginal.pkl")
DATA_DB_PATH = Path("data/marginal.sqlite")
CONFIG_PATH = Path("data/dataset_config.json")
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
DATA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DATASET_CONFIG: Dict[str, Any] = {
    "excel_path": "",
    "sheet_name": "",
}

HEADER_HINTS = {
    "日期",
    "时点",
    "时段",
    "time_slot",
    "竞价空间(MW)",
    "在线机组容量(MW)",
    "价格",
    "price",
    "负荷率(%)",
    "load_rate",
}

HEADER_ANCHORS = [
    "日期",
    "时点",
    "时段",
    "date",
    "time",
    "竞价",
    "容量",
    "价格",
    "负荷",
    "load",
    "price",
]


def _has_header_tokens(columns: List[Any]) -> bool:
    normalized = {
        str(col).replace("\xa0", "").strip()
        for col in columns
        if col is not None
    }
    return bool(HEADER_HINTS & normalized)


RAW_COLUMN_RENAME_MAP = {
    '日期': 'date',
    'date': 'date',
    '时点': 'time_slot',
    '时段': 'time_slot',
    '时间': 'time_slot',
    '边界数据类型': 'boundary_type',
    '竞价空间(MW)': 'bidding_space_mw',
    '竞价空间': 'bidding_space_mw',
    '省调负荷(MW)': 'dispatch_load_mw',
    '风电(MW)': 'wind_mw',
    '光伏(MW)': 'solar_mw',
    '新能源负荷(MW)': 'new_energy_mw',
    '非市场化出力(MW)': 'non_market_output_mw',
    '水电出力(MW)': 'hydro_mw',
    '联络线计划(MW)': 'tie_line_plan_mw',
    '在线机组容量(MW)': 'online_capacity_mw',
    '日前出清价格(元/MWh)': 'day_ahead_price',
    '实时出清价格(元/MWh)': 'real_time_price',
    '负荷率(%)': 'load_rate_pct',
}

RAW_NUMERIC_COLUMNS = {
    'bidding_space_mw',
    'dispatch_load_mw',
    'wind_mw',
    'solar_mw',
    'new_energy_mw',
    'non_market_output_mw',
    'hydro_mw',
    'tie_line_plan_mw',
    'online_capacity_mw',
    'day_ahead_price',
    'real_time_price',
    'load_rate_pct',
}

RAW_TABLE_CORE_COLUMNS = [
    'date',
    'time_slot',
    'boundary_type',
    'bidding_space_mw',
    'dispatch_load_mw',
    'wind_mw',
    'solar_mw',
    'new_energy_mw',
    'non_market_output_mw',
    'hydro_mw',
    'tie_line_plan_mw',
    'online_capacity_mw',
    'day_ahead_price',
    'real_time_price',
    'load_rate_pct',
]

RAW_TABLE_COLUMNS = RAW_TABLE_CORE_COLUMNS + ['extras_json']


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value

# -------------------------------------------
# Data cleaning (mirrors reappropriation.py)
# -------------------------------------------

def _normalize_token(val: str) -> str:
    simplified = val.replace("（", "(").replace("）", ")")
    simplified = simplified.replace(" ", "").replace("\t", "")
    return simplified.lower()


def _header_score(values: List[str]) -> int:
    score = 0
    for val in values:
        if not val:
            continue
        token = _normalize_token(val)
        for anchor in HEADER_ANCHORS:
            if anchor in token:
                score += 1
                break
    return score


def _maybe_promote_header(df: pd.DataFrame) -> pd.DataFrame:
    """If dataframe columns look like Unnamed, promote the first meaningful row as header."""

    def _norm_value(val: Any) -> str:
        if val is None:
            return ""
        if pd.isna(val):
            return ""
        return str(val).strip()

    current_cols = set(df.columns)
    if HEADER_HINTS & current_cols:
        return df

    max_scan = min(len(df), 10)
    for idx in range(max_scan):
        row = df.iloc[idx]
        normalized = [_norm_value(v) for v in row.tolist()]
        if not any(normalized):
            continue
        if (HEADER_HINTS & set(normalized)) or _header_score(normalized) >= 2:
            pass
        else:
            continue
        new_columns = []
        for col_idx, val in enumerate(normalized):
            if val:
                new_columns.append(val)
            else:
                fallback = str(df.columns[col_idx])
                new_columns.append(fallback if not fallback.startswith("Unnamed") else f"col_{col_idx}")
        df = df.iloc[idx + 1 :].reset_index(drop=True)
        df.columns = new_columns
        return df
    return df


def clean_marginal_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names, coerce dtypes, drop NaNs & outliers."""
    # Flexible rename map – extend if needed

    df = _maybe_promote_header(df)
    df.columns = [str(col).strip() for col in df.columns]

    # 2) 列名映射 (可按需要扩充)
    rename_map = {
        '日期': 'date',
        'date': 'date',

        '时点': 'time_slot',
        '时段': 'time_slot',
        'time_slot': 'time_slot',

        '(调控后)日前出清价格(元/MWh)': 'price',
        '日前出清价格(元/MWh)': 'price',
        '价格': 'price',
        'price': 'price',

        '日前负荷率(%)': 'load_rate',
        '负荷率(%)': 'load_rate',
        '负荷率': 'load_rate',
        'load_rate': 'load_rate',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Template fallback: derive load_rate if only capacity + space are provided.
    if (
        'load_rate' not in df.columns
        and '竞价空间(MW)' in df.columns
        and '在线机组容量(MW)' in df.columns
    ):
        df['load_rate'] = (
            pd.to_numeric(df['竞价空间(MW)'], errors='coerce')
            / pd.to_numeric(df['在线机组容量(MW)'], errors='coerce')
        ) * 100.0

    required = ['date', 'time_slot', 'price', 'load_rate']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'以下列缺失或列名不匹配: {missing}\n当前列: {list(df.columns)}')

    # 3) 类型转换
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['load_rate'] = pd.to_numeric(df['load_rate'], errors='coerce')

    df = df.dropna(subset=['price', 'load_rate'])
    # Drop invalid load rates to avoid negative breakpoints in pwlf
    df = df[(df['load_rate'] > 0) & (df['load_rate'] <= 100)].reset_index(drop=True)
    return df

# ------------------------------------
# Model training & management
# ------------------------------------

def train_model(
    df: pd.DataFrame,
    model_name: str,
    *,
    n_segments: int = 3,
) -> Tuple[pwlf.PiecewiseLinFit, np.ndarray]:
    """Train pwlf model on provided DataFrame and save to models/{model_name}.pkl

    The input DataFrame is cleaned via clean_marginal_data.
    Returns (model, breakpoints).
    """
    cleaned = clean_marginal_data(df.copy())
    x = cleaned["load_rate"].values
    y = cleaned["price"].values
    if len(x) < max(2, n_segments + 1):
        raise ValueError("数据量过少，无法训练指定段数的模型")
    model = pwlf.PiecewiseLinFit(x, y)
    breakpoints = model.fit(n_segments)
    out_path = MODELS_DIR / f"{model_name}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    return model, breakpoints


def list_models() -> List[str]:
    """Return list of available model names (without .pkl suffix)."""
    return sorted(p.stem for p in MODELS_DIR.glob("*.pkl"))


def load_model(name: str) -> Optional[pwlf.PiecewiseLinFit]:
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

# ------------------------------------
# Helpers for forecast functionality
# ------------------------------------

def load_or_train_model(n_segments: int = 3) -> Optional[pwlf.PiecewiseLinFit]:
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    # If model absent but dataset present -> auto‑train
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        x, y = df["load_rate"].values, df["price"].values
        if len(x) < n_segments + 2:
            return None
        model = pwlf.PiecewiseLinFit(x, y)
        model.fit(n_segments)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        return model
    return None


def predict_price(model: pwlf.PiecewiseLinFit, x: np.ndarray) -> np.ndarray:
    # Clamp predictions to non-negative to avoid spurious negatives from extrapolation
    return np.clip(model.predict(x), 0, None)


# -------------------------------------------------
# Dataset configuration & Excel → SQLite sync helpers
# -------------------------------------------------

def load_dataset_config() -> Dict[str, Any]:
    cfg = DEFAULT_DATASET_CONFIG.copy()
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            if isinstance(user_cfg, dict):
                for key in DEFAULT_DATASET_CONFIG.keys():
                    if key in user_cfg and user_cfg[key] is not None:
                        cfg[key] = str(user_cfg[key]).strip()
        except Exception:
            # Ignore config parse errors; fall back to defaults
            pass
    return cfg


def update_dataset_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    cfg = load_dataset_config()
    for key, default_val in DEFAULT_DATASET_CONFIG.items():
        if key in updates:
            val = updates[key]
            cfg[key] = str(val).strip() if val is not None else default_val
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return cfg


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DATA_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS marginal_data (
            date TEXT NOT NULL,
            time_slot TEXT NOT NULL,
            price REAL NOT NULL,
            load_rate REAL NOT NULL,
            PRIMARY KEY (date, time_slot)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_marginal_data (
            date TEXT,
            time_slot TEXT,
            boundary_type TEXT,
            bidding_space_mw REAL,
            dispatch_load_mw REAL,
            wind_mw REAL,
            solar_mw REAL,
            new_energy_mw REAL,
            non_market_output_mw REAL,
            hydro_mw REAL,
            tie_line_plan_mw REAL,
            online_capacity_mw REAL,
            day_ahead_price REAL,
            real_time_price REAL,
            load_rate_pct REAL,
            extras_json TEXT,
            PRIMARY KEY (date, time_slot, boundary_type)
        )
        """
    )


def _get_meta_dict(conn: sqlite3.Connection) -> Dict[str, str]:
    cur = conn.execute("SELECT key, value FROM dataset_meta")
    return {row[0]: row[1] for row in cur.fetchall()}


def _set_meta_bulk(conn: sqlite3.Connection, values: Dict[str, Any]) -> None:
    if not values:
        return
    conn.executemany(
        """
        INSERT INTO dataset_meta(key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        [(k, str(v)) for k, v in values.items()],
    )


def _count_rows(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COUNT(*) FROM marginal_data")
    return int(cur.fetchone()[0])


def _count_raw_rows(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COUNT(*) FROM raw_marginal_data")
    row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _get_date_span(conn: sqlite3.Connection) -> Tuple[Optional[str], Optional[str]]:
    cur = conn.execute("SELECT MIN(date), MAX(date) FROM marginal_data")
    row = cur.fetchone()
    return row[0], row[1]


def _format_epoch(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _normalize_date(val: Optional[Any]) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, str):
        stripped = val.strip()
        return stripped or None
    if isinstance(val, date):
        return val.strftime("%Y-%m-%d")
    return str(val)


def _load_source_frame(path: Path, sheet_name: Optional[str]) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        read_kwargs: Dict[str, Any] = {}
        if sheet_name:
            read_kwargs["sheet_name"] = sheet_name
        df = pd.read_excel(path, **read_kwargs)
        if _has_header_tokens(df.columns.tolist()):
            return df
        read_kwargs["header"] = None
        return pd.read_excel(path, **read_kwargs)
    if suffix in {".csv", ".txt"}:
        df = pd.read_csv(path)
        if _has_header_tokens(df.columns.tolist()):
            return df
        return pd.read_csv(path, header=None)
    raise ValueError(f"不支持的文件类型: {path.suffix}")


def _calc_dataframe_hash(df: pd.DataFrame) -> str:
    subset = df[["date", "time_slot", "price", "load_rate"]]
    hashed = pd.util.hash_pandas_object(subset, index=False).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def _prepare_raw_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    base = _maybe_promote_header(df)
    base.columns = [str(col).strip() for col in base.columns]
    rename = {k: v for k, v in RAW_COLUMN_RENAME_MAP.items() if k in base.columns}
    base = base.rename(columns=rename)
    if 'date' in base.columns:
        base['date'] = pd.to_datetime(base['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    for col in RAW_NUMERIC_COLUMNS:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors='coerce')
    extras_cols = [c for c in base.columns if c not in RAW_TABLE_CORE_COLUMNS]
    return base, extras_cols


def _replace_raw_table(conn: sqlite3.Connection, df: pd.DataFrame, extras_cols: List[str]) -> None:
    if df.empty:
        conn.execute("DELETE FROM raw_marginal_data")
        return
    records = df.to_dict(orient='records')
    rows = []
    for record in records:
        extras_payload = {}
        for col in extras_cols:
            if col in record:
                value = record[col]
                if pd.notna(value):
                    extras_payload[col] = _to_json_safe(value)
        extras_json = json.dumps(extras_payload, ensure_ascii=False) if extras_payload else None
        row_values = [record.get(col) for col in RAW_TABLE_CORE_COLUMNS]
        if row_values:
            boundary_idx = RAW_TABLE_CORE_COLUMNS.index('boundary_type')
            if row_values[boundary_idx] is None:
                row_values[boundary_idx] = ''
        rows.append(tuple(row_values + [extras_json]))
    conn.execute("DELETE FROM raw_marginal_data")
    conn.executemany(
        """
        INSERT INTO raw_marginal_data (
            date, time_slot, boundary_type, bidding_space_mw, dispatch_load_mw,
            wind_mw, solar_mw, new_energy_mw, non_market_output_mw, hydro_mw,
            tie_line_plan_mw, online_capacity_mw, day_ahead_price, real_time_price,
            load_rate_pct, extras_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def get_dataset_status() -> Dict[str, Any]:
    cfg = load_dataset_config()
    configured_path = cfg.get("excel_path", "").strip()
    configured_sheet = cfg.get("sheet_name", "").strip()
    excel_path = Path(configured_path).expanduser() if configured_path else None
    excel_exists = excel_path.exists() if excel_path else False
    excel_mtime = excel_path.stat().st_mtime if excel_exists else None

    with _get_conn() as conn:
        _ensure_db(conn)
        row_count = _count_rows(conn)
        meta = _get_meta_dict(conn)
        date_min, date_max = _get_date_span(conn)

    last_sync = meta.get("last_sync_time") if meta else None
    last_hash = meta.get("last_hash") if meta else None
    meta_row_count = meta.get("last_row_count") if meta else None
    last_row_count = int(meta_row_count) if meta_row_count else row_count
    meta_excel_mtime = meta.get("last_excel_mtime") if meta else None
    recorded_mtime = float(meta_excel_mtime) if meta_excel_mtime else None

    needs_sync = False
    reason = None
    if not configured_path:
        needs_sync = True
        reason = "未配置 Excel 路径"
    elif not excel_exists:
        needs_sync = True
        reason = "Excel 文件不存在"
    elif row_count == 0:
        needs_sync = True
        reason = "数据库为空"
    elif excel_mtime and recorded_mtime and excel_mtime > recorded_mtime + 1e-6:
        needs_sync = True
        reason = "Excel 文件有更新"

    return {
        "row_count": row_count,
        "date_min": date_min,
        "date_max": date_max,
        "last_sync_time": last_sync,
        "last_hash": last_hash,
        "last_row_count": last_row_count,
        "excel_path": configured_path,
        "sheet_name": configured_sheet,
        "excel_exists": excel_exists,
        "excel_mtime": excel_mtime,
        "excel_mtime_human": _format_epoch(excel_mtime) if excel_mtime else None,
        "needs_sync": needs_sync,
        "status_reason": reason,
        "state_label": "待同步" if needs_sync else "已同步",
    }


def sync_dataset_from_excel(force: bool = False) -> Dict[str, Any]:
    cfg = load_dataset_config()
    path_str = cfg.get("excel_path", "").strip()
    if not path_str:
        raise ValueError("请先在配置中设置 Excel 文件路径")
    source_path = Path(path_str).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Excel 文件不存在: {source_path}")
    sheet_name = cfg.get("sheet_name", "").strip() or None

    raw_df = _load_source_frame(source_path, sheet_name)
    raw_prepared, raw_extra_cols = _prepare_raw_dataframe(raw_df.copy())
    cleaned = clean_marginal_data(raw_df.copy())
    cleaned = cleaned.sort_values(["date", "time_slot"]).reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("清洗后的数据为空，无法导入")

    fingerprint = _calc_dataframe_hash(cleaned)
    excel_mtime = source_path.stat().st_mtime

    with _get_conn() as conn:
        _ensure_db(conn)
        meta = _get_meta_dict(conn)
        previous_hash = meta.get("last_hash") if meta else None
        previous_rows = _count_rows(conn)
        raw_rows_meta = meta.get("last_raw_row_count") if meta else None
        raw_rows_meta_val = int(raw_rows_meta) if raw_rows_meta else None
        raw_rows_current = _count_raw_rows(conn)
        raw_ready = meta.get("raw_table_ready") == "1" if meta else False
        raw_consistent = (
            raw_ready
            and raw_rows_meta_val is not None
            and raw_rows_meta_val == len(raw_prepared)
            and raw_rows_current == raw_rows_meta_val
        )
        if (not force) and previous_hash == fingerprint and previous_rows == len(cleaned) and raw_consistent:
            return {
                "status": "skipped",
                "reason": "数据无变化",
                "rows": previous_rows,
                "hash": fingerprint,
                "last_sync_time": meta.get("last_sync_time") if meta else None,
            }

        conn.execute("DELETE FROM marginal_data")
        conn.executemany(
            "INSERT INTO marginal_data(date, time_slot, price, load_rate) VALUES (?, ?, ?, ?)",
            cleaned[["date", "time_slot", "price", "load_rate"]].itertuples(index=False, name=None),
        )
        _replace_raw_table(conn, raw_prepared, raw_extra_cols)

        now_iso = datetime.utcnow().isoformat() + "Z"
        _set_meta_bulk(
            conn,
            {
                "last_sync_time": now_iso,
                "last_hash": fingerprint,
                "last_excel_path": str(source_path),
                "last_excel_mtime": excel_mtime,
                "last_row_count": len(cleaned),
                "last_sheet_name": sheet_name or "",
                "raw_table_ready": "1",
                "last_raw_row_count": len(raw_prepared),
            },
        )

    cleaned.to_csv(DATA_PATH, index=False)

    return {
        "status": "updated",
        "rows": len(cleaned),
        "hash": fingerprint,
        "last_sync_time": now_iso,
        "excel_path": str(source_path),
        "sheet_name": sheet_name,
    }


def fetch_dataset(date_start: Optional[Any] = None, date_end: Optional[Any] = None) -> pd.DataFrame:
    start = _normalize_date(date_start)
    end = _normalize_date(date_end)
    with _get_conn() as conn:
        _ensure_db(conn)
        query = "SELECT date, time_slot, price, load_rate FROM marginal_data WHERE 1=1"
        params: List[Any] = []
        if start:
            query += " AND date >= ?"
            params.append(start)
        if end:
            query += " AND date <= ?"
            params.append(end)
        query += " ORDER BY date, time_slot"
        df = pd.read_sql_query(query, conn, params=params)
    return df


# -------------------------------------------------
# Extended model management (artifacts + metadata)
# -------------------------------------------------

def _paths(name: str) -> Dict[str, Path]:
    base = MODELS_DIR
    return {
        "pkl": base / f"{name}.pkl",
        "meta": base / f"{name}.meta.json",
        "train": base / f"{name}.train.csv",
    }


def train_model_ex(
    df: pd.DataFrame,
    model_name: str,
    *,
    n_segments: int = 3,
    source: Optional[str] = None,
) -> Tuple[pwlf.PiecewiseLinFit, np.ndarray]:
    """Train + save model, metadata, and cleaned training data.

    This does not modify legacy train_model; callers should use this function
    to ensure artifacts exist for management UI.
    """
    cleaned = clean_marginal_data(df.copy())
    x = cleaned["load_rate"].values
    y = cleaned["price"].values
    if len(x) < max(2, n_segments + 1):
        raise ValueError("数据量过少，无法训练指定段数的模型")
    model = pwlf.PiecewiseLinFit(x, y)
    breakpoints = model.fit(n_segments)

    p = _paths(model_name)
    with open(p["pkl"], "wb") as f:
        pickle.dump(model, f)
    cleaned.to_csv(p["train"], index=False)
    meta: Dict[str, Any] = {
        "name": model_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "n_segments": n_segments,
        "rows": int(len(cleaned)),
        "source": source or "unknown",
        "breakpoints": [float(v) for v in np.asarray(breakpoints).tolist()],
    }
    with open(p["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return model, breakpoints


def load_model_meta(name: str) -> Optional[Dict[str, Any]]:
    p = _paths(name)["meta"]
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def delete_models(names: List[str]) -> Dict[str, List[str]]:
    removed: List[str] = []
    missing: List[str] = []
    for n in names:
        paths = _paths(n)
        any_removed = False
        for fp in paths.values():
            if fp.exists():
                try:
                    fp.unlink()
                    any_removed = True
                except Exception:
                    pass
        if any_removed:
            removed.append(n)
        else:
            missing.append(n)
    return {"removed": removed, "missing": missing}
