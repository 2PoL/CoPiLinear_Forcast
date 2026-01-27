# ---------------------------------------------------------------------------
# model_utils.py – shared business logic for both modules
# ---------------------------------------------------------------------------

from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Union

import hashlib
import json
import pickle
import sqlite3
from datetime import datetime, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pwlf

# Application paths
DATA_PATH = Path("data/marginal.csv")
MODEL_PATH = Path("models/2025_marginal.pkl")
DATA_DB_PATH = Path("data/marginal.sqlite")
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
DATA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SH_TZ = ZoneInfo("Asia/Shanghai")

DB_COL_DATE = "日期"
DB_COL_TIME = "时点"
DB_COL_PRICE = "日前出清价格(元/MWh)"
DB_COL_LOAD_RATE = "负荷率(%)"
RAW_DB_COLUMNS = [
    DB_COL_DATE,
    DB_COL_TIME,
    "边界数据类型",
    "竞价空间(MW)",
    "省调负荷(MW)",
    "风电(MW)",
    "光伏(MW)",
    "新能源负荷(MW)",
    "非市场化出力(MW)",
    "水电出力(MW)",
    "联络线计划(MW)",
    "在线机组容量(MW)",
    DB_COL_PRICE,
    "实时出清价格(元/MWh)",
    DB_COL_LOAD_RATE,
]

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


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DATA_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS marginal_data (
            "日期" TEXT NOT NULL,
            "时点" TEXT NOT NULL,
            "日前出清价格(元/MWh)" REAL NOT NULL,
            "负荷率(%)" REAL NOT NULL,
            PRIMARY KEY ("日期", "时点")
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
            "日期" TEXT,
            "时点" TEXT,
            "边界数据类型" TEXT,
            "竞价空间(MW)" REAL,
            "省调负荷(MW)" REAL,
            "风电(MW)" REAL,
            "光伏(MW)" REAL,
            "新能源负荷(MW)" REAL,
            "非市场化出力(MW)" REAL,
            "水电出力(MW)" REAL,
            "联络线计划(MW)" REAL,
            "在线机组容量(MW)" REAL,
            "日前出清价格(元/MWh)" REAL,
            "实时出清价格(元/MWh)" REAL,
            "负荷率(%)" REAL,
            extras_json TEXT,
            PRIMARY KEY ("日期", "时点", "边界数据类型")
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
    cur = conn.execute(f"SELECT MIN(\"{DB_COL_DATE}\"), MAX(\"{DB_COL_DATE}\") FROM marginal_data")
    row = cur.fetchone()
    return row[0], row[1]


def _format_epoch(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, SH_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _now_shanghai() -> datetime:
    return datetime.now(SH_TZ)


def format_time_shanghai(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, SH_TZ).strftime("%Y-%m-%d %H:%M:%S")
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(SH_TZ).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return text


def _normalize_date(val: Optional[Any]) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, str):
        stripped = val.strip()
        return stripped or None
    if isinstance(val, date):
        return val.strftime("%Y-%m-%d")
    return str(val)



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
        rows.append(
            (
                record.get('date'),
                record.get('time_slot'),
                record.get('boundary_type') or '',
                record.get('bidding_space_mw'),
                record.get('dispatch_load_mw'),
                record.get('wind_mw'),
                record.get('solar_mw'),
                record.get('new_energy_mw'),
                record.get('non_market_output_mw'),
                record.get('hydro_mw'),
                record.get('tie_line_plan_mw'),
                record.get('online_capacity_mw'),
                record.get('day_ahead_price'),
                record.get('real_time_price'),
                record.get('load_rate_pct'),
                extras_json,
            )
        )
    # 增量更新原始数据表：只删除新数据涉及的日期范围
    if rows:
        # 获取新数据的日期范围
        new_dates = list(set(row[0] for row in rows if row[0] is not None))
        if new_dates:
            date_placeholders = ",".join("?" * len(new_dates))
            conn.execute(
                f"DELETE FROM raw_marginal_data WHERE \"日期\" IN ({date_placeholders})",
                new_dates
            )

    conn.executemany(
        """
        INSERT INTO raw_marginal_data (
            "日期", "时点", "边界数据类型", "竞价空间(MW)", "省调负荷(MW)",
            "风电(MW)", "光伏(MW)", "新能源负荷(MW)", "非市场化出力(MW)", "水电出力(MW)",
            "联络线计划(MW)", "在线机组容量(MW)", "日前出清价格(元/MWh)", "实时出清价格(元/MWh)",
            "负荷率(%)", extras_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def get_dataset_status() -> Dict[str, Any]:
    with _get_conn() as conn:
        _ensure_db(conn)
        row_count = _count_rows(conn)
        meta = _get_meta_dict(conn)
        date_min, date_max = _get_date_span(conn)

    last_sync = format_time_shanghai(meta.get("last_sync_time") if meta else None)
    last_hash = meta.get("last_hash") if meta else None
    meta_row_count = meta.get("last_row_count") if meta else None
    last_row_count = int(meta_row_count) if meta_row_count else row_count
    data_dir = meta.get("last_data_dir") if meta else None
    data_source = meta.get("last_data_source") if meta else None
    raw_rows_meta = meta.get("last_raw_row_count") if meta else None
    raw_row_count = int(raw_rows_meta) if raw_rows_meta else None

    needs_sync = row_count == 0
    reason = "数据库为空，请先运行数据预处理" if needs_sync else None

    return {
        "row_count": row_count,
        "date_min": date_min,
        "date_max": date_max,
        "last_sync_time": last_sync,
        "last_hash": last_hash,
        "last_row_count": last_row_count,
        "needs_sync": needs_sync,
        "status_reason": reason,
        "state_label": "未导入" if needs_sync else "已同步",
        "data_dir": data_dir,
        "data_source": data_source,
        "raw_row_count": raw_row_count,
    }


def preprocess_dataset_and_sync(
    force: bool = False,
    data_dir: Optional[Union[str, Path]] = None,
    preprocessed_df: Optional[pd.DataFrame] = None,
    source_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Run scripts.pre_process (or use provided df) and persist the result into SQLite."""

    from scripts.pre_process import preprocess_data

    if preprocessed_df is not None:
        raw_df = preprocessed_df.copy()
        base_dir = Path(data_dir) if data_dir else Path("margin_data")
    else:
        base_dir = Path(data_dir) if data_dir else Path("margin_data")
        raw_df = preprocess_data(data_dir=base_dir, verbose=False)
    if raw_df is None:
        raise ValueError("预处理数据为空")
    if raw_df.empty:
        raise ValueError("预处理结果为空，无法导入")

    raw_df = raw_df.reset_index(drop=True)
    raw_prepared, raw_extra_cols = _prepare_raw_dataframe(raw_df.copy())

    cleaned = clean_marginal_data(raw_df.copy())
    cleaned = cleaned.sort_values(["date", "time_slot"]).reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("清洗后的数据为空，无法导入")

    fingerprint = _calc_dataframe_hash(cleaned)
    if source_label:
        source_dir = source_label
    else:
        source_dir = str(base_dir.resolve()) if base_dir.exists() else str(base_dir)
    source_type = source_label or ("preprocess_script" if preprocessed_df is None else "preprocessed_dataframe")

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
                "data_dir": source_dir,
            }

        # 增量更新：只删除新数据涉及的日期范围，避免清空整个数据库
        if not cleaned.empty:
            # 获取新数据的日期范围
            new_dates = cleaned["date"].unique()
            date_placeholders = ",".join("?" * len(new_dates))

            # 只删除新数据涉及日期的记录，实现增量更新
            conn.execute(
                f"DELETE FROM marginal_data WHERE \"{DB_COL_DATE}\" IN ({date_placeholders})",
                new_dates.tolist()
            )

            # 插入新数据
            conn.executemany(
                f"INSERT INTO marginal_data(\"{DB_COL_DATE}\", \"{DB_COL_TIME}\", \"{DB_COL_PRICE}\", \"{DB_COL_LOAD_RATE}\")"
                " VALUES (?, ?, ?, ?)",
                cleaned[["date", "time_slot", "price", "load_rate"]].itertuples(index=False, name=None),
            )
        _replace_raw_table(conn, raw_prepared, raw_extra_cols)

        # 重新计算数据库中的实际总行数（增量更新后）
        actual_total_rows = _count_rows(conn)
        actual_raw_rows = _count_raw_rows(conn)

        now_iso = _now_shanghai().strftime("%Y-%m-%d %H:%M:%S")
        _set_meta_bulk(
            conn,
            {
                "last_sync_time": now_iso,
                "last_hash": fingerprint,
                "last_row_count": actual_total_rows,  # 使用实际总行数
                "raw_table_ready": "1",
                "last_raw_row_count": actual_raw_rows,  # 使用实际原始数据行数
                "last_data_dir": source_dir,
                "last_data_source": source_type,
            },
        )

    cleaned.to_csv(DATA_PATH, index=False)

    return {
        "status": "updated",
        "rows": actual_total_rows,  # 返回数据库中的实际总行数
        "hash": fingerprint,
        "last_sync_time": now_iso,
        "data_dir": source_dir,
    }


def fetch_dataset(date_start: Optional[Any] = None, date_end: Optional[Any] = None) -> pd.DataFrame:
    start = _normalize_date(date_start)
    end = _normalize_date(date_end)
    with _get_conn() as conn:
        _ensure_db(conn)
        query = (
            f'SELECT "{DB_COL_DATE}" AS date, "{DB_COL_TIME}" AS time_slot, '
            f'"{DB_COL_PRICE}" AS price, "{DB_COL_LOAD_RATE}" AS load_rate '
            f"FROM marginal_data WHERE 1=1"
        )
        params: List[Any] = []
        if start:
            query += f' AND "{DB_COL_DATE}" >= ?'
            params.append(start)
        if end:
            query += f' AND "{DB_COL_DATE}" <= ?'
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
        "created_at": _now_shanghai().strftime("%Y-%m-%d %H:%M:%S"),
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
