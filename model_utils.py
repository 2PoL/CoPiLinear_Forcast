# ---------------------------------------------------------------------------
# model_utils.py – shared business logic for both modules
# ---------------------------------------------------------------------------

from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import pwlf
import pickle
from scipy import stats

# Application paths
DATA_PATH = Path("data/marginal.csv")
MODEL_PATH = Path("models/2025_marginal.pkl")
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------
# Data cleaning (mirrors reappropriation.py)
# -------------------------------------------

def clean_marginal_data(df: pd.DataFrame, iqr_factor: float = 3.0) -> pd.DataFrame:
    """Standardise column names, coerce dtypes, drop NaNs & outliers."""
    # Flexible rename map – extend if needed

    df.columns = df.columns.str.strip()

    # 2) 列名映射 (可按需要扩充)
    rename_map = {
        '日期': 'date',
        'date': 'date',

        '时点': 'time_slot',
        '时段': 'time_slot',
        'time_slot': 'time_slot',

        '(调控后)日前出清价格(元/MWh)': 'price',
        '价格': 'price',
        'price': 'price',

        '日前负荷率(%)': 'load_rate',
        '负荷率': 'load_rate',
        'load_rate': 'load_rate',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ['date', 'time_slot', 'price', 'load_rate']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'以下列缺失或列名不匹配: {missing}\n当前列: {list(df.columns)}')

    # 3) 类型转换
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['load_rate'] = pd.to_numeric(df['load_rate'], errors='coerce')

    df = df.dropna(subset=['price', 'load_rate'])

    # 4) IQR 去离群：仅对非零价格做 IQR，保留正常的 0 价格行
    mask_zero_price = df["price"] == 0
    price_for_iqr = df.loc[~mask_zero_price, "price"]
    if len(price_for_iqr) >= 4:
        q1, q3 = price_for_iqr.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
        keep_mask = mask_zero_price | ((df["price"] >= lower) & (df["price"] <= upper))
        df = df[keep_mask].reset_index(drop=True)
    else:
        # 样本数量过少，不做 IQR 过滤
        df = df.reset_index(drop=True)
    return df

# ------------------------------------
# Model training & management
# ------------------------------------

def train_model(
    df: pd.DataFrame,
    model_name: str,
    *,
    n_segments: int = 3,
    iqr_factor: float = 3.0,
) -> Tuple[pwlf.PiecewiseLinFit, np.ndarray]:
    """Train pwlf model on provided DataFrame and save to models/{model_name}.pkl

    The input DataFrame is cleaned via clean_marginal_data.
    Returns (model, breakpoints).
    """
    cleaned = clean_marginal_data(df.copy(), iqr_factor=iqr_factor)
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
# Extended model management (artifacts + metadata)
# -------------------------------------------------
from typing import Dict, Any  # reuse typing, safe to re-import names
from datetime import datetime
import json

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
    iqr_factor: float = 1.5,
    source: Optional[str] = None,
) -> Tuple[pwlf.PiecewiseLinFit, np.ndarray]:
    """Train + save model, metadata, and cleaned training data.

    This does not modify legacy train_model; callers should use this function
    to ensure artifacts exist for management UI.
    """
    cleaned = clean_marginal_data(df.copy(), iqr_factor=iqr_factor)
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
        "iqr_factor": iqr_factor,
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
