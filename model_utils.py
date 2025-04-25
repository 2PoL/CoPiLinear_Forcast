# ---------------------------------------------------------------------------
# model_utils.py – shared business logic for both modules
# ---------------------------------------------------------------------------

from pathlib import Path
from typing import Tuple, Optional
import warnings

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

    # 4) IQR 去离群
    q1, q3 = df['price'].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
    df = df[(df['price'] >= lower) & (df['price'] <= upper)].reset_index(drop=True)
    return df

# ---------------------------------------------------
# Incremental retrain (core of reappropriation.py)
# ---------------------------------------------------

def incremental_retrain(
    new_df: pd.DataFrame,
    *,
    n_segments: int = 3,
    iqr_factor: float = 3.0,
) -> Tuple[pwlf.PiecewiseLinFit, np.ndarray, pd.DataFrame]:
    """Merge new data, retrain pwlf, save both dataset & model."""

    # Clean *again* in case caller skipped
    new_df = clean_marginal_data(new_df, iqr_factor=iqr_factor)

    # Merge with existing
    if DATA_PATH.exists():
        base = pd.read_csv(DATA_PATH)
        combined = pd.concat([base, new_df]).drop_duplicates(
            subset=["date", "time_slot", "load_rate", "price"], keep="last"
        )
    else:
        combined = new_df.copy()

    # Persist merged dataset
    combined.to_csv(DATA_PATH,index=False)

    # Train pwlf model
    x = combined["load_rate"].values
    y = combined["price"].values

    if len(x) < n_segments + 2:
        warnings.warn(
            "Sample size too small for requested segments; reducing segment count."
        )
        n_segments = max(1, len(x) - 2)

    model = pwlf.PiecewiseLinFit(x, y)
    breakpoints = model.fit(n_segments)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model, breakpoints, combined

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
    return model.predict(x)