from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd

from model_utils import predict_price


SHEET_NAME = "24点区间"
LOAD_RATE_COLUMN = "负荷率(%)"
VALUE_LABEL = "预测值"
MIN_LABEL = "最小值"
MAX_LABEL = "最大值"


def _normalize_header(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("\xa0", "")
        .replace("\u3000", "")
        .replace("（", "(")
        .replace("）", ")")
        .strip()
    )


def _first_matching_column(columns: Iterable[Any], top_name: str, sub_name: Optional[str] = None) -> Any:
    normalized_top = _normalize_header(top_name)
    normalized_sub = _normalize_header(sub_name) if sub_name is not None else None
    for col in columns:
        if isinstance(col, tuple):
            top = _normalize_header(col[0])
            sub = _normalize_header(col[1]) if len(col) > 1 else ""
            if top == normalized_top and (normalized_sub is None or sub == normalized_sub):
                return col
        elif _normalize_header(col) == normalized_top and normalized_sub is None:
            return col
    raise ValueError(f"未找到列：{top_name}" + (f" / {sub_name}" if sub_name else ""))


def _model_breakpoints(model: Any) -> np.ndarray:
    for attr in ("fit_breaks", "breaks"):
        values = getattr(model, attr, None)
        if values is not None:
            return np.asarray(values, dtype=float)
    return np.array([], dtype=float)


def _predict_interval(model: Any, low: float, high: float) -> tuple[float, float]:
    if np.isnan(low) or np.isnan(high):
        return np.nan, np.nan
    left, right = sorted((float(low), float(high)))
    breakpoints = _model_breakpoints(model)
    candidates = [left, right]
    if breakpoints.size:
        candidates.extend(breakpoints[(breakpoints >= left) & (breakpoints <= right)].tolist())
    prices = predict_price(model, np.asarray(candidates, dtype=float))
    return float(np.nanmin(prices)), float(np.nanmax(prices))


def read_daily_rolling_template(
    file: Union[str, Path, Any],
    *,
    sheet_name: str = SHEET_NAME,
) -> pd.DataFrame:
    """Read the 24-hour interval sheet and normalize load-rate interval columns."""

    try:
        raw = pd.read_excel(file, sheet_name=sheet_name, header=[0, 1])
    except ValueError as exc:
        raise ValueError(f"模板中未找到「{sheet_name}」sheet") from exc

    date_col = _first_matching_column(raw.columns, "日期")
    period_col = _first_matching_column(raw.columns, "时段")
    load_value_col = _first_matching_column(raw.columns, LOAD_RATE_COLUMN, VALUE_LABEL)
    load_min_col = _first_matching_column(raw.columns, LOAD_RATE_COLUMN, MIN_LABEL)
    load_max_col = _first_matching_column(raw.columns, LOAD_RATE_COLUMN, MAX_LABEL)

    df = pd.DataFrame(
        {
            "日期": pd.to_datetime(raw[date_col], errors="coerce").dt.strftime("%Y-%m-%d"),
            "时段": raw[period_col].astype(str).str.strip(),
            "负荷率预测值(%)": pd.to_numeric(raw[load_value_col], errors="coerce"),
            "负荷率最小值(%)": pd.to_numeric(raw[load_min_col], errors="coerce"),
            "负荷率最大值(%)": pd.to_numeric(raw[load_max_col], errors="coerce"),
        }
    )
    df = df.dropna(subset=["日期", "时段", "负荷率最小值(%)", "负荷率最大值(%)"])
    if df.empty:
        raise ValueError("「24点区间」sheet 中没有可预测的负荷率区间数据")
    return df.reset_index(drop=True)


def predict_daily_rolling_price_interval(
    file: Union[str, Path, Any],
    model: Any,
    *,
    model_name: str,
    sheet_name: str = SHEET_NAME,
) -> pd.DataFrame:
    """Predict price ranges from load-rate ranges in the daily rolling template."""

    df = read_daily_rolling_template(file, sheet_name=sheet_name)
    load_value = df["负荷率预测值(%)"].to_numpy(dtype=float)
    price_value = predict_price(model, load_value)

    intervals = [
        _predict_interval(model, low, high)
        for low, high in zip(df["负荷率最小值(%)"], df["负荷率最大值(%)"])
    ]
    interval_df = pd.DataFrame(intervals, columns=["预测价格最小值(元/MWh)", "预测价格最大值(元/MWh)"])

    result = pd.concat([df, interval_df], axis=1)
    result.insert(2, "模型", model_name)
    result["预测价格(元/MWh)"] = price_value
    ordered_cols = [
        "日期",
        "时段",
        "模型",
        "负荷率预测值(%)",
        "负荷率最小值(%)",
        "负荷率最大值(%)",
        "预测价格(元/MWh)",
        "预测价格最小值(元/MWh)",
        "预测价格最大值(元/MWh)",
    ]
    return result[ordered_cols]
