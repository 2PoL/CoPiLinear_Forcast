import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import pwlf
from model_utils import (
    clean_marginal_data,
    incremental_retrain,
    load_or_train_model,
    predict_price,
    DATA_PATH,
    MODEL_PATH,
)

st.set_page_config(page_title="", layout="wide")

st.title("CoPiLinear模型日前价格预测")

MENU = st.sidebar.radio("导航", ["新边际数据导入", "日前价格预测", "关于"])

# ---------------------------------------------------------------------------
# 1. Retrain model tab
# ---------------------------------------------------------------------------
if MENU == "新边际数据导入":
    st.subheader("上传最新的边际数据")

    file = st.file_uploader("选择 CSV/XLSX 文件", type=["csv", "xlsx"])

    n_segments = st.slider("pwlf 拟合段数", min_value=2, max_value=8, value=3)
    iqr_factor = st.slider("IQR outlier factor (e.g. 1.5 = Tukey fence)", 0.5, 5.0, 1.5, 0.1)

    if st.button("🚀 开始增量训练"):
        if not file:
            st.warning("请先上传文件")
            st.stop()
        # -------------------------------------------------------------------
        # Load & clean new data
        # -------------------------------------------------------------------
        raw_df = (
            pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
        )
        st.write(f"原始数据: **{len(raw_df):,}**")
        cleaned_df = clean_marginal_data(raw_df.copy(), iqr_factor=iqr_factor)
        st.write(f"清理后的原始数据: **{len(cleaned_df):,}**")

        # -------------------------------------------------------------------
        # Incremental retraining
        # -------------------------------------------------------------------
        model, breakpoints, combined = incremental_retrain(
            cleaned_df, n_segments=n_segments
        )
        # Show **only** rows that were removed by cleaning
        removed = raw_df.loc[raw_df.index.difference(cleaned_df.index)]
        if removed.empty:
            st.success("No rows were removed as outliers or NaNs.")
        else:
            st.warning(f"Removed rows: {len(removed)}")
            st.dataframe(removed.head(20))

        # Continue pipeline with cleaned data
        model, breakpoints, combined = incremental_retrain(
            cleaned_df, n_segments=n_segments, iqr_factor=iqr_factor
        )
        st.success(
            f"模型重新训练完成，共 **{len(combined):,}** 条记录。断点: {np.round(breakpoints, 4)}"
        )
        if "date" in combined.columns:
            start, end = combined["date"].min(), combined["date"].max()
            st.info(f"模型训练数据区间：**{start}** — **{end}**")
        # Allow download of updated model & dataset
        with open(MODEL_PATH, "rb") as f:
            st.download_button("⬇️ 下载新模型", f, file_name="pwlf_model.pkl")
        st.download_button(
            "⬇️ 下载合并后的边际数据 (CSV)",
            combined.to_csv(index=False),
            file_name="marginal_merged.csv",
        )

# ---------------------------------------------------------------------------
# 2. Forecast tab (implements *forecast* module)
# ---------------------------------------------------------------------------
elif MENU == "日前价格预测":
    st.subheader("根据负荷率预测日前价格预测")
    # --- 显示当前 marginal.csv 的日期范围 ---
    if DATA_PATH.exists():
        meta_df = pd.read_csv(DATA_PATH, parse_dates=["date"], usecols=["date"], nrows=100000)
        if "date" in meta_df.columns and not meta_df["date"].isna().all():
            start, end = meta_df["date"].min(), meta_df["date"].max()
            st.info(f"当前CoPiLinear模型包含日期范围：**{start.date()}** — **{end.date()}**")
        else:
            st.warning("数据源中缺少可解析的 date 列。")
    else:
        st.warning("未找到数据源，请先在 Retrain 页上传数据并训练模型。")

    model = load_or_train_model()
    if model is None:
        st.error("未找到模型。")
        st.stop()

    # Input options: manual values or batch file
    # mode = st.radio("Prediction mode", ["Manual input", "Batch CSV/XLSX"])

    # if mode == "Manual input":
    #     x_vals = st.text_input(
    #         "Enter one or more load‑rate values (0‑1) separated by commas", "0.55, 0.78, 0.93"
    #     )
    #     if st.button("Predict"):
    #         try:
    #             xs = np.array([float(v) for v in x_vals.split(",")])
    #         except ValueError:
    #             st.error("Invalid numeric input.")
    #             st.stop()
    #         preds = predict_price(model, xs)
    #         st.write(pd.DataFrame({"load_rate": xs, "predicted_price": preds}))
    #
    # else:
    file = st.file_uploader("上传预测标的日负荷率数据 CSV/XLSX", type=["csv", "xlsx"])
    x_col = st.text_input("负荷率列名称", "load_rate")
    if st.button("预测") and file:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        if x_col not in df.columns:
            st.error(f"找不到 '{x_col}'")
            st.stop()
        preds = predict_price(model, df[x_col].values)
        out_df = df.copy()
        out_df["日前预测价格"] = preds
        st.dataframe(out_df.head())
        st.download_button(
            "⬇️ 下载预测",
            out_df.to_csv(index=False).encode(),
            file_name="price_predictions.csv",
        )
        chart_df = pd.DataFrame({
            "time_slot": np.arange(1, len(preds) + 1),
            "price": preds,
        })
        st.line_chart(
            chart_df.set_index("time_slot"),
            height=300,
        )

# ---------------------------------------------------------------------------
# 3. About tab
# ---------------------------------------------------------------------------
else:
    st.markdown(
        """
### About this tool
This Streamlit application wraps two once‑separate command‑line pipelines:
1. **reappropriation** – incremental cleaning & retraining of a piece‑wise linear (pwlf) model.
2. **forecast** – price prediction from new load‑rate values.

The entire workflow is now point‑and‑click but conserves the underlying Python logic (see `model_utils.py`).
        """
    )

# ---------------------------------------------------------------------------
# End of app.py
# ---------------------------------------------------------------------------