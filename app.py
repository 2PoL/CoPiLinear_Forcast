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

MENU = st.sidebar.radio("导航", ["新边际数据导入", "日前价格预测", "数据展示", "关于"])

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
    x_col = st.text_input("负荷率列名称", "日前负荷率(%)")
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
            "time_slot": np.arange(2, len(preds) + 1),
            "price": preds,
        })
        st.line_chart(
            chart_df.set_index("time_slot"),
            height=300,
        )

# ---------------------------------------------------------------------------
# 3. Data Display tab
# ---------------------------------------------------------------------------
elif MENU == "数据展示":
    st.subheader("CSV数据展示与筛选")
    
    # Load default data
    if DATA_PATH.exists():
        st.info("数据来源: marginal.csv")
        original_df = pd.read_csv(DATA_PATH)
    else:
        st.warning("未找到数据文件 marginal.csv")
        st.stop()
    
    if original_df.empty:
        st.warning("数据文件为空")
        st.stop()
    
    # Display basic info
    st.write(f"**数据总行数:** {len(original_df):,}")
    st.write(f"**数据列数:** {len(original_df.columns)}")
    
    # Create filter section
    st.subheader("筛选条件")
    
    # Create three columns for filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**时间筛选**")
        
        # Try to detect date columns
        date_columns = []
        for col in original_df.columns:
            if 'date' in col.lower() or '日期' in col or '时间' in col:
                date_columns.append(col)
        
        date_filter_enabled = False
        start_date = None
        end_date = None
        date_col = None
        
        if date_columns:
            date_col = st.selectbox("选择日期列", date_columns)
            
            # Convert to datetime if possible
            try:
                temp_df = original_df.copy()
                temp_df[date_col] = pd.to_datetime(temp_df[date_col])
                
                # Date range filter
                min_date = temp_df[date_col].min().date()
                max_date = temp_df[date_col].max().date()
                
                start_date = st.date_input("开始日期", min_date, min_value=min_date, max_value=max_date)
                end_date = st.date_input("结束日期", max_date, min_value=min_date, max_value=max_date)
                date_filter_enabled = True
                
            except:
                st.warning(f"无法解析日期列 '{date_col}'")
        else:
            st.info("未检测到日期列")
    
    with col2:
        st.write("**条件筛选**")
        
        # Column selection for filtering
        filter_columns = st.multiselect("选择要筛选的列", original_df.columns.tolist())
        
        filters = {}
        for col in filter_columns:
            if original_df[col].dtype in ['object', 'string']:
                # For text columns, use multiselect
                unique_values = original_df[col].dropna().unique().tolist()
                if len(unique_values) <= 50:  # Only show if not too many unique values
                    selected_values = st.multiselect(f"筛选 {col}", unique_values, default=unique_values)
                    if selected_values != unique_values:  # Only add filter if changed
                        filters[col] = selected_values
                else:
                    # For columns with too many unique values, use text input
                    filter_text = st.text_input(f"筛选 {col} (包含文本)")
                    if filter_text:
                        filters[col] = filter_text
            else:
                # For numeric columns, use range slider
                try:
                    min_val = float(original_df[col].min())
                    max_val = float(original_df[col].max())
                    if min_val != max_val:
                        range_values = st.slider(
                            f"筛选 {col} 范围",
                            min_val, max_val, (min_val, max_val)
                        )
                        if range_values != (min_val, max_val):  # Only add filter if changed
                            filters[col] = range_values
                except:
                    pass
    
    with col3:
        st.write("**执行筛选**")
        st.write("")  # Add some spacing
        apply_filter = st.button("🔍 执行筛选", type="primary")
    
    # Apply filters only when button is clicked
    if apply_filter or 'filtered_df' not in st.session_state:
        df = original_df.copy()
        
        # Apply date filter
        if date_filter_enabled and date_col:
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                mask = (df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)
                df = df[mask]
            except:
                pass
        
        # Apply other filters
        for col, filter_value in filters.items():
            if isinstance(filter_value, list):
                if df[col].dtype in ['object', 'string']:
                    df = df[df[col].isin(filter_value)]
            elif isinstance(filter_value, str):
                df = df[df[col].astype(str).str.contains(filter_value, case=False, na=False)]
            elif isinstance(filter_value, tuple):
                df = df[(df[col] >= filter_value[0]) & (df[col] <= filter_value[1])]
        
        # Store filtered data in session state
        st.session_state.filtered_df = df
    else:
        # Use previously filtered data
        df = st.session_state.filtered_df if 'filtered_df' in st.session_state else original_df
    
    # Display filtered results
    st.subheader("筛选结果")
    st.write(f"**筛选后行数:** {len(df):,}")
    
    # Column selection for display
    display_columns = st.multiselect("选择要显示的列", df.columns.tolist(), default=df.columns.tolist()[:10])
    
    if display_columns:
        display_df = df[display_columns]
    else:
        display_df = df
    
    # Pagination
    rows_per_page = st.selectbox("每页显示行数", [10, 25, 50, 100, 500], index=2)
    
    if len(display_df) > 0:
        total_pages = (len(display_df) - 1) // rows_per_page + 1
        page = st.selectbox("页码", range(1, total_pages + 1))
        
        start_idx = (page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        
        st.dataframe(display_df.iloc[start_idx:end_idx], use_container_width=True)
        
        # Download filtered data
        csv = display_df.to_csv(index=False)
        st.download_button(
            "⬇️ 下载筛选后的数据",
            csv,
            file_name="filtered_data.csv",
            mime="text/csv"
        )
        
        # Basic statistics for numeric columns
        numeric_cols = display_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.subheader("数值列统计")
            st.dataframe(display_df[numeric_cols].describe())
            
            # Simple charts
            if len(numeric_cols) >= 1:
                st.subheader("数据可视化")
                chart_col = st.selectbox("选择要绘制的数值列", numeric_cols)
                chart_type = st.selectbox("图表类型", ["折线图", "柱状图", "直方图"])
                
                # Try to find a suitable x-axis column (prioritize time_slot, then date)
                x_axis_col = None
                if 'time_slot' in display_df.columns:
                    x_axis_col = 'time_slot'
                elif date_col and date_col in display_df.columns:
                    x_axis_col = date_col
                
                if chart_type == "折线图":
                    if x_axis_col:
                        # Create a proper chart with time_slot or date on x-axis
                        chart_df = display_df[[x_axis_col, chart_col]].copy().dropna()
                        if x_axis_col == date_col:
                            try:
                                chart_df[x_axis_col] = pd.to_datetime(chart_df[x_axis_col])
                            except:
                                pass
                        # Sort by x-axis for proper time series display
                        chart_df = chart_df.sort_values(x_axis_col)
                        st.line_chart(chart_df.set_index(x_axis_col))
                    else:
                        # Use row index as x-axis
                        chart_df = display_df[chart_col].reset_index()
                        chart_df.columns = ['序号', chart_col]
                        st.line_chart(chart_df.set_index('序号'))
                        
                elif chart_type == "柱状图":
                    if x_axis_col and len(display_df) <= 100:  # Limit for readability
                        chart_df = display_df[[x_axis_col, chart_col]].copy().dropna()
                        if x_axis_col == date_col:
                            try:
                                chart_df[x_axis_col] = pd.to_datetime(chart_df[x_axis_col])
                            except:
                                pass
                        # Sort by x-axis for proper time series display
                        chart_df = chart_df.sort_values(x_axis_col)
                        st.bar_chart(chart_df.set_index(x_axis_col))
                    else:
                        # Use row index as x-axis, limit to first 50 rows for readability
                        limited_df = display_df[chart_col].head(50).reset_index()
                        limited_df.columns = ['序号', chart_col]
                        st.bar_chart(limited_df.set_index('序号'))
                        if len(display_df) > 50:
                            st.info("为了图表可读性，仅显示前50行数据")
                            
                elif chart_type == "直方图":
                    # Show distribution of values
                    value_counts = display_df[chart_col].value_counts().head(20)
                    st.bar_chart(value_counts)
                
                # Show which column is being used as x-axis
                if x_axis_col:
                    st.info(f"横轴使用列: {x_axis_col}")
    else:
        st.warning("没有数据可显示")

# ---------------------------------------------------------------------------
# 4. About tab
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
