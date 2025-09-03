import streamlit as st
import pandas as pd
import numpy as np
from model_utils import (
    DATA_PATH,
    train_model,
    list_models,
    load_model,
    predict_price,
)

st.set_page_config(page_title="", layout="wide")

st.title("CoPiLinear 模型工具")

MENU = st.sidebar.radio("导航", ["模型训练", "日前价格预测", "数据展示", "关于"])

# ----------------------------------------
# 1) 训练页面
# ----------------------------------------
if MENU == "模型训练":
    st.subheader("模型训练")
    file = st.file_uploader("上传训练用 Excel 或 CSV", type=["xlsx", "csv"])
    model_name = st.text_input("模型名称", value="model_v1").strip()
    n_segments = st.slider("分段数 (pwlf)", min_value=2, max_value=8, value=3)
    iqr_factor = st.slider("IQR 去极值因子", 0.5, 5.0, 1.5, 0.1)

    if st.button("训练模型"):
        if not file:
            st.warning("请先上传训练数据文件")
            st.stop()
        if not model_name:
            st.warning("请填写模型名称")
            st.stop()
        df = pd.read_excel(file) if file.name.lower().endswith(".xlsx") else pd.read_csv(file)
        try:
            model, breakpoints = train_model(
                df,
                model_name=model_name,
                n_segments=n_segments,
                iqr_factor=iqr_factor,
            )
            st.success(f"训练完成：{model_name}，断点：{np.round(breakpoints, 4)}")
        except Exception as e:
            st.error(f"训练失败：{e}")

# ----------------------------------------
# 2) 预测页面
# ----------------------------------------
elif MENU == "日前价格预测":
    st.subheader("根据负荷率预测日前价格")

    # 选择模型
    models = list_models()
    if not models:
        st.warning("暂无可用模型，请先在‘模型训练’页训练并保存模型。")
        st.stop()
    chosen = st.selectbox("选择模型", models)
    model = load_model(chosen)
    if model is None:
        st.error("模型加载失败")
        st.stop()

    # 上传预测数据
    file = st.file_uploader("上传含负荷率列的 CSV/XLSX", type=["csv", "xlsx"])
    x_col = st.text_input("负荷率列名称", "load_rate")
    if st.button("预测") and file:
        df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
        if x_col not in df.columns:
            st.error(f"未找到列：'{x_col}'")
            st.stop()
        preds = predict_price(model, df[x_col].values)
        out_df = df.copy()
        out_df["predicted_price"] = preds
        st.dataframe(out_df.head())
        st.download_button(
            "下载预测结果",
            out_df.to_csv(index=False).encode(),
            file_name="price_predictions.csv",
        )
        chart_df = pd.DataFrame({
            "time_slot": np.arange(1, len(preds) + 1),
            "price": preds,
        })
        st.line_chart(chart_df.set_index("time_slot"), height=300)

# ----------------------------------------
# 3) 数据展示
# ----------------------------------------
elif MENU == "数据展示":
    st.subheader("CSV 数据展示与筛选")

    if DATA_PATH.exists():
        st.info("数据来源: data/marginal.csv")
        original_df = pd.read_csv(DATA_PATH)
    else:
        st.warning("未找到数据文件 data/marginal.csv")
        st.stop()

    if original_df.empty:
        st.warning("数据文件为空")
        st.stop()

    st.write(f"数据总行数: {len(original_df):,}")
    st.write(f"数据列数: {len(original_df.columns)}")

    st.subheader("筛选条件")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("时间筛选")
        date_columns = [c for c in original_df.columns if ('date' in c.lower()) or ('日期' in c) or ('时间' in c)]
        date_filter_enabled = False
        start_date = end_date = None
        date_col = None
        if date_columns:
            date_col = st.selectbox("选择日期列", date_columns)
            try:
                temp_df = original_df.copy()
                temp_df[date_col] = pd.to_datetime(temp_df[date_col])
                min_date = temp_df[date_col].min().date()
                max_date = temp_df[date_col].max().date()
                start_date = st.date_input("开始日期", min_date, min_value=min_date, max_value=max_date)
                end_date = st.date_input("结束日期", max_date, min_value=min_date, max_value=max_date)
                date_filter_enabled = True
            except Exception:
                st.warning(f"无法解析日期列 '{date_col}'")
        else:
            st.info("未检测到日期列")

    with col2:
        st.write("条件筛选")
        filter_columns = st.multiselect("选择要筛选的列", original_df.columns.tolist())
        filters = {}
        for col in filter_columns:
            if original_df[col].dtype in ['object', 'string']:
                unique_values = original_df[col].dropna().unique().tolist()
                if len(unique_values) <= 50:
                    selected = st.multiselect(f"筛选 {col}", unique_values, default=unique_values)
                    if selected != unique_values:
                        filters[col] = selected
                else:
                    text = st.text_input(f"筛选 {col} (包含文本)")
                    if text:
                        filters[col] = text
            else:
                try:
                    min_val = float(original_df[col].min())
                    max_val = float(original_df[col].max())
                    if min_val != max_val:
                        rng = st.slider(f"筛选 {col} 范围", min_val, max_val, (min_val, max_val))
                        if rng != (min_val, max_val):
                            filters[col] = rng
                except Exception:
                    pass

    with col3:
        st.write("执行筛选")
        apply_filter = st.button("执行")

    if apply_filter or 'filtered_df' not in st.session_state:
        df = original_df.copy()
        if date_filter_enabled and date_col:
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                mask = (df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)
                df = df[mask]
            except Exception:
                pass
        for col, val in filters.items():
            if isinstance(val, list):
                if df[col].dtype in ['object', 'string']:
                    df = df[df[col].isin(val)]
            elif isinstance(val, str):
                df = df[df[col].astype(str).str.contains(val, case=False, na=False)]
            elif isinstance(val, tuple):
                df = df[(df[col] >= val[0]) & (df[col] <= val[1])]
        st.session_state.filtered_df = df
    else:
        df = st.session_state.filtered_df if 'filtered_df' in st.session_state else original_df

    st.subheader("筛选结果")
    st.write(f"筛选后行数: {len(df):,}")
    display_columns = st.multiselect("选择要显示的列", df.columns.tolist(), default=df.columns.tolist()[:10])
    display_df = df[display_columns] if display_columns else df

    rows_per_page = st.selectbox("每页显示行数", [10, 25, 50, 100, 500], index=2)
    if len(display_df) > 0:
        total_pages = (len(display_df) - 1) // rows_per_page + 1
        page = st.selectbox("页码", range(1, total_pages + 1))
        start_idx = (page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        st.dataframe(display_df.iloc[start_idx:end_idx], use_container_width=True)

        csv = display_df.to_csv(index=False)
        st.download_button("下载筛选后的数据", csv, file_name="filtered_data.csv", mime="text/csv")

        numeric_cols = display_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.subheader("数值列统计")
            st.dataframe(display_df[numeric_cols].describe())

            st.subheader("数据可视化")
            chart_col = st.selectbox("选择要绘制的数值列", numeric_cols)
            chart_type = st.selectbox("图表类型", ["折线", "柱状", "直方"]) 
            x_axis_col = None
            if 'time_slot' in display_df.columns:
                x_axis_col = 'time_slot'
            elif date_col and date_col in display_df.columns:
                x_axis_col = date_col

            if chart_type == "折线":
                if x_axis_col:
                    chart_df = display_df[[x_axis_col, chart_col]].copy().dropna()
                    if x_axis_col == date_col:
                        try:
                            chart_df[x_axis_col] = pd.to_datetime(chart_df[x_axis_col])
                        except Exception:
                            pass
                    chart_df = chart_df.sort_values(x_axis_col)
                    st.line_chart(chart_df.set_index(x_axis_col))
                else:
                    chart_df = display_df[chart_col].reset_index()
                    chart_df.columns = ['序号', chart_col]
                    st.line_chart(chart_df.set_index('序号'))

            elif chart_type == "柱状":
                if x_axis_col and len(display_df) <= 100:
                    chart_df = display_df[[x_axis_col, chart_col]].copy().dropna()
                    if x_axis_col == date_col:
                        try:
                            chart_df[x_axis_col] = pd.to_datetime(chart_df[x_axis_col])
                        except Exception:
                            pass
                    chart_df = chart_df.sort_values(x_axis_col)
                    st.bar_chart(chart_df.set_index(x_axis_col))
                else:
                    limited_df = display_df[chart_col].head(50).reset_index()
                    limited_df.columns = ['序号', chart_col]
                    st.bar_chart(limited_df.set_index('序号'))

            elif chart_type == "直方":
                value_counts = display_df[chart_col].value_counts().head(20)
                st.bar_chart(value_counts)

# ----------------------------------------
# 4) 关于
# ----------------------------------------
else:
    st.markdown(
        """
### 关于
本应用提供：
1. 模型训练（上传Excel/CSV，训练并保存模型）。
2. 日前价格预测（选择已训练模型进行预测）。
3. 数据浏览和筛选（展示 data/marginal.csv）。

底层逻辑见 model_utils.py。
        """
    )

