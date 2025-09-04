import io
import zipfile

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from model_utils import (
    list_models,
    load_model,
    predict_price,
    train_model_ex,
    load_model_meta,
    delete_models,
)


st.set_page_config(page_title="CoPiLinear 工具", layout="wide")
st.title("CoPiLinear 模型工具")

MENU = st.sidebar.radio("导航", ["模型管理", "模型训练", "日前价格预测"])


# ----------------------------------------
# 1) 模型管理
# ----------------------------------------
if MENU == "模型管理":
    st.subheader("模型管理")
    models = list_models()
    if not models:
        st.info("暂无模型，请先到‘模型训练’页创建模型。")
    else:
        selected = st.multiselect("选择模型（用于批量操作）", models)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("删除所选模型"):
                if not selected:
                    st.warning("请先选择至少一个模型")
                else:
                    summary = delete_models(selected)
                    removed = summary.get('removed', [])
                    missing = summary.get('missing', [])
                    if removed:
                        st.success(f"已删除: {', '.join(removed)}")
                    if missing:
                        st.info(f"未找到: {', '.join(missing)}")
        with col_b:
            if st.button("打包下载所选模型"):
                if not selected:
                    st.warning("请先选择至少一个模型")
                else:
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                        for name in selected:
                            for ext in (".pkl", ".meta.json", ".train.csv"):
                                path = f"models/{name}{ext}"
                                try:
                                    with open(path, "rb") as f:
                                        zf.writestr(f"{name}/{name}{ext}", f.read())
                                except Exception:
                                    pass
                    st.download_button(
                        "下载ZIP",
                        data=buf.getvalue(),
                        file_name="models_selected.zip",
                        mime="application/zip",
                    )

        st.markdown("---")
        st.write("模型列表与详情：")
        for name in models:
            with st.expander(name, expanded=False):
                meta = load_model_meta(name)
                if meta:
                    st.write(f"创建时间: {meta.get('created_at')}")
                    st.write(f"分段数: {meta.get('n_segments')}  IQR因子: {meta.get('iqr_factor')}")
                    bps = meta.get('breakpoints')
                    if bps is not None:
                        st.write(f"断点: {np.round(np.array(bps), 4)}")
                    st.write(f"训练数据来源: {meta.get('source')}")
                else:
                    st.info("缺少元数据（meta），仅提供模型文件操作。")

                c1, c2, c3 = st.columns(3)
                with c1:
                    try:
                        with open(f"models/{name}.pkl", "rb") as f:
                            st.download_button("下载模型(.pkl)", f, file_name=f"{name}.pkl")
                    except Exception:
                        st.warning("模型文件缺失")
                with c2:
                    try:
                        with open(f"models/{name}.train.csv", "rb") as f:
                            st.download_button("下载训练数据", f, file_name=f"{name}.train.csv")
                    except Exception:
                        st.info("无训练数据文件")
                with c3:
                    if st.button(f"删除 {name}"):
                        delete_models([name])
                        st.success(f"已删除 {name}")

                # 训练数据预览
                try:
                    train_df = pd.read_csv(f"models/{name}.train.csv")
                    st.write(f"训练数据行数: {len(train_df):,}")
                    st.dataframe(train_df.head(10))
                except Exception:
                    pass


# ----------------------------------------
# 2) 模型训练
# ----------------------------------------
elif MENU == "模型训练":
    st.subheader("模型训练")
    file = st.file_uploader("上传训练用 Excel", type=["xlsx", "csv"])
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
            model, breakpoints = train_model_ex(
                df,
                model_name=model_name,
                n_segments=n_segments,
                iqr_factor=iqr_factor,
                source=file.name,
            )
            st.success(f"训练完成：{model_name}，断点：{np.round(breakpoints, 4)}")
        except Exception as e:
            st.error(f"训练失败：{e}")


# ----------------------------------------
# 3) 日前价格预测
# ----------------------------------------
elif MENU == "日前价格预测":
    st.subheader("多模型日前价格预测")

    models = list_models()
    if not models:
        st.warning("暂无可用模型，请先在‘模型训练’页训练并保存模型。")
        st.stop()
    selected_models = st.multiselect("选择模型", models, default=models[:1])
    if not selected_models:
        st.warning("请至少选择一个模型")
        st.stop()

    capacity = st.number_input("日前在线机组容量 (MW)", min_value=0.0, value=1000.0, step=10.0)
    file = st.file_uploader("上传包含‘竞价空间’列的 CSV/XLSX", type=["csv", "xlsx"])
    if st.button("预测") and file:
        df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
        if capacity <= 0:
            st.error("容量必须大于 0 MW")
            st.stop()
        # 自动识别竞价空间列
        space_col = None
        for cand in ["竞价空间(MW)", "竞价空间"]:
            if cand in df.columns:
                space_col = cand
                break
        if space_col is None:
            fuzzy = [c for c in df.columns if ("竞价" in str(c)) and ("空间" in str(c))]
            if fuzzy:
                space_col = fuzzy[0]
        if space_col is None:
            st.error("未找到包含‘竞价空间’的列")
            st.stop()

        space_vals = pd.to_numeric(df[space_col], errors="coerce").values
        if np.isnan(space_vals).all():
            st.error("‘竞价空间’列无法解析为数值，请检查数据")
            st.stop()

        # 负荷率(%) = 竞价空间 / 日前在线机组容量(MW) * 100
        load_rate = (space_vals / capacity) * 100.0

        preds_map = {}
        for name in selected_models:
            m = load_model(name)
            if m is None:
                st.warning(f"模型 '{name}' 加载失败，已跳过")
                continue
            preds_map[name] = predict_price(m, load_rate)
        if not preds_map:
            st.error("没有可用预测结果")
            st.stop()

        # 生成 15 分钟粒度时间标签：00:15, 00:30, ...（若为 96 条则最后为 24:00）
        _mins = np.arange(1, len(load_rate) + 1) * 15
        _hrs = _mins // 60
        _mm = _mins % 60
        time_labels = [f"{int(h):02d}:{int(m):02d}" for h, m in zip(_hrs, _mm)]

        # 表格显示：包括容量与负荷率
        display_df = pd.DataFrame({
            "time_slot": time_labels,
            "日前在线机组容量 (MW)": np.full(len(time_labels), capacity, dtype=float),
            "负荷率": load_rate,
        })
        for name, arr in preds_map.items():
            display_df[name] = arr
        st.dataframe(display_df.head())

        # 下载结果：包含 time_slot、容量、负荷率、各模型预测价格
        download_df = display_df.copy()
        st.download_button(
            "下载预测结果（多模型）",
            download_df.to_csv(index=False).encode(),
            file_name="price_predictions.csv",
        )

        # 图表：仅显示各模型预测，不显示容量与负荷率；Y 轴锁定 0-1500，X 轴包含 24:00
        model_cols = [c for c in download_df.columns if c not in ["time_slot", "日前在线机组容量 (MW)", "负荷率"]]
        chart_df = download_df[["time_slot", *model_cols]].melt('time_slot', var_name='model', value_name='price')
        hour_ticks = [t for t in time_labels if t.endswith(':00')]
        if '24:00' in time_labels and '24:00' not in hour_ticks:
            hour_ticks.append('24:00')
        chart = (
            alt.Chart(chart_df)
            .mark_line()
            .encode(
                x=alt.X('time_slot:N', sort=None, scale=alt.Scale(domain=time_labels), axis=alt.Axis(values=hour_ticks, title='time_slot')),
                y=alt.Y('price:Q', scale=alt.Scale(domain=[0, 1500]), title='price'),
                color=alt.Color('model:N', title='model')
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)


else:
    st.stop()

