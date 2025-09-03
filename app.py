import io
import zipfile

import streamlit as st
import pandas as pd
import numpy as np

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
# 3) 多模型日前价格预测
# ----------------------------------------
elif MENU == "日前价格预测":
    st.subheader("根据负荷率预测日前价格（可多模型对比）")

    models = list_models()
    if not models:
        st.warning("暂无可用模型，请先在‘模型训练’页训练并保存模型。")
        st.stop()
    selected_models = st.multiselect("选择模型", models, default=models[:1])
    if not selected_models:
        st.warning("请至少选择一个模型")
        st.stop()

    file = st.file_uploader("上传含负荷率列的 CSV/XLSX", type=["csv", "xlsx"])
    x_col = st.text_input("负荷率列名称", "日前负荷率(%)")
    if st.button("预测") and file:
        df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
        if x_col not in df.columns:
            st.error(f"未找到列：'{x_col}'")
            st.stop()
        x_vals = df[x_col].values
        preds_map = {}
        for name in selected_models:
            m = load_model(name)
            if m is None:
                st.warning(f"模型 '{name}' 加载失败，已跳过")
                continue
            preds_map[name] = predict_price(m, x_vals)
        if not preds_map:
            st.error("没有可用预测结果")
            st.stop()
        time_slot = np.arange(1, len(x_vals) + 1)
        wide_df = pd.DataFrame({"time_slot": time_slot})
        for name, arr in preds_map.items():
            wide_df[name] = arr
        st.dataframe(wide_df.head())
        st.download_button(
            "下载预测结果（多模型）",
            wide_df.to_csv(index=False).encode(),
            file_name="price_predictions.csv",
        )
        st.line_chart(wide_df.set_index("time_slot"), height=300)


else:
    st.stop()

