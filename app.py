import io
import os
import zipfile
from datetime import date, timedelta
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from model_utils import (
    list_models,
    load_model,
    predict_price,
    clean_marginal_data,
    train_model_ex,
    load_model_meta,
    delete_models,
    load_dataset_config,
    update_dataset_config,
    get_dataset_status,
    sync_dataset_from_excel,
    fetch_dataset,
    format_time_shanghai,
)


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


st.set_page_config(page_title="CoPiLinear", layout="wide")
st.title("CoPiLinear 价格预测工具")

MENU = st.sidebar.radio("导航", ["模型管理", "模型训练", "日前价格预测"])


# ----------------------------------------
# 1) 模型管理
# ----------------------------------------
if MENU == "模型管理":
    st.subheader("模型数据库")
    all_models = list_models()
    if not all_models:
        st.info("暂无模型，请先到“模型训练”页创建模型。")
    else:
        # 构建汇总表
        records = []
        for name in all_models:
            meta = load_model_meta(name) or {}
            pkl_path = os.path.join("models", f"{name}.pkl")
            meta_path = os.path.join("models", f"{name}.meta.json")
            train_path = os.path.join("models", f"{name}.train.csv")
            rec = {
                "选中": False,
                "名称": name,
                "创建时间": format_time_shanghai(meta.get("created_at")),
                "分段数": meta.get("n_segments"),
                "样本行数": meta.get("rows"),
                "数据来源": meta.get("source"),
                "断点": ", ".join([f"{v:.4f}" for v in meta.get("breakpoints", [])]) if meta.get("breakpoints") else None,
            }
            records.append(rec)
        df_models = pd.DataFrame.from_records(records)

        # 顶部工具条
        c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
        with c1:
            q = st.text_input("搜索名称", value="").strip()
        with c2:
            seg_options = sorted([v for v in df_models["分段数"].dropna().unique().tolist()])
            seg_filter = st.multiselect("分段数筛选", seg_options, default=[])
        # 过滤
        view_df = df_models.copy()
        if q:
            view_df = view_df[view_df["名称"].str.contains(q, case=False, na=False)]
        if seg_filter:
            view_df = view_df[view_df["分段数"].isin(seg_filter)]
            
        # 列配置：仅“选中”可编辑
        col_cfg = {
            "选中": st.column_config.CheckboxColumn("选中", help="勾选以加入批量操作"),
            "名称": st.column_config.TextColumn("名称", disabled=True),
            "创建时间": st.column_config.TextColumn("创建时间", disabled=True),
            "分段数": st.column_config.NumberColumn("分段数", disabled=True),
            "样本行数": st.column_config.NumberColumn("样本行数", disabled=True),
            "数据来源": st.column_config.TextColumn("数据来源", disabled=True),
            "断点": st.column_config.TextColumn("断点", disabled=True),
        }

        edited = st.data_editor(
            view_df,
            hide_index=True,
            use_container_width=True,
            column_config=col_cfg,
        )

        # 当前选择
        chosen_names = []
        if "选中" in edited.columns:
            chosen_names = edited.loc[edited["选中"] == True, :]["名称"].tolist()

        st.markdown("---")
        # 详情预览（单选时展示）
        if len(chosen_names) == 1:
            name = chosen_names[0]
            st.subheader(f"详情：{name}")
            meta = load_model_meta(name)
            if meta:
                st.write(f"创建时间: {format_time_shanghai(meta.get('created_at'))}")
                bps = meta.get('breakpoints')
                if bps is not None:
                    st.write(f"断点: {np.round(np.array(bps), 4)}")
                st.write(f"训练数据来源: {meta.get('source')}")
                st.write(f"样本行数: {meta.get('rows')}")
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
    st.subheader("数据源管理")
    dataset_status = get_dataset_status()
    cfg = load_dataset_config()

    with st.form("dataset_config_form"):
        cols = st.columns([3, 1])
        excel_path = cols[0].text_input("市场边际 Excel 路径", value=cfg.get("excel_path", ""))
        sheet_name = cols[1].text_input("工作表名称(可选)", value=cfg.get("sheet_name", ""))
        if st.form_submit_button("保存配置"):
            cfg = update_dataset_config({"excel_path": excel_path, "sheet_name": sheet_name})
            dataset_status = get_dataset_status()
            st.success("配置已保存")

    force_sync = st.checkbox("忽略缓存强制刷新", value=False)
    if st.button("刷新数据库", type="primary"):
        try:
            result = sync_dataset_from_excel(force=force_sync)
            dataset_status = get_dataset_status()
            st.success(f"已导入 {result['rows']:,} 行数据")
        except Exception as e:
            st.error(f"刷新失败：{e}")

    c1, c2, c3 = st.columns(3)
    c1.metric("状态", dataset_status["state_label"], dataset_status.get("status_reason") or "")
    c2.metric("数据库行数", f"{dataset_status['row_count']:,}")
    c3.metric("上次同步", dataset_status.get("last_sync_time") or "—")
    if dataset_status.get("excel_path"):
        st.caption(f"Excel: {dataset_status['excel_path']}")
    if dataset_status.get("excel_mtime_human"):
        st.caption(f"Excel 修改时间: {dataset_status['excel_mtime_human']}")

    st.markdown("---")
    st.subheader("模型训练")
    data_source = st.radio("训练数据来源", ["上传文件", "数据库"], horizontal=True)

    upload_file = None
    db_start: Optional[date] = None
    db_end: Optional[date] = None

    if data_source == "上传文件":
        upload_file = st.file_uploader("上传训练数据 Excel/CSV", type=["xlsx", "csv"])
    else:
        if dataset_status["row_count"] == 0:
            st.info("数据库为空，请先完成 Excel 导入。")
        min_date = _parse_date(dataset_status.get("date_min"))
        max_date = _parse_date(dataset_status.get("date_max"))
        today = date.today()
        default_end = max_date or today
        default_start = default_end - timedelta(days=30)
        if min_date:
            default_start = max(min_date, default_start)
        cols = st.columns(2)
        db_start = cols[0].date_input(
            "起始日期",
            value=default_start,
            min_value=min_date or default_start,
            max_value=default_end,
        )
        db_end = cols[1].date_input(
            "结束日期",
            value=default_end,
            min_value=db_start,
            max_value=max_date or default_end,
        )
        st.caption(
            f"可用数据区间：{dataset_status.get('date_min') or '未知'} 至 {dataset_status.get('date_max') or '未知'}"
        )

    model_name = st.text_input("模型名称", value="model_v1").strip()
    n_segments = st.slider("分段数 (pwlf)", min_value=2, max_value=8, value=3)

    if st.button("训练模型"):
        df_source = None
        source_label = None

        if data_source == "上传文件":
            if not upload_file:
                st.warning("请先上传训练数据文件")
            else:
                suffix = upload_file.name.lower()
                if suffix.endswith(".xlsx") or suffix.endswith(".xls"):
                    df_source = pd.read_excel(upload_file, header=None)
                else:
                    df_source = pd.read_csv(upload_file, header=None)
                source_label = upload_file.name
        else:
            if dataset_status["row_count"] == 0:
                st.warning("数据库为空，请先导入数据")
            elif not db_start or not db_end:
                st.warning("请选择完整的日期范围")
            elif db_start > db_end:
                st.warning("起始日期需早于结束日期")
            else:
                df = fetch_dataset(db_start, db_end)
                if df.empty:
                    st.warning("所选日期范围没有数据")
                else:
                    df_source = df
                    source_label = f"sqlite:{db_start}->{db_end}"

        if df_source is None:
            st.stop()
        if not model_name:
            st.warning("请填写模型名称")
            st.stop()

        try:
            model, breakpoints = train_model_ex(
                df_source,
                model_name=model_name,
                n_segments=n_segments,
                source=source_label,
            )
            st.success(
                f"训练完成：{model_name}，断点：{np.round(breakpoints, 4)}，来源：{source_label or 'unknown'}"
            )

            cleaned = clean_marginal_data(df_source.copy())
            scatter_df = cleaned[["load_rate", "price"]].copy()
            x_line = np.linspace(scatter_df["load_rate"].min(), scatter_df["load_rate"].max(), 200)
            line_df = pd.DataFrame(
                {
                    "load_rate": x_line,
                    "price": predict_price(model, x_line),
                }
            )
            bp_df = pd.DataFrame({"load_rate": np.asarray(breakpoints)})

            points = (
                alt.Chart(scatter_df)
                .mark_point(opacity=0.4, size=35)
                .encode(
                    x=alt.X("load_rate:Q", title="负荷率(%)"),
                    y=alt.Y("price:Q", title="价格"),
                )
            )
            line = (
                alt.Chart(line_df)
                .mark_line(color="#d62728", strokeWidth=2)
                .encode(x="load_rate:Q", y="price:Q")
            )
            rules = (
                alt.Chart(bp_df)
                .mark_rule(color="#7f7f7f", strokeDash=[4, 4])
                .encode(x="load_rate:Q")
            )

            st.altair_chart((points + line + rules).properties(title="负荷率拟合图"), use_container_width=True)
        except Exception as e:
            st.error(f"训练失败：{e}")


# ----------------------------------------
# 3) 日前价格预测
# ----------------------------------------
elif MENU == "日前价格预测":
    st.subheader("多模型日前价格预测")

    models = list_models()
    if not models:
        st.warning("暂无可用模型，请先在“模型训练”页训练并保存模型。")
        st.stop()
    selected_models = st.multiselect("选择模型", models, default=models[:1])
    if not selected_models:
        st.warning("请至少选择一个模型")
        st.stop()

    cap_input = st.text_input("日前在线机组容量 (MW，多值用逗号/空格分隔)", value="45000, 50000, 55000")
    capacities = []
    for part in cap_input.replace("，", ",").replace(" ", ",").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            v = float(p)
        except ValueError:
            st.error(f"容量值无法解析：'{p}'")
            st.stop()
        if v <= 0:
            st.error("容量必须大于 0 MW")
            st.stop()
        capacities.append(v)
    if not capacities:
        st.info("请输入至少一个正的容量值")

    file = st.file_uploader("上传包含“竞价空间”列的 CSV/XLSX", type=["csv", "xlsx"])
    if st.button("预测") and file:
        df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
        if not capacities:
            st.error("容量不能为空")
            st.stop()
        # 规范化列名后查找“竞价空间”
        def _normalize_col(col: str) -> str:
            s = str(col)
            for ch in [" ", "\u3000", "\t", "\n", "\r"]:
                s = s.replace(ch, "")
            return s.replace("（", "(").replace("）", ")")

        norm_map = {_normalize_col(c): c for c in df.columns}
        space_col = None
        for cand in ["竞价空间(MW)", "竞价空间"]:
            norm_cand = _normalize_col(cand)
            if norm_cand in norm_map:
                space_col = norm_map[norm_cand]
                break
        if space_col is None:
            fuzzy = [orig for norm, orig in norm_map.items() if ("竞价" in norm) and ("空间" in norm)]
            if fuzzy:
                space_col = fuzzy[0]
        if space_col is None:
            st.error(f"未找到包含“竞价空间”的列，当前列名：{list(df.columns)}")
            st.stop()

        space_vals = pd.to_numeric(df[space_col], errors="coerce").values
        if np.isnan(space_vals).all():
            st.error("“竞价空间”列无法解析为数值，请检查数据")
            st.stop()

        # 生成 15 分钟粒度时间标签（若为 96 条则最后为 24:00）
        _mins = np.arange(1, len(space_vals) + 1) * 15
        _hrs = _mins // 60
        _mm = _mins % 60
        time_labels = [f"{int(h):02d}:{int(m):02d}" for h, m in zip(_hrs, _mm)]

        # 多容量预测：为每个容量计算负荷率和各模型价格
        display_df = pd.DataFrame({"time_slot": time_labels})
        for cap in capacities:
            load_rate = (space_vals / cap) * 100.0
            display_df[f"负荷率(%)@{cap}MW"] = load_rate
            for name in selected_models:
                m = load_model(name)
                if m is None:
                    st.warning(f"模型 '{name}' 加载失败，已跳过")
                    continue
                preds = predict_price(m, load_rate)
                display_df[f"{name}@{cap}MW"] = preds
        st.dataframe(display_df.head())

        # 下载结果：包含 time_slot、各容量对应负荷率、各模型预测价格
        download_df = display_df.copy()
        st.download_button(
            "下载预测结果",
            download_df.to_csv(index=False).encode(),
            file_name="price_predictions.csv",
        )

        # 图表：仅显示各模型预测，不显示容量与负荷率；Y 轴锁定 0-1500，X 轴包含 24:00
        model_cols = [c for c in download_df.columns if c not in ["time_slot"] and not c.startswith("负荷率(%)@")]
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
