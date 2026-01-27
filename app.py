import io
import os
import shutil
from datetime import date, timedelta
from pathlib import Path
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
    get_dataset_status,
    preprocess_dataset_and_sync,
    fetch_dataset,
    format_time_shanghai,
)

from scripts.pre_process import preprocess_data, preprocess_template_file


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _altair_chart(chart, *, width: str = "stretch") -> None:
    try:
        st.altair_chart(chart, width=width)
    except TypeError:
        # Backward-compatible fallback for older Streamlit versions.
        st.altair_chart(chart, use_container_width=(width == "stretch"))


REQUIRED_BOUNDARY_FILES = [
    "日前统调系统负荷预测_REPORT0.xlsx",
    "日前新能源负荷预测_REPORT0.xlsx",
    "披露信息96点数据_REPORT0.xlsx",
    "日前联络线计划_REPORT0.xlsx",
    "日前市场出清情况_TABLE.xlsx",
    "日前水电计划发电总出力预测_REPORT0.xlsx",
    "96点电网运行实际值_REPORT0.xlsx",
    "实时联络线计划_REPORT0.xlsx",
    "现货出清电价_REPORT0.xlsx",
]

PUBLIC_REALTIME_CAPACITY_LABEL = "公有数据看板-实时(天际云).xlsx"


def _df_to_excel_bytes(df: pd.DataFrame, *, sheet_name: str = "合并数据") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer.getvalue()


def _clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _persist_uploaded_files(files, dest_dir: Path) -> None:
    stored_files = []
    for file in files:
        file.seek(0)
        stored_files.append((file.name, file.getvalue()))
    _clear_directory(dest_dir)
    for name, content in stored_files:
        with open(dest_dir / name, "wb") as f:
            f.write(content)


st.set_page_config(page_title="CoPiLinear", layout="wide")
st.title("CoPiLinear 价格预测工具")

MENU = st.sidebar.radio("导航", ["数据预处理", "模型管理", "模型训练", "日前价格预测"])


# ----------------------------------------
# 0) 数据预处理
# ----------------------------------------
if MENU == "数据预处理":
    st.subheader("预处理边界数据")
    st.caption("上传 10 个边界 Excel 文件或 0市场边际数据库.xlsx 模板，完成数据清洗并导入数据库。")

    margin_dir = Path("margin_data")
    margin_dir.mkdir(parents=True, exist_ok=True)

    import_mode = st.radio(
        "导入方式",
        ["逐文件合并（10 个文件）", "模板批量导入 (0市场边际数据库.xlsx)"],
        horizontal=True,
        key="preprocess_import_mode",
    )

    boundary_success: Optional[str] = None
    boundary_error: Optional[str] = None

    if import_mode == "逐文件合并（10 个文件）":
        st.markdown("### 📤 上传边界数据文件")
        st.warning("⚠️ 请上传以下 10 个必需的 Excel 文件：")
        st.markdown(
            """
1. 日前统调系统负荷预测_REPORT0.xlsx  
2. 日前新能源负荷预测_REPORT0.xlsx  
3. 披露信息96点数据_REPORT0.xlsx  
4. 日前联络线计划_REPORT0.xlsx  
5. 日前市场出清情况_TABLE.xlsx  
6. 日前水电计划发电总出力预测_REPORT0.xlsx  
7. 96点电网运行实际值_REPORT0.xlsx  
8. 实时联络线计划_REPORT0.xlsx  
9. 现货出清电价_REPORT0.xlsx  
10. 公有数据看板-实时（文件名包含日期范围）
            """
        )

        boundary_files = st.file_uploader(
            "选择Excel文件（支持多选）",
            type=["xlsx"],
            accept_multiple_files=True,
            help="请一次性上传全部 10 个文件（含公有数据看板-实时）",
            key="boundary_file_uploader",
        )

        required_status = {name: False for name in REQUIRED_BOUNDARY_FILES}
        capacity_file_found = False
        if boundary_files:
            st.markdown(f"✅ 已选择 {len(boundary_files)} 个文件：")
            for file in boundary_files:
                st.write(f"  - {file.name}")
                if file.name in required_status:
                    required_status[file.name] = True
                if file.name.startswith("公有数据看板-实时"):
                    capacity_file_found = True

        missing_files = [name for name, found in required_status.items() if not found]
        if not capacity_file_found:
            missing_files.append(PUBLIC_REALTIME_CAPACITY_LABEL)

        # 显示文件状态信息
        if boundary_files:
            found_files = [name for name, found in required_status.items() if found]
            if capacity_file_found:
                found_files.append("公有数据看板-实时文件")

            if missing_files:
                st.info(f"📊 已上传 {len(found_files)} 个文件，还可选择上传 {len(missing_files)} 个文件：")
                for name in missing_files:
                    st.write(f"  - {name}")
                st.caption("💡 提示：可以使用部分文件进行合并，系统会根据可用文件生成相应的数据。")
            else:
                st.success("✅ 所有推荐文件已上传！")

        force_sync = st.checkbox("忽略缓存强制导入", value=False, key="boundary_force_sync")

        run_clicked = st.button(
            "🔄 保存文件并导入数据库",
            type="primary",
            key="boundary_process",
            disabled=(not boundary_files),  # 只要有文件就可以处理
        )
        merge_only_clicked = st.button(
            "⚙️ 仅合并生成预处理结果 (不导入数据库)",
            key="boundary_merge_only",
            disabled=(not boundary_files),  # 只要有文件就可以处理
        )

        if run_clicked or merge_only_clicked:
            if not boundary_files:
                boundary_error = "请先上传至少一个文件后再处理"
            else:
                try:
                    _persist_uploaded_files(boundary_files, margin_dir)
                    spinner_msg = (
                        "正在运行预处理脚本并写入数据库..."
                        if run_clicked
                        else "正在运行预处理脚本..."
                    )
                    with st.spinner(spinner_msg):
                        preview_df = preprocess_data(data_dir=margin_dir, verbose=False)
                        sync_result = None
                        if run_clicked:
                            sync_result = preprocess_dataset_and_sync(
                                force=force_sync,
                                data_dir=margin_dir,
                                source_label="multi_file_bundle",
                            )

                    st.session_state["boundary_result"] = preview_df
                    st.session_state["boundary_filename"] = "预处理结果_新版.xlsx"
                    if run_clicked and sync_result is not None:
                        boundary_success = f"{sync_result['status']}，导入 {sync_result['rows']:,} 行数据"
                    else:
                        boundary_success = f"合并完成，共 {len(preview_df):,} 行数据，可直接下载"
                except Exception as exc:
                    boundary_error = str(exc)

    else:
        st.markdown("### 📥 上传模板 (0市场边际数据库.xlsx)")
        template_file = st.file_uploader(
            "选择模板 Excel",
            type=["xlsx"],
            accept_multiple_files=False,
            key="template_file_uploader",
        )
        force_sync_template = st.checkbox("忽略缓存强制导入", value=False, key="template_force_sync")
        template_import = st.button(
            "🔄 导入模板到数据库",
            type="primary",
            key="template_import",
            disabled=template_file is None,
        )
        template_merge = st.button(
            "⚙️ 仅合并模板 (不导入数据库)",
            key="template_merge",
            disabled=template_file is None,
        )

        if template_import or template_merge:
            if not template_file:
                boundary_error = "请先上传模板文件"
            else:
                try:
                    _persist_uploaded_files([template_file], margin_dir)
                    template_path = margin_dir / template_file.name
                    spinner_msg = (
                        "正在处理模板并写入数据库..."
                        if template_import
                        else "正在处理模板..."
                    )
                    with st.spinner(spinner_msg):
                        template_df = preprocess_template_file(template_path, verbose=False)
                        sync_result = None
                        if template_import:
                            sync_result = preprocess_dataset_and_sync(
                                force=force_sync_template,
                                preprocessed_df=template_df,
                                source_label="template_file",
                            )

                    st.session_state["boundary_result"] = template_df
                    st.session_state["boundary_filename"] = template_file.name or "预处理结果_新版.xlsx"
                    if template_import and sync_result is not None:
                        boundary_success = f"{sync_result['status']}，导入 {sync_result['rows']:,} 行数据"
                    else:
                        boundary_success = f"模板合并完成，共 {len(template_df):,} 行，可直接下载"
                except Exception as exc:
                    boundary_error = str(exc)

    if boundary_success:
        st.success(boundary_success)
    if boundary_error:
        st.error(boundary_error)

    if "boundary_result" in st.session_state:
        result_df = st.session_state["boundary_result"]
        st.markdown("### 📊 处理结果统计")
        col1, col2, col3 = st.columns(3)
        col1.metric("总行数", len(result_df))
        col2.metric(
            "日前数据行数",
            len(result_df[result_df["边界数据类型"] == "日前"]),
        )
        col3.metric(
            "实时数据行数",
            len(result_df[result_df["边界数据类型"] == "实时"]),
        )

        if "在线机组容量(MW)" in result_df.columns:
            online_capacity = result_df["在线机组容量(MW)"].dropna()
            cap_val = (
                online_capacity.iloc[0]
                if not online_capacity.empty
                else "未找到"
            )
            st.info(f"💡 提取到在线机组容量: {cap_val}")

        st.markdown("### 👀 数据预览")
        st.dataframe(result_df.head(30), use_container_width=True)

        st.markdown("### 📥 下载预处理结果")
        excel_data = _df_to_excel_bytes(result_df, sheet_name="预处理数据")
        st.download_button(
            label="📥 下载预处理后的Excel文件",
            data=excel_data,
            file_name=st.session_state.get("boundary_filename", "预处理结果_新版.xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.markdown("---")
    st.markdown("### 数据库状态")
    dataset_status = get_dataset_status()
    c1, c2, c3 = st.columns(3)
    c1.metric("状态", dataset_status["state_label"], dataset_status.get("status_reason") or "")
    c2.metric("数据库行数", f"{dataset_status['row_count']:,}")
    c3.metric("上次同步", dataset_status.get("last_sync_time") or "—")
    st.caption(
        f"可用数据区间：{dataset_status.get('date_min') or '—'} 至"
        f" {dataset_status.get('date_max') or '—'}"
    )
    if dataset_status.get("data_dir"):
        st.caption(f"最近导入目录：{dataset_status['data_dir']}")
    if dataset_status.get("raw_row_count") is not None:
        st.caption(f"原始表条目：{dataset_status['raw_row_count']:,}")

    st.caption("💡 提示：上传的文件仅用于当前会话，不会被永久保存。")


# ----------------------------------------
# 1) 模型管理
# ----------------------------------------
elif MENU == "模型管理":
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
            width="stretch",
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
    st.subheader("数据源概览")
    dataset_status = get_dataset_status()

    c1, c2, c3 = st.columns(3)
    c1.metric("状态", dataset_status["state_label"], dataset_status.get("status_reason") or "")
    c2.metric("数据库行数", f"{dataset_status['row_count']:,}")
    c3.metric("上次同步", dataset_status.get("last_sync_time") or "—")
    st.caption(
        f"数据区间：{dataset_status.get('date_min') or '—'} 至"
        f" {dataset_status.get('date_max') or '—'}"
    )
    if dataset_status.get("data_dir"):
        st.caption(f"最近使用的数据目录：{dataset_status['data_dir']}")
    st.info("若需刷新数据库，请前往“数据预处理”页运行脚本。")

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
            st.info("数据库为空，请先到“数据预处理”页导入数据。")
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

            _altair_chart((points + line + rules).properties(title="负荷率拟合图"), width="stretch")
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
        _altair_chart(chart, width="stretch")


else:
    st.stop()
