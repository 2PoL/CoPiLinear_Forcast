import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Union


def _log(message: str, *, verbose: bool) -> None:
    if verbose:
        print(message)


def extract_online_capacity(text):
    """从出清概况中提取在线机组容量"""
    if pd.isna(text):
        return None
    # 匹配 "运行机组容量42340.00MW"
    match = re.search(r'运行机组容量(\d+\.?\d*)\s*MW', str(text))
    if match:
        return float(match.group(1))
    return None


def _find_column(columns, keyword_groups):
    for keywords in keyword_groups:
        for col in columns:
            text = str(col)
            normalized = text.replace(" ", "").replace("\u3000", "").lower()
            if all(keyword.lower() in normalized for keyword in keywords):
                return col
    return None


def _finalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    columns = [
        '日期', '时点', '边界数据类型', '竞价空间(MW)', '省调负荷(MW)', '风电(MW)',
        '光伏(MW)', '新能源负荷(MW)', '非市场化出力(MW)', '水电出力(MW)',
        '联络线计划(MW)', '在线机组容量(MW)', '日前出清价格(元/MWh)',
        '实时出清价格(元/MWh)', '负荷率(%)'
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = None

    numeric_cols = [
        '省调负荷(MW)', '风电(MW)', '光伏(MW)', '新能源负荷(MW)', '非市场化出力(MW)',
        '水电出力(MW)', '联络线计划(MW)', '在线机组容量(MW)', '竞价空间(MW)'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    load = df['省调负荷(MW)']
    tie_line = df['联络线计划(MW)'].abs()
    new_energy = df['新能源负荷(MW)']
    non_market = df['非市场化出力(MW)']
    hydro = df['水电出力(MW)']

    df['竞价空间(MW)'] = load + tie_line - new_energy - non_market - hydro

    online_capacity = df['在线机组容量(MW)']
    with np.errstate(divide='ignore', invalid='ignore'):
        load_ratio = (df['竞价空间(MW)'] / online_capacity) * 100.0
    df['负荷率(%)'] = load_ratio.replace([np.inf, -np.inf], np.nan)

    df = df[columns]
    df['时点_排序'] = pd.to_datetime(df['时点'], format='%H:%M', errors='coerce')
    df['边界数据类型_排序'] = df['边界数据类型'].map({'日前': 0, '实时': 1})
    df = df.sort_values(['边界数据类型_排序', '日期', '时点_排序']).reset_index(drop=True)
    df = df.drop(columns=['时点_排序', '边界数据类型_排序'])
    return df


def preprocess_data(data_dir: Union[str, Path] = "margin_data", *, verbose: bool = False):
    """预处理所有数据文件"""
    data_dir = Path(data_dir)

    # 初始化结果DataFrame
    result_df = pd.DataFrame()

    # 1. 读取日前统调系统负荷预测_REPORT0 - D列 -> 省调负荷(MW) E列
    _log("处理日前统调系统负荷预测...", verbose=verbose)
    df_load = pd.read_excel(data_dir / "日前统调系统负荷预测_REPORT0.xlsx", header=0)
    # 从第2行开始读取数据（跳过表头行）
    df_load = df_load.iloc[1:].reset_index(drop=True)
    df_load['日期'] = pd.to_datetime(df_load.iloc[:, 1]).dt.date
    df_load['时点'] = df_load.iloc[:, 2].astype(str)
    df_load['省调负荷(MW)'] = pd.to_numeric(df_load.iloc[:, 3], errors='coerce')

    # 2. 读取日前新能源负荷预测_REPORT0 - E,F列 -> F,G列, D列 -> H列
    _log("处理日前新能源负荷预测...", verbose=verbose)
    df_renewable = pd.read_excel(data_dir / "日前新能源负荷预测_REPORT0.xlsx", header=0)
    df_renewable = df_renewable.iloc[1:].reset_index(drop=True)
    df_renewable['日期'] = pd.to_datetime(df_renewable.iloc[:, 1]).dt.date
    df_renewable['时点'] = df_renewable.iloc[:, 2].astype(str)
    df_renewable['风电(MW)'] = pd.to_numeric(df_renewable.iloc[:, 4], errors='coerce')
    df_renewable['光伏(MW)'] = pd.to_numeric(df_renewable.iloc[:, 5], errors='coerce')
    df_renewable['新能源负荷(MW)'] = pd.to_numeric(df_renewable.iloc[:, 3], errors='coerce')

    # 3. 读取披露信息96点数据_REPORT0 - D列 -> 非市场化出力(MW) I列
    _log("处理披露信息96点数据...", verbose=verbose)
    df_disclosure = pd.read_excel(data_dir / "披露信息96点数据_REPORT0.xlsx", header=0)
    df_disclosure = df_disclosure.iloc[1:].reset_index(drop=True)
    df_disclosure['日期'] = pd.to_datetime(df_disclosure.iloc[:, 1]).dt.date
    df_disclosure['时点'] = df_disclosure.iloc[:, 2].astype(str)
    df_disclosure['非市场化出力(MW)'] = pd.to_numeric(df_disclosure.iloc[:, 3], errors='coerce')

    # 4. 读取日前联络线计划_REPORT0 - E列 -> 联络线计划(MW) K列
    _log("处理日前联络线计划...", verbose=verbose)
    df_tie_line = pd.read_excel(data_dir / "日前联络线计划_REPORT0.xlsx", header=0)
    df_tie_line = df_tie_line.iloc[1:].reset_index(drop=True)
    df_tie_line = df_tie_line[df_tie_line.iloc[:, 1] == '总加']  # 只取总加行
    df_tie_line['日期'] = pd.to_datetime(df_tie_line.iloc[:, 2]).dt.date
    df_tie_line['时点'] = df_tie_line.iloc[:, 3].astype(str)
    df_tie_line['联络线计划(MW)'] = pd.to_numeric(df_tie_line.iloc[:, 4], errors='coerce')

    # 5. 读取日前市场出清情况_TABLE - 提取在线机组容量 -> L列
    _log("处理日前市场出清情况...", verbose=verbose)
    df_clearing = pd.read_excel(data_dir / "日前市场出清情况_TABLE.xlsx", header=0)
    df_clearing = df_clearing.iloc[1:].reset_index(drop=True)
    # 提取在线机组容量（只有一个值，应用到所有行）
    online_capacity = extract_online_capacity(df_clearing.iloc[0, 2])
    _log(f"  提取到在线机组容量: {online_capacity} MW", verbose=verbose)

    # 7. 读取日前水电计划发电总出力预测_REPORT0 - D列 -> 水电出力(MW) J列
    _log("处理日前水电计划...", verbose=verbose)
    df_hydro = pd.read_excel(data_dir / "日前水电计划发电总出力预测_REPORT0.xlsx", header=0)
    df_hydro = df_hydro.iloc[1:].reset_index(drop=True)
    df_hydro['日期'] = pd.to_datetime(df_hydro.iloc[:, 1]).dt.date
    df_hydro['时点'] = df_hydro.iloc[:, 2].astype(str)
    df_hydro['水电出力(MW)'] = pd.to_numeric(df_hydro.iloc[:, 3], errors='coerce')

    # 8. 读取96点电网运行实际值_REPORT0 - 实时数据
    _log("处理96点电网运行实际值...", verbose=verbose)
    df_actual = pd.read_excel(data_dir / "96点电网运行实际值_REPORT0.xlsx", header=0)
    df_actual = df_actual.iloc[1:].reset_index(drop=True)
    df_actual['日期'] = pd.to_datetime(df_actual.iloc[:, 1]).dt.date
    df_actual['时点'] = df_actual.iloc[:, 2].astype(str)
    df_actual['省调负荷(MW)'] = pd.to_numeric(df_actual.iloc[:, 3], errors='coerce')
    df_actual['风电(MW)'] = pd.to_numeric(df_actual.iloc[:, 5], errors='coerce')
    df_actual['光伏(MW)'] = pd.to_numeric(df_actual.iloc[:, 6], errors='coerce')
    df_actual['新能源负荷(MW)'] = pd.to_numeric(df_actual.iloc[:, 7], errors='coerce')
    df_actual['水电出力(MW)'] = pd.to_numeric(df_actual.iloc[:, 8], errors='coerce')
    df_actual['非市场化出力(MW)'] = pd.to_numeric(df_actual.iloc[:, 11], errors='coerce')

    # 9. 读取实时联络线计划_REPORT0 - E列 -> 联络线计划(MW) K列（实时）
    _log("处理实时联络线计划...", verbose=verbose)
    df_tie_line_rt = pd.read_excel(data_dir / "实时联络线计划_REPORT0.xlsx", header=0)
    df_tie_line_rt = df_tie_line_rt.iloc[1:].reset_index(drop=True)
    df_tie_line_rt = df_tie_line_rt[df_tie_line_rt.iloc[:, 1] == '总加']  # 只取总加行
    df_tie_line_rt['日期'] = pd.to_datetime(df_tie_line_rt.iloc[:, 2]).dt.date
    df_tie_line_rt['时点'] = df_tie_line_rt.iloc[:, 3].astype(str)
    df_tie_line_rt['联络线计划(MW)'] = pd.to_numeric(df_tie_line_rt.iloc[:, 4], errors='coerce')

    # 10. 读取现货出清电价_REPORT0 - 实时和日前出清价格
    _log("处理现货出清电价...", verbose=verbose)
    df_price = pd.read_excel(data_dir / "现货出清电价_REPORT0.xlsx")
    # 过滤掉均价汇总行（序号不是数字的行）
    df_price = df_price[pd.to_numeric(df_price['序号'], errors='coerce').notna()]
    df_price['日期'] = pd.to_datetime(df_price['日期']).dt.date
    df_price['时点'] = df_price['时点'].astype(str)
    df_price['实时出清价格(元/MWh)'] = pd.to_numeric(df_price['实时出清价格(元/MWh)'], errors='coerce')
    df_price['日前出清价格(元/MWh)'] = pd.to_numeric(df_price['日前出清价格(元/MWh)'], errors='coerce')

    # 合并所有日前数据
    _log("合并日前数据...", verbose=verbose)
    day_ahead_data = pd.merge(
        df_load[['日期', '时点', '省调负荷(MW)']],
        df_renewable[['日期', '时点', '风电(MW)', '光伏(MW)', '新能源负荷(MW)']],
        on=['日期', '时点'],
        how='outer'
    )
    day_ahead_data = pd.merge(
        day_ahead_data,
        df_disclosure[['日期', '时点', '非市场化出力(MW)']],
        on=['日期', '时点'],
        how='outer'
    )
    day_ahead_data = pd.merge(
        day_ahead_data,
        df_tie_line[['日期', '时点', '联络线计划(MW)']],
        on=['日期', '时点'],
        how='outer'
    )
    day_ahead_data = pd.merge(
        day_ahead_data,
        df_hydro[['日期', '时点', '水电出力(MW)']],
        on=['日期', '时点'],
        how='outer'
    )
    day_ahead_data = pd.merge(
        day_ahead_data,
        df_price[['日期', '时点', '日前出清价格(元/MWh)']],
        on=['日期', '时点'],
        how='outer'
    )

    # 添加边界数据类型和在线机组容量
    day_ahead_data['边界数据类型'] = '日前'
    day_ahead_data['在线机组容量(MW)'] = online_capacity

    # 额外：尝试加载公有数据看板实时在线容量
    real_time_capacity = None
    capacity_files = sorted(data_dir.glob("公有数据看板-实时*.xlsx"))
    if capacity_files:
        capacity_path = capacity_files[-1]
        _log(f"处理公有数据看板实时容量: {capacity_path.name}", verbose=verbose)
        try:
            capacity_raw = pd.read_excel(capacity_path, header=0)
            date_col = _find_column(capacity_raw.columns, [["日期"], ["date"]])
            time_col = _find_column(capacity_raw.columns, [["时点"], ["时段"], ["时间"], ["time"]])
            cap_col = _find_column(capacity_raw.columns, [["实时", "在线机组容量"], ["在线机组容量"]])
            if date_col and time_col and cap_col:
                capacity_df = capacity_raw[[date_col, time_col, cap_col]].copy()
                capacity_df.columns = ['日期', '时点', '在线机组容量(MW)']
                capacity_df['日期'] = pd.to_datetime(capacity_df['日期'], errors='coerce').dt.date
                capacity_df['时点'] = capacity_df['时点'].astype(str)
                capacity_df['在线机组容量(MW)'] = pd.to_numeric(capacity_df['在线机组容量(MW)'], errors='coerce')
                capacity_df = capacity_df.dropna(subset=['日期', '时点'])
                real_time_capacity = capacity_df
            else:
                _log("  未找到实时容量文件中的日期/时点/容量列，已跳过。", verbose=verbose)
        except Exception as exc:
            _log(f"  实时在线容量文件解析失败: {exc}", verbose=verbose)

    # 合并所有实时数据
    _log("合并实时数据...", verbose=verbose)
    real_time_data = df_actual[['日期', '时点', '省调负荷(MW)', '风电(MW)', '光伏(MW)',
                                  '新能源负荷(MW)', '水电出力(MW)', '非市场化出力(MW)']].copy()
    real_time_data = pd.merge(
        real_time_data,
        df_tie_line_rt[['日期', '时点', '联络线计划(MW)']],
        on=['日期', '时点'],
        how='left'
    )
    real_time_data = pd.merge(
        real_time_data,
        df_price[['日期', '时点', '实时出清价格(元/MWh)']],
        on=['日期', '时点'],
        how='left'
    )
    if real_time_capacity is not None:
        real_time_data = pd.merge(
            real_time_data,
            real_time_capacity,
            on=['日期', '时点'],
            how='left',
        )
    if '在线机组容量(MW)' not in real_time_data.columns:
        real_time_data['在线机组容量(MW)'] = None
    real_time_data['边界数据类型'] = '实时'

    # 合并日前和实时数据
    _log("合并所有数据...", verbose=verbose)
    result_df = pd.concat([day_ahead_data, real_time_data], ignore_index=True)
    return _finalize_dataframe(result_df)


def preprocess_template_file(template_path: Union[str, Path], *, verbose: bool = False) -> pd.DataFrame:
    template_path = Path(template_path)
    if not template_path.exists():
        raise FileNotFoundError(f"模板文件不存在: {template_path}")
    _log(f"读取模板文件: {template_path}", verbose=verbose)
    df = pd.read_excel(template_path, header=0)
    df.columns = [str(col).strip() for col in df.columns]
    if '日期' in df.columns:
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce').dt.date
    if '时点' in df.columns:
        df['时点'] = df['时点'].astype(str)
    return _finalize_dataframe(df)


def main():
    """主函数"""
    print("=" * 50)
    print("开始数据预处理")
    print("=" * 50)

    # 预处理数据
    result_df = preprocess_data(verbose=True)

    # 保存结果
    output_path = "预处理结果_新版.xlsx"
    result_df.to_excel(output_path, index=False, sheet_name="预处理数据")

    print("\n" + "=" * 50)
    print("预处理完成！")
    print(f"输出文件: {output_path}")
    print(f"总行数: {len(result_df)}")
    print("\n数据预览:")
    print(result_df.head(20).to_string())
    print("\n数据统计:")
    print(f"  日前数据行数: {len(result_df[result_df['边界数据类型'] == '日前'])}")
    print(f"  实时数据行数: {len(result_df[result_df['边界数据类型'] == '实时'])}")
    print("=" * 50)


if __name__ == "__main__":
    main()
