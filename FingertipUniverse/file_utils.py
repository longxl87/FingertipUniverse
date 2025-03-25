import os
import re
import traceback

import pandas as pd
from tqdm import tqdm
from openpyxl.worksheet.worksheet import Worksheet


def data_of_dir(dir_path, contains_flags="", start_time=None, end_time=None):
    """
    基于关键字以及时间来扫描文件目录中的数据块
    :param dir_path: 待扫描的目录
    :param contains_flags: 关键字标签，同时支持 字符串以及 字符串数组两种格式
    :param start_time: 开始时间，如果 是'2023-01-01' 格式，则按照日期进行扫描，否则为月份扫描
    :param end_time: 结束时间，本函数采用左开右开的方式进行时间范围框定
    :return:
    """

    def check_date_format(date_str):
        patterns = {
            r"^\d{4}-\d{2}$": "Month",
            r"^\d{4}-\d{2}-\d{2}$": "Dayt"
        }
        for pattern, label in patterns.items():
            if re.match(pattern, date_str):
                return label
        return "Unknown"

    def _fetch_filenams(dir_path: str, contain_flag, start_time=None, end_time=None):
        file_paths = []
        contain_flag = contain_flag or ""
        if start_time is not None:
            time_type = check_date_format(start_time)
            if time_type == 'Dayt':
                pattern = r"\d{4}-\d{2}-\d{2}"
                end_time = '2999-12-01' if end_time is None else end_time
            elif time_type == 'Month':
                pattern = r"\d{4}-\d{2}"
                end_time = '2999-12' if end_time is None else end_time
            else:
                raise ValueError(f'调用函数的入参start_time={start_time}，格式错误')
        for file_name in os.listdir(dir_path):
            if (contain_flag in file_name) and (
            file_name.endswith(('.feather','.fth', '.pqt', '.parquet', '.csv', '.xlsx', '.pickle', '.pkl'))):
                if start_time is None:
                    file_paths.append(os.path.join(dir_path, file_name))
                else:
                    match = re.search(pattern, file_name)
                    time = match.group()
                    if (time >= start_time) and (time <= end_time):
                        file_paths.append(os.path.join(dir_path, file_name))
        file_paths.sort()
        return file_paths

    if isinstance(contains_flags, str) or contains_flags is None:
        return _fetch_filenams(dir_path, contains_flags, start_time, end_time)
    elif isinstance(contains_flags, list):
        file_names = None
        for contains_flag in contains_flags:  # type: ignore
            if file_names is None:
                file_names = _fetch_filenams(dir_path, contains_flag, start_time, end_time)
            else:
                file_names = file_names + _fetch_filenams(dir_path, contains_flag, start_time, end_time)
        return file_names

def batch_load_data(file_paths, load_function=pd.read_parquet):
    result = None
    for file_path in tqdm(file_paths):
        try:
            if result is None:
                result = load_function(file_path)
            else:
                result = pd.concat([result, load_function(file_path)])
        except:
            print(f"文件加载异常:{file_path},跳过该文件\n{traceback.format_exc()}")
    return result

def save_data_to_excel(df, sheet, row_number, col_number):
    """
    将指定的内容输出到对应的sheet页面中
    :param df:
    :param sheet:
    :param row_number:
    :param col_number:
    :return:
    """
    for row_num, index in enumerate(df.index, row_number):
        for col_num, col in enumerate(df.keys(), col_number):
            cell = sheet.cell(row=row_num, column=col_num, value=df[col][row_num - row_number])


def save_to_excel(data, sheet, row_number=1, col_number=1, write_header=False):
    """
    将数据（DataFrame 或 list of list）写入 openpyxl 的 Excel sheet，从指定的 (row_number, col_number) 开始。
    :param data: Pandas DataFrame 或 list of list
    :param sheet: openpyxl worksheet
    :param row_number: 开始写入的行号（Excel 1-based index）
    :param col_number: 开始写入的列号（Excel 1-based index）
    :param write_header: 是否写入列名（仅在 data 为 DataFrame 时有效）
    """
    if not isinstance(sheet, Worksheet):
        raise TypeError("参数 `sheet` 必须是 openpyxl 的 Worksheet 对象")

    if isinstance(data, pd.DataFrame):
        # 如果需要写入列名
        if write_header:
            data = [data.columns.tolist()] + data.values.tolist()
        else:
            data = data.values.tolist()  # 转换为 list of list
    elif not isinstance(data, list) or not all(isinstance(row, list) for row in data):
        raise TypeError("参数 `data` 必须是 Pandas DataFrame 或 list of list")

    # 遍历数据写入 Excel
    for i, row in enumerate(data, start=row_number):
        for j, value in enumerate(row, start=col_number):
            sheet.cell(row=i, column=j, value=value)

