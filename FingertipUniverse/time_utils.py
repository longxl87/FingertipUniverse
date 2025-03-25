from datetime import timedelta,datetime
import pandas as pd


def week_n(date_time, n=1, rettype="str"):
    """
    获取给定日期所在周的星期N（1=周一，7=周日）。

    :param date_time: datetime 或 时间字符串（标准格式）
    :param n: 目标星期几（必须是 1-7 之间）
    :param rettype: 返回类型，"str" 返回 'YYYY-MM-DD'，"datetime" 返回 datetime 对象
    :return: 该周的星期N对应的日期（字符串或 datetime）
    """
    if not (1 <= n <= 7):
        raise ValueError("参数 `n` 必须在 1-7 之间（1=周一，7=周日）")

    if rettype not in {"str", "datetime"}:
        raise ValueError("参数 `rettype` 只能是 'str' 或 'datetime'")

    date_time = pd.to_datetime(date_time)  # 转换为 datetime
    delt = (n - 1) - date_time.weekday()  # 计算偏移量
    target_date = date_time + timedelta(days=delt)  # 计算目标日期
    target_date = target_date.normalize()
    return target_date.strftime('%Y-%m-%d') if rettype == "str" else target_date # 返回格式化日期


# if __name__ == '__main__':
#     from datetime import datetime, timedelta
#     tmp = week_n(datetime.now(),1,rettype="datetime")
#     print(tmp)



