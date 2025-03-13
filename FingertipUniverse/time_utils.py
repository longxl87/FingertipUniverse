from datetime import timedelta

import pandas as pd


def day_n_of_week(date_time,n=1):
    """
    获取当周的 星期N
    :param date_time: 需要转换的时间
    :param n: 星期n 必须是 1-7之间的数
    :return:
    """
    assert (n>0) and (n<8) ,f'n的参数必须在1-7之间'
    if isinstance(date_time, str):
        date_time = pd.to_datetime(date_time)
    delt = (n-1) -date_time.weekday()
    start_of_week = date_time + timedelta(days=delt)
    return start_of_week.strftime('%Y-%m-%d')



