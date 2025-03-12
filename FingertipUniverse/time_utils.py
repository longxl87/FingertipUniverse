from datetime import timedelta


def first_day_of_week(date_obj):
    """
    获取该日期对象，对应的每周第一天
    :param date_obj:
    :return:
    """
    weekday = date_obj.weekday()
    start_of_week = date_obj - timedelta(days=weekday)
    return start_of_week.strftime('%Y-%m-%d')

