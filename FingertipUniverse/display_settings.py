import warnings

import pandas as pd


def set_warnings(need=True):
    if need:
        warnings.filterwarnings('default')
    else:
        warnings.filterwarnings('ignore')

def set_pd_show(max_rows=500, max_columns=200, max_colwidth=300):
    if max_rows:
        pd.set_option("display.max_rows", max_rows)
    else:
        pd.reset_option('display.max_rows')

    if max_columns:
        pd.set_option("display.max_columns", max_columns)
    else:
        pd.reset_option('display.max_columns')

    if max_colwidth:
        pd.set_option("display.max_colwidth", max_colwidth)
    else:
        pd.reset_option('display.max_colwidth')

def set_pd_float(f_formart="{:,.2f}"):
    """
    设置数值显示方式
    :param f_formart: 没有值默认，有则按照需要设定
    :return:
    """
    if f_formart:
        pd.options.display.float_format = f_formart.format
    else:
        pd.reset_option('display.float_format')