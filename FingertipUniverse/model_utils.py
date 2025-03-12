import numpy as np
import pandas as pd

def prob2score(prob):
    """
    prob转化为概率分的工具
    :param prob:
    :return:
    """
    return round(550 - 60 / np.log(2) * np.log(prob / (1 - prob)), 0)

def feature_class(df, model_fea):
    """
    区分出数据集中数值特征和非数字特征
    :param df:pd.Dataframe
    :param model_fea:
    :return:
    """
    num_fea = []
    cate_fea = []
    for col in model_fea:
        if pd.api.types.is_numeric_dtype(df[col]):
            num_fea.append(col)
        else:
            cate_fea.append(col)
    return num_fea, cate_fea