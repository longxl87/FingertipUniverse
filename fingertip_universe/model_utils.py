import numpy as np
import pandas as pd

def prob2score(prob):
    """
    prob转化为概率分的工具
    :param prob:
    :return:
    """
    return round(550 - 60 / np.log(2) * np.log(prob / (1 - prob)), 0)