import pandas as pd
import numpy as np


def cut_bins(x, n_bins=10, method='freq'):
    """
    数值类特征基础的获取分箱bound的方式
    :param x: array_like 待分箱的数据
    :param n_bins: 分箱数量
    :param method: 分箱的方式 默认 'freq' 等频率分箱，'dist' 对应等间距分箱 ,'chi2' 卡方分箱 ,'bestks' 对应best ks分箱
    :return: list
    """
    data = np.array(x, copy=True, dtype=np.float64)
    data = data[~np.isnan(data)]
    if method == 'freq':
        sorted_data = np.sort(data)
        # 计算每个箱子的数据点索引
        indices = np.linspace(0, len(sorted_data) - 1, n_bins + 1, dtype=int)
        bin_edges = sorted_data[indices]
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < n_bins+1:
            bin_edges = np.insert(bin_edges, 0, -np.inf)
        else:
            bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        return bin_edges
    elif method == 'dist':
        max_v = np.max(data)
        min_v = np.min(data)
        binlen = (max_v - min_v) / n_bins
        bin_edges = [min_v + i * binlen for i in range(n_bins + 1)]
        bin_edges = np.unique(bin_edges)  # #np.unique(bin_edges).astype(float)
        return bin_edges
    else:
        raise ValueError('method must be \'freq\' or \'dist\'')

def chi2merge(y, x, num_bins=10, num_fillna=-999, cate_fillna=""):
    def _init_grouped(grouped):
        grd_arr = grouped.values
        bin_n = len(grd_arr)
        del_mask = []
        for i in range(bin_n):
            if i == bin_n - 1:
                if (grd_arr[i][1] == 0) or (grd_arr[i][2] == 0):
                    grd_arr[i - 1][0] = grd_arr[i][0]
                    grd_arr[i - 1][1] = grd_arr[i - 1][1] + grd_arr[i][1]
                    grd_arr[i - 1][2] = grd_arr[i - 1][2] + grd_arr[i][2]
                    del_mask.append(i)
            else:
                if (grd_arr[i][1] == 0) or (grd_arr[i][2] == 0):
                    grd_arr[i + 1][1] = grd_arr[i + 1][1] + grd_arr[i][1]
                    grd_arr[i + 1][2] = grd_arr[i + 1][2] + grd_arr[i][2]
                    del_mask.append(i)
        return np.delete(grd_arr, del_mask, axis=0)

    def _chi2(a, b, c, d):
        """
        如下横纵标对应的卡方计算公式为： K^2 = n (ad - bc) ^ 2 / [(a+b)(c+d)(a+c)(b+d)]　其中n=a+b+c+d为样本容量
            y1   y2
        x1  a    b
        x2  c    d
        :return: 卡方值
        """
        a, b, c, d = float(a), float(b), float(c), float(d)
        return ((a + b + c + d) * ((a * d - b * c) ** 2)) / ((a + b) * (c + d) * (b + d) * (a + c))

    y = np.asarray(y)
    x = np.asarray(x)
    data = pd.DataFrame({
        'y': y, 'x': x
    })
    if pd.api.types.is_numeric_dtype(x):
        data['x'] = data['x'].fillna(num_fillna)
    else:
        data['x'] = data['x'].fillna(cate_fillna)

    grouped = data.groupby("x").agg(
        bad=("y", 'sum'),
        total=('y', 'count'),
        bad_rate=('y', 'mean')
    ).reset_index()
    grouped['good'] = grouped['total'] - grouped['bad']
    if pd.api.types.is_numeric_dtype(x):
        grouped['var'] = pd.qcut(grouped['x'], 200, duplicates='drop')
        grouped['var'] = grouped['var'].map(lambda x: x.right)
    else:
        grouped = grouped.sort_values('bad_rate')
        grouped['var'] = grouped['bad_rate'].rank(method='dense')

    x_var_df = grouped.copy()
    max_var = grouped['var'].max() + 0.1
    min_var = grouped['var'].min() - 0.1
    grouped = grouped.groupby('var').agg(
        bad=('bad', 'sum'),
        good=('good', 'sum')
    ).reset_index()

    grd_arr = _init_grouped(grouped)

    chi_dict = {}
    while len(grd_arr) > num_bins:
        for i in range(1, len(grd_arr)):
            a = grd_arr[i - 1][1]
            b = grd_arr[i - 1][2]
            c = grd_arr[i][1]
            d = grd_arr[i][2]
            chi = _chi2(a, b, c, d)
            chi_dict[i] = chi
        min_chi_idx = min(chi_dict, key=chi_dict.get)
        grd_arr[min_chi_idx - 1][1] = grd_arr[min_chi_idx - 1][1] + grd_arr[min_chi_idx][1]
        grd_arr[min_chi_idx - 1][2] = grd_arr[min_chi_idx - 1][2] + grd_arr[min_chi_idx][2]
        grd_arr[min_chi_idx - 1][0] = grd_arr[min_chi_idx][0]
        grd_arr = np.delete(grd_arr, min_chi_idx, axis=0)
        chi_dict = {}
    bounds = grd_arr[:, 0]
    bounds = np.concatenate(([min_var], bounds))
    bounds[len(bounds) - 1] = max_var
    x_var_df['bin'] = x_var_df['var'].map(lambda x: np.digitize(x, bounds))

    if pd.api.types.is_numeric_dtype(x):
        result = bounds
    else:
        result = dict(zip(x_var_df['x'], x_var_df['bin']))
    return result

def make_bin(x, bins, num_fillna=-999, cate_fillna=""):
    """
    根据传入的数据对结果映射分箱，自动填充缺失值，并约束上限
    :param x: 待分箱的特征
    :param bins: 数值型特征为 list:number, 类别特征为 dict(string:int)
    :param num_fillna: 数值型特征默认填充的方式
    :param cate_fillna: 类别特征默认的填充方式
    :return:
    """
    data = np.array(x,copy=True)
    if pd.api.types.is_numeric_dtype(x):
        bins = np.asarray(bins)
        data[np.isnan(data)] = num_fillna
        bound_max = np.max(bins)
        data[data > bound_max] = np.max(bins)
        indices = np.digitize(data, bins, right=True)
        return bins[indices]
    else:
        data = pd.Series(data).fillna(cate_fillna)
        keys = bins.keys()
        data = data.map(lambda x: x if x in keys else cate_fillna)
        return np.asarray(pd.Series(data).map(bins).fillna(0))