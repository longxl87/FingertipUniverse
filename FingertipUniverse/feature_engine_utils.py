import numpy as np
import pandas as pd
import math

from sklearn.metrics import (
    roc_auc_score,
    roc_curve)

from FingertipUniverse.binning_utils import (
    cut_bins,
    make_bin,
    chi2merge
)

def calc_auc(y_true, y_score):
    """
    计算评分或者特征的auc
    :param y_true:
    :param y_score:
    :return:
    """
    return roc_auc_score(y_true, y_score)

def calc_psi(x_true, x_pred, n_bins=10, method='freq', alpha=0.1):
    """
    计算两组特征的psi
    :param x_true:
    :param x_pred:
    :param n_bins:
    :param method:
    :param alpha:
    :return:
    """
    if pd.api.types.is_numeric_dtype(x_true):
        bound = cut_bins(x_true, n_bins=n_bins, method=method)
        x_base_bins = np.array(make_bin(x_true, bound, num_fillna=-999))
        x_pred_bins = np.array(make_bin(x_pred, bound, num_fillna=-999))
    else:
        x_base_bins = x_true
        x_pred_bins = x_pred

    b_bins = np.unique(np.concatenate((x_base_bins, x_pred_bins)))
    n_base = len(x_true) + len(b_bins) * alpha
    n_pred = len(x_pred) + len(b_bins) * alpha

    psi_mapping = []
    for bin in b_bins:
        c_base = np.count_nonzero(x_base_bins == bin)
        c_pred = np.count_nonzero(x_pred_bins == bin)
        pct_base = (c_base + alpha) / n_base
        pct_pred = (c_pred + alpha) / n_pred
        csi = (pct_base - pct_pred) * math.log(pct_base / pct_pred)
        psi_mapping.append((bin, c_base, c_pred, csi))

    psi = sum([x[3] for x in psi_mapping])
    return psi


def calc_iv(y_true, y_score, n_bins=10, method='freq', fillna=-999, alpha=0.1):
    """
    简单计算IV值
    :param y_true:
    :param y_score:
    :param n_bins:
    :param method:
    :param fillna:
    :param alpha:
    :return:
    """
    y_true = np.array(y_true,copy=True)
    y_score = np.array(y_score,copy=True)

    if pd.api.types.is_numeric_dtype(y_score):
        bound = cut_bins(y_score, n_bins, method)
        y_pred_bins = np.asarray(make_bin(y_score, bound, num_fillna=fillna))
    else:
        bound = chi2merge(y_true, y_score, n_bins)
        y_pred_bins = np.asarray(make_bin(y_score, bound))

    data = np.column_stack((y_true, y_pred_bins))

    iv_arr = []
    bins = np.unique(y_pred_bins)
    N_g = (y_true == 0).sum() + (alpha * len(bins))
    N_b = (y_true == 1).sum() + (alpha * len(bins))
    for bin in bins:
        data_bin = data[data[:, 1] == bin]
        n_g = (data_bin[:, 0] == 0).sum() + alpha
        n_b = (data_bin[:, 0] == 1).sum() + alpha
        pct_g = (n_g * 1.0 / N_g)
        pct_b = (n_b * 1.0 / N_b)
        iv = (pct_g - pct_b) * math.log(pct_g/pct_b)
        iv_arr.append(iv)
    iv = sum(iv_arr)
    return iv

def calc_ks(y_true, y_score):
    """
    计算评分或者特征的ks结果
    :param y_true:
    :param y_score:
    :return:
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ks = np.max(np.abs(fpr - tpr))
    return ks


def univariate(y, x, n_bins=10, bins=None, alpha=0.1, num_fill=-999.0, cate_fill=''):
    y = np.array(y, copy=True, dtype=np.float64)
    if pd.api.types.is_numeric_dtype(x):
        x = np.array(x, copy=True, dtype=float)
        if bins is None:
            bins = cut_bins(x, n_bins, method='freq')
        data = pd.DataFrame({'x': x, 'y': y})
        data.fillna(num_fill, inplace=True)
        data['bin'] = pd.cut(data['x'], bins=bins, right=True, retbins=False)
        corr = data['x'].corr(data['y'])
        auc = roc_auc_score(data['y'], data['x'] * -1.0) if corr < 0 else roc_auc_score(data['y'], data['x'])
    else:
        data = pd.DataFrame({'x': x, 'y': y})
        if bins is not None:
            data['bin'] = data['x'].map(bins).fillna(cate_fill)
        else:
            data['bin'] = data['x'].fillna(cate_fill)
            corr = 0.1
            auc = 0

    ascending = corr > 0

    br = data['y'].mean()
    dti = data.groupby('bin', observed=False).agg(
        bad=('y', 'sum'),
        cnt=('y', 'count'),
        brate=('y', 'mean')
    ).sort_values('bin', ascending=ascending).assign(
        good=lambda x: x['cnt'] - x['bad'],
        pct=lambda x: x['cnt'] / (x['cnt'].sum())
    ).assign(
        bad_cum=lambda x: x['bad'].cumsum(),
        good_cum=lambda x: x['good'].cumsum(),
        bad_prime=lambda x: x['bad'] + alpha,
        good_prime=lambda x: x['good'] + alpha,
        lift=lambda x: x['brate'] / br
    ).assign(
        brate_cum=lambda x: x['bad_cum'] / (x['bad_cum'] + x['good_cum']),
        bad_cum_raito=lambda x: x['bad_cum'] / (x['bad'].sum()),
        good_cum_raito=lambda x: x['good_cum'] / (x['good'].sum()),
        bad_ratio_prime=lambda x: x['bad_prime'] / (x['bad_prime'].sum()),
        good_ratio_prime=lambda x: x['good_prime'] / (x['good_prime'].sum()),
    ).assign(
        woe=lambda x: np.log(x['bad_ratio_prime'] / x['good_ratio_prime']),
    ).assign(
        ks=lambda x: abs(x['bad_cum_raito'] - x['good_cum_raito']),
        iv=lambda x: (x['bad_ratio_prime'] - x['good_ratio_prime']) * x['woe'],
        auc=auc
    ).assign(
        iv=lambda x: x['iv'].sum()
    ).reset_index()
    return dti[['bin', 'bad', 'cnt', 'pct', 'brate', 'brate_cum', 'lift', 'woe', 'ks', 'iv', 'auc']]
