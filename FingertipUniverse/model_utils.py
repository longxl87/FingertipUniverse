import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from tqdm import tqdm


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

def oppsite_features(df1, df2, model_feas, target):
    """
    计算两组数据中，反向的特征
    :param df1:
    :param df2:
    :param model_feas:
    :param target:
    :return:
    """
    oppsite_feas = []
    for fea in tqdm(model_feas):
        corr1 = df1[fea].corr(df1[target])
        corr2 = df2[fea].corr(df2[target])
        if corr1 * corr2 <= 0:
            oppsite_feas.append(fea)
    return oppsite_feas

def oppsite_feature_kfold(df, model_feas, target, n_splits=5, random_state=42):
    """
    计算该数据集中，随机分组后都不稳定的特征
    :param df:
    :param model_feas:
    :param target:
    :param n_splits:
    :param random_state:
    :return:
    """
    df = df.copy()
    df = df.reset_index(drop=True)
    assert df.index.is_unique, '输入数据的index有重复值，请优先确保数据格式的正确性'
    oppsite_feas = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for idxs1, idxs2 in kfold.split(df):
        df1 = df.iloc[idxs1, :]
        df2 = df.iloc[idxs2, :]
        for x in tqdm(model_feas):
            corr1 = df1[x].corr(df1[target])
            corr2 = df2[x].corr(df2[target])
            if corr1 * corr2 <= 0:
                oppsite_feas.append(x)
    oppsite_feas = list(set(oppsite_feas))
    return oppsite_feas

def model_features(lgb_model, importance_type='gain',filter_zero=True):
    """
    分析模型特征及权重
    :param lgb_model:
    :param importance_type:
    :return:
    """
    fea_rs = pd.DataFrame(
        {"var": lgb_model.feature_name(), "imps": lgb_model.feature_importance(importance_type=importance_type)}
    ).sort_values("imps", ascending=False)
    if filter_zero:
        fea_rs = fea_rs[fea_rs['importance'] > 0]
    return fea_rs

def plot_roc_ks(y_true, y_scores, data_desc=None):
    """
    绘制auc和ks
    :param y_true:
    :param y_scores:
    :param data_desc:
    :return:
    """
    y_scores = np.asarray(y_scores)
    y_true = np.asarray(y_true)
    # 计算FPR, TPR 和 AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    # 按照预测概率排序
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_y_true = y_true[sorted_indices]
    # 计算累计分布
    total_positives = np.sum(y_true == 1)
    total_negatives = np.sum(y_true == 0)
    cumulative_positives = np.cumsum(sorted_y_true == 1) / total_positives
    cumulative_negatives = np.cumsum(sorted_y_true == 0) / total_negatives
    # KS统计量
    ks_statistic = max(abs(cumulative_positives - cumulative_negatives))
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 9})  # 将默认字体大小设置为10
    # 创建一个1行2列的子图布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))  # 减小图表尺寸
    # 绘制ROC曲线
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC)')
    ax1.legend(loc="lower right")
    # 绘制KS曲线
    ax2.plot(cumulative_positives, label='Cumulative Positives')
    ax2.plot(cumulative_negatives, label='Cumulative Negatives')
    ax2.axvline(np.argmax(abs(cumulative_positives - cumulative_negatives)), color='r', linestyle='--',
                label=f'Max KS: {ks_statistic:.2f}')
    ax2.set_xlabel('Sample Fraction (sorted by score)')
    ax2.set_ylabel('Cumulative Distribution Function')
    ax2.set_title('KS Statistic Curve')
    ax2.legend()
    if data_desc:  # 添加主标题
        plt.suptitle(f'Performance on {data_desc}', fontsize=14)  # 主标题字体大小也相应减小

    # 调整子图之间的间距
    plt.tight_layout(rect=[0, 0, 1, 1])  # rect参数确保标题不会被裁剪
    plt.show()