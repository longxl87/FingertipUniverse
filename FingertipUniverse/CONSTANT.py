
MON_PARTTEN = '%Y-%m'

DAY_PARTTEN = '%Y-%m-%d'

STANDARD_TIME_PARTTEN = '%Y-%m-%d %H:%M:%S'

LONG_TIME_PARTTEN = "%Y-%m-%d %H:%M:%S.%f"


LGB_PARAMS = { # 汇总列出常用的参数
    "task": "train",
    "boosting_type": "gbdt",  # 设置提升类型
    "objective": "binary",  # 目标函数 "regression"
    "metric": {"auc"},  # 评估函数
    "max_depth": 2,
    "num_leaves": 2,  # 叶子节点数
    "n_estimators": 800,
    "learning_rate": 0.1,  # 学习速率
    'feature_fraction': 0.8,  # 使用80%的特征
    'bagging_fraction': 0.8,  # 使用80%的样本
    "bagging_freq": 5,  # k 意味看每 k 次迭代执行bagging
    "lambda_l1": 0,
    "lambda_l2": 0,
    'min_gain_to_split':0.05,
    "verbose": -1,
}