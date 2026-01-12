import numpy as np


def calc_entropy_weights(X):
    """
    熵权法计算权重 (Entropy Weight Method).

    原理: 数据差异越小，熵越大，信息量越小，权重越低。
    """
    # X shape: (n_samples, n_features)
    X = np.array(X)

    # 1. 归一化 (Min-Max Normalization)
    # 必须保证数据非负，且为了避免 log(0)，需要加一个极小值
    # 注意：这里默认假设数据已经正向化了。
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)

    # 防止分母为0
    denominator = max_val - min_val
    denominator[denominator == 0] = 1

    p = (X - min_val) / denominator

    # 2. 计算比重 P_ij
    # 如果某一列和为0（全是一样的数），则权重均分
    col_sum = np.sum(p, axis=0)
    col_sum[col_sum == 0] = 1  # 避免除以0

    P = p / col_sum

    # 3. 计算熵值 E_j
    # k = 1 / ln(n)
    n = X.shape[0]
    if n <= 1:
        return np.ones(X.shape[1]) / X.shape[1]  # 样本太少，均权

    k = 1 / np.log(n)

    # 加上极小值防止 log(0)
    epsilon = 1e-10
    E = -k * np.sum(P * np.log(P + epsilon), axis=0)

    # 4. 计算差异系数 D_j = 1 - E_j
    D = 1 - E

    # 5. 计算权重 W_j
    W = D / np.sum(D)

    return W