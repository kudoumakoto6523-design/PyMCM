import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_heatmap(mcm_data, method='pearson', title="Correlation Heatmap"):
    """
    绘制相关系数热力图 (Seaborn style).

    Args:
        mcm_data: MCMData object or pd.DataFrame.
        method: 'pearson' (linear) or 'spearman' (rank/nonlinear).
    """
    # 1. 获取数据
    if hasattr(mcm_data, 'df'):
        df = mcm_data.df
    elif isinstance(mcm_data, pd.DataFrame):
        df = mcm_data
    else:
        raise ValueError("Input must be MCMData or DataFrame")

    # 2. 计算相关系数矩阵
    corr_matrix = df.corr(method=method)

    # 3. 画图
    plt.figure(figsize=(10, 8))
    # annot=True 显示数字, fmt='.2f' 保留两位小数, cmap='coolwarm' 红蓝配色
    sns.heatmap(corr_matrix, annot=True, fmt='.2f',
                cmap='coolwarm', vmin=-1, vmax=1,
                square=True, linewidths=.5)

    plt.title(f"{title} ({method.capitalize()})")
    plt.show()

    return corr_matrix