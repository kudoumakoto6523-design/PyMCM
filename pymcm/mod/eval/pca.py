import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler
from pymcm.core.base import BaseModel


class PCAModel(BaseModel):
    """
    Principal Component Analysis (PCA) for Evaluation.

    Uses PCA to reduce dimensions and calculate a comprehensive score based on variance contribution.
    """

    def __init__(self, n_components=None, threshold=0.85):
        """
        Args:
            n_components (int): Number of components to keep.
            threshold (float): If n_components is None, keep components until
                               cumulative variance ratio > threshold (e.g., 85%).
        """
        super().__init__(name="PCA Evaluation")
        self.n_components = n_components
        self.threshold = threshold
        self.pca = None
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.scores = None
        self.rankings = None

    def fit(self, mcm_data):
        X = mcm_data.get_X()

        # 1. PCA 必须标准化 (Standardization)
        # 均值为0，方差为1
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 2. 初始化 PCA
        # 如果指定了个数就用个数，没指定就用阈值(0.85)自动定个数
        if self.n_components:
            self.pca = SklearnPCA(n_components=self.n_components)
        else:
            self.pca = SklearnPCA(n_components=self.threshold)

        # 3. 拟合
        X_pca = self.pca.fit_transform(X_scaled)

        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.components_ = self.pca.components_

        print(f"[{self.name}] Preserved {self.pca.n_components_} components.")
        print(f"   -> Variance Ratios: {np.round(self.explained_variance_ratio_, 4)}")
        print(f"   -> Total Info Retained: {sum(self.explained_variance_ratio_):.2%}")

        # 4. 计算综合得分 (Comprehensive Score)
        # 核心逻辑：得分 = Sum(主成分值 * 该成分的方差贡献率)
        # X_pca 是降维后的数据 (n_samples, n_components)
        # weights 是贡献率
        weights = self.explained_variance_ratio_

        # 加权求和
        # 归一化权重（可选，为了让分数好看点）
        weights_norm = weights / np.sum(weights)
        self.scores = np.dot(X_pca, weights_norm)

        # 5. 排名
        self.rankings = np.argsort(-self.scores) + 1
        self.is_fitted = True
        return self

    def _predict_single(self, x):
        return 0

    def plot(self):
        if not self.is_fitted: return
        # 画碎石图 (Scree Plot) - 看看每个成分有多重要
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.explained_variance_ratio_) + 1),
                 self.explained_variance_ratio_, 'bo-', linewidth=2)
        plt.title('Scree Plot (Variance Contribution)')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Ratio')
        plt.grid(True)
        plt.show()