import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pymcm.core.base import BaseModel


class KMeansModel(BaseModel):
    """
    K-Means Clustering.

    Features:
    - Auto-scaling (StandardScaler).
    - Elbow Method (Find optimal K).
    - 2D Visualization (using PCA if dimensions > 2).
    """

    def __init__(self, n_clusters=3, random_state=42):
        super().__init__(name="K-Means")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.labels = None
        self.centers = None
        self.scaler = StandardScaler()
        self.X_scaled = None

    def find_optimal_k(self, mcm_data, max_k=10):
        """
        Elbow Method Visualization.
        Returns the inertia values to help user decide K.
        """
        X = mcm_data.get_X()
        X_scaled = self.scaler.fit_transform(X)

        inertias = []
        K_range = range(1, max_k + 1)

        print(f"[{self.name}] Running Elbow Method (k=1 to {max_k})...")
        for k in K_range:
            km = SklearnKMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)

        # Plot
        plt.figure(figsize=(8, 4))
        plt.plot(K_range, inertias, 'bo-', linewidth=2)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia (Sum of Squared Distances)')
        plt.title('Elbow Method For Optimal k')
        plt.grid(True)
        plt.show()

        return inertias

    def fit(self, mcm_data):
        X = mcm_data.get_X()
        # 1. 归一化 (聚类对距离非常敏感)
        self.X_scaled = self.scaler.fit_transform(X)

        # 2. 训练
        self.model = SklearnKMeans(n_clusters=self.n_clusters,
                                   random_state=self.random_state,
                                   n_init=10)
        self.model.fit(self.X_scaled)

        self.labels = self.model.labels_
        self.centers = self.model.cluster_centers_
        self.is_fitted = True

        print(f"[{self.name}] Clustering finished. Found {self.n_clusters} clusters.")
        return self

    def plot(self):
        """
        Visualize clusters.
        If dim > 2, use PCA to reduce to 2D for plotting.
        """
        if not self.is_fitted: return

        # 如果维度 > 2，用 PCA 降维画图
        if self.X_scaled.shape[1] > 2:
            pca = PCA(n_components=2)
            X_vis = pca.fit_transform(self.X_scaled)
            print(f"[{self.name}] Dimensions > 2. Using PCA for 2D visualization.")
        else:
            X_vis = self.X_scaled

        plt.figure(figsize=(8, 6))

        # 画散点，颜色根据分类标签
        scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=self.labels, cmap='viridis', s=50, alpha=0.7)

        # 画中心点 (注意中心点也要降维)
        if self.X_scaled.shape[1] > 2:
            # 这里中心点降维其实不完全准确，只是为了大概看位置，严谨的做法只画样本
            pass
        else:
            plt.scatter(self.centers[:, 0], self.centers[:, 1], c='red', s=200, marker='X', label='Centroids')

        plt.title(f"K-Means Clustering Results (k={self.n_clusters})")
        plt.colorbar(scatter, label='Cluster ID')
        plt.grid(True, alpha=0.3)
        plt.show()

    def _predict_single(self, x):
        return 0
    