import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pymcm.core.base import BaseModel


class RandomForest(BaseModel):
    """
    Random Forest Regressor.

    Best for: Multi-variable regression & Feature Importance analysis.
    Key Advantage: Robust, handles non-linear relationships, interpretable importance.
    """

    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Args:
            n_estimators (int): Number of trees (default 100).
            max_depth (int): Max depth of trees (None means infinite).
            random_state (int): Seed for reproducibility.
        """
        super().__init__(name="Random Forest")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )
        self.feature_names = None
        self.feature_importances = None

    def fit(self, mcm_data):
        """
        Fit the model using X (features) and y (target).
        """
        # 1. 获取数据
        X = mcm_data.get_X()
        y = mcm_data.get_y()

        if y is None:
            raise ValueError("Random Forest requires a target (y) to train.")

        # 记录特征名字（如果有的话，画图用）
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        else:
            # 如果是 numpy array，就自动命名 Feature 1, Feature 2...
            self.feature_names = [f"Feat_{i}" for i in range(X.shape[1])]

        # 2. 训练
        # ravel() 将 y 变成一维数组
        self.model.fit(X, y.ravel() if hasattr(y, 'ravel') else y)

        # 3. 提取特征重要性
        self.feature_importances = self.model.feature_importances_
        self.is_fitted = True

        # 打印简单的评估指标 (R2 Score)
        score = self.model.score(X, y)
        print(f"[{self.name}] Training R2 Score: {score:.4f} (1.0 is perfect)")

        return self

    def predict(self, X_new):
        """
        Predict new data.
        Args:
            X_new: 2D array or DataFrame of features.
        """
        if not self.is_fitted:
            raise Exception("Model not fitted.")
        return self.model.predict(X_new)

    def _predict_single(self, x):
        return 0

    def plot_importance(self):
        """
        专门用来画特征重要性的图。
        """
        if not self.is_fitted: return

        # 排序
        indices = np.argsort(self.feature_importances)[::-1]
        sorted_names = [self.feature_names[i] for i in indices]
        sorted_scores = self.feature_importances[indices]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance (What matters most?)")
        plt.bar(range(len(sorted_scores)), sorted_scores, color='teal', align='center')
        plt.xticks(range(len(sorted_scores)), sorted_names, rotation=45)
        plt.ylabel("Importance Score (Sum=1)")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot(self):
        """
        Redirect generic plot to importance plot.
        """
        self.plot_importance()