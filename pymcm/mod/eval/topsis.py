import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymcm.core.base import BaseModel
# 引入我们刚才写的熵权工具
from pymcm.mod.eval.entropy import calc_entropy_weights


class Topsis(BaseModel):
    """
    Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS).

    Features:
    - Default: Uses Entropy Weight Method (if weights=None).
    - Manual: Accepts user-defined weights.
    - Output: Score (0-1), Ranking.
    """

    def __init__(self, weights=None):
        """
        Args:
            weights (list/array, optional):
                If None, auto-calculated using Entropy Method during fit().
                If provided, must sum to 1.
        """
        super().__init__(name="TOPSIS")
        self.weights = weights  # Store user preference
        self.scores = None
        self.rankings = None
        self.Z = None  # Normalized matrix

    def _normalize(self, X, directions):
        """
        Vector Normalization & Positive Processing.
        directions: list of 1 (benefit) or -1 (cost).
        """
        X = np.array(X, dtype=float)
        n, m = X.shape
        Z = np.zeros_like(X)

        # 1. Positivization (处理极大型/极小型)
        for j in range(m):
            if directions[j] == -1:  # Cost type (smaller is better)
                # Formula: max - x
                Z[:, j] = np.max(X[:, j]) - X[:, j]
            else:
                Z[:, j] = X[:, j]

        # 2. Vector Normalization (分母为平方和开根号)
        # norm_j = sqrt(sum(x_ij^2))
        for j in range(m):
            norm = np.linalg.norm(Z[:, j])
            if norm == 0: norm = 1
            Z[:, j] = Z[:, j] / norm

        return Z

    def fit(self, mcm_data, directions=None):
        """
        Args:
            mcm_data: Data object.
            directions (list): [1, 1, -1...] (1=High best, -1=Low best). Default all 1.
        """
        X = mcm_data.get_X().astype(float)
        n, m = X.shape

        # Default directions: All are benefit type (1)
        if directions is None:
            directions = [1] * m

        if len(directions) != m:
            raise ValueError(f"Directions length ({len(directions)}) != Features ({m})")

        # 1. Normalize
        self.Z = self._normalize(X, directions)

        # 2. Weight Determination
        if self.weights is None:
            print(f"[{self.name}] No weights provided. Calculating Entropy Weights...")
            # 注意：熵权法通常要求非负数据，Vector Normalization 后的数据是非负的
            # 我们用正向化后的 Z 来算熵权比较合理

            # 为了防止 Z 中有 0 导致熵权计算 log(0)，我们在 calc_entropy_weights 里处理了
            # 但这里我们可以传一个基于 Min-Max 的副本给熵权法（更稳健），
            # 或者直接传 Z (只要 Z 非负)。
            # 为了简单通用，直接传原始 X 的正向化版本给熵权法通常更符合直觉，
            # 但既然我们已经有了 Z，就用 Z 吧。

            # 这里调用熵权工具：
            self.weights = calc_entropy_weights(self.Z)
            print(f"   -> Calculated Weights: {np.round(self.weights, 4)}")
        else:
            self.weights = np.array(self.weights)
            # Normalize weights just in case
            if np.sum(self.weights) != 1:
                self.weights = self.weights / np.sum(self.weights)
            print(f"[{self.name}] Using Manual Weights: {self.weights}")

        # 3. Weighted Matrix
        Z_weighted = self.Z * self.weights

        # 4. Determine Ideal Solutions
        # Z+ (Best), Z- (Worst)
        # Since we positivized everything, Best is always Max, Worst is always Min
        Z_plus = np.max(Z_weighted, axis=0)
        Z_minus = np.min(Z_weighted, axis=0)

        # 5. Calculate Distances
        # D+ = sqrt(sum((z_ij - z_j+)^2))
        D_plus = np.sqrt(np.sum((Z_weighted - Z_plus) ** 2, axis=1))
        D_minus = np.sqrt(np.sum((Z_weighted - Z_minus) ** 2, axis=1))

        # 6. Calculate Score
        # S = D- / (D+ + D-)
        # Avoid division by zero
        denom = D_plus + D_minus
        denom[denom == 0] = 1

        self.scores = D_minus / denom

        # 7. Ranking (Sort descending)
        # argsort gives indices of sorted elements, [::-1] reverses it
        # +1 to make rank start from 1
        # Trick to get rank values:
        temp = np.argsort(-self.scores)
        self.rankings = np.empty_like(temp)
        self.rankings[temp] = np.arange(len(self.scores)) + 1

        self.is_fitted = True
        return self

    def report(self):
        if not self.is_fitted: return
        print("\n--- TOPSIS Report ---")
        df_res = pd.DataFrame({
            'Score': self.scores,
            'Rank': self.rankings
        })
        print(df_res.sort_values(by='Rank'))
        return df_res

    def plot(self):
        if not self.is_fitted: return
        # Sort by score for better visualization
        indices = np.argsort(self.scores)

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), self.scores[indices], color='skyblue')
        plt.yticks(range(len(indices)), [f"Sample {i + 1}" for i in indices])
        plt.xlabel("TOPSIS Score (Closeness to Ideal)")
        plt.title("TOPSIS Evaluation Results")
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.show()

    def _predict_single(self, x):
        return 0