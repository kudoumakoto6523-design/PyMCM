import numpy as np
from pymcm.core.base import BaseModel


class AHP(BaseModel):
    """
    Analytic Hierarchy Process (AHP).

    Used to determine weights based on a pairwise comparison matrix (Judgment Matrix).
    Includes Consistency Check (CR test).
    """

    def __init__(self):
        super().__init__(name="AHP")
        self.weights = None
        self.CR = None  # Consistency Ratio
        self.is_consistent = False

        # RI 表 (平均随机一致性指标) - 查表用
        self.RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12,
                        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}

    def fit(self, judgment_matrix):
        """
        Args:
            judgment_matrix (list/array): NxN matrix where a_ij = importance of i over j.
        """
        A = np.array(judgment_matrix, dtype=float)
        n = A.shape[0]

        # 1. 计算权重 (特征值法)
        # 求最大特征值及其对应的特征向量
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # 找到最大特征值的索引
        max_idx = np.argmax(eigenvalues)
        lambda_max = np.real(eigenvalues[max_idx])
        w_vector = np.real(eigenvectors[:, max_idx])

        # 归一化特征向量得到权重
        self.weights = w_vector / np.sum(w_vector)

        # 2. 一致性检验
        # CI = (lambda_max - n) / (n - 1)
        CI = (lambda_max - n) / (n - 1)

        if n in self.RI_dict:
            RI = self.RI_dict[n]
        else:
            RI = 1.45  # n > 9 近似处理

        if RI == 0:
            self.CR = 0
        else:
            self.CR = CI / RI

        print(f"[{self.name}] Max Eigenvalue: {lambda_max:.4f}")
        print(f"[{self.name}] Consistency Ratio (CR): {self.CR:.4f}")

        if self.CR < 0.1:
            print("✅ Consistency Check Passed (CR < 0.1).")
            self.is_consistent = True
        else:
            print("⚠️ Consistency Check FAILED (CR >= 0.1). Please adjust the matrix.")
            self.is_consistent = False

        self.is_fitted = True
        return self

    def get_weights(self):
        return self.weights

    def _predict_single(self, x):
        return 0

    def plot(self):
        pass