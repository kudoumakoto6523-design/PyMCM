import numpy as np
import pandas as pd
from pymcm.core.base import BaseModel
from pymcm.mod.eval.entropy import calc_entropy_weights


class FuzzyEval(BaseModel):
    """
    Fuzzy Comprehensive Evaluation (FCE).

    Model: B = W * R
    W: Weight vector (1 x m)
    R: Membership matrix (m x n)
    B: Evaluation result (1 x n)
    """

    def __init__(self, weights=None, operator='M(*, +)'):
        super().__init__(name="Fuzzy Eval")
        self.weights = weights
        self.result_vector = None

    def fit(self, mcm_data=None, R=None):
        """
        Args:
            mcm_data: Optional. Raw data to calculate Entropy Weights if weights is None.
            R: (Required) Membership Matrix (m factors x n levels).
               每一行代表一个因素在不同评价等级上的隶属度。
        """
        if R is None:
            raise ValueError("Membership Matrix R must be provided.")
        R = np.array(R)

        # 1. 确定权重 W
        if self.weights is None:
            if mcm_data is not None:
                print(f"[{self.name}] Using Entropy Method to determine weights from raw data...")
                # 注意：这里假设 mcm_data 的列数等于 R 的行数（因素数）
                self.weights = calc_entropy_weights(mcm_data.get_X())
            else:
                raise ValueError("No weights provided and no raw data for Entropy calculation.")
        else:
            self.weights = np.array(self.weights)

        # 归一化权重
        if np.sum(self.weights) != 1.0:
            self.weights = self.weights / np.sum(self.weights)

        # 2. 模糊合成 B = W * R
        # (1, m) * (m, n) -> (1, n)
        # 这里使用常用的加权平均算子 M(*, +) -> 矩阵乘法
        self.result_vector = np.dot(self.weights, R)

        print(f"[{self.name}] Evaluation Result Vector: {np.round(self.result_vector, 4)}")

        # 归一化结果 (如果是求概率分布)
        # self.result_vector = self.result_vector / np.sum(self.result_vector)

        self.is_fitted = True
        return self

    def _predict_single(self, x):
        return 0