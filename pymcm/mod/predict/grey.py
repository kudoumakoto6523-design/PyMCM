import numpy as np
import matplotlib.pyplot as plt
from pymcm.core.base import BaseModel


class GreyModel(BaseModel):
    """
    Grey Prediction Model GM(1,1) with Stepwise Ratio Test.

    Features:
    - Automatic Stepwise Ratio Test (级比检验).
    - Automatic Data Translation (平移变换) if the test fails.
    """

    def __init__(self, max_iter):
        super().__init__(name="GreyModel GM(1,1)")
        self.a = None
        self.b = None
        self.x0 = None  # The processed data (after optional translation)
        self.raw_x0 = None  # The original user data
        self.c = 0  # Translation constant (平移常数)
        self.max_iter = max_iter

    def fit(self, mcm_data):
        """
        Fit GM(1,1) model.
        Automatically performs the Stepwise Ratio Test and translates data if needed.
        """
        y = mcm_data.get_y()
        # Fallback logic
        if y is None:
            X = mcm_data.get_X()
            if X.shape[1] > 0:
                y = X[:, 0]
            else:
                raise ValueError("GreyModel requires a data sequence.")

        self.raw_x0 = y.copy()
        n = len(y)

        # --- Step 1: Stepwise Ratio Test (级比检验) ---
        # Range: (e^(-2/(n+1)), e^(2/(n+1)))
        lower = np.exp(-2 / (n + 1))
        upper = np.exp(2 / (n + 1))

        # Check if translation is needed
        self.c = 0
        max_iter = self.max_iter
        is_passed = False

        # Iterate to find a suitable constant 'c' if ratio test fails
        for i in range(max_iter):
            temp_x0 = self.raw_x0 + self.c

            # Prevent division by zero
            if np.any(temp_x0 == 0):
                self.c += 1
                continue

            # Lambda(k) = x(k-1) / x(k)
            # Note: Standard definition usually compares k-1 and k
            ratios = temp_x0[:-1] / temp_x0[1:]

            # Check if all ratios are within range
            if np.all((ratios > lower) & (ratios < upper)):
                is_passed = True
                break
            else:
                # If failed, add a small shift and retry
                # Heuristic: Add 10% of mean value or simply +1 depending on magnitude
                step = max(np.abs(np.mean(self.raw_x0)) * 0.05, 1.0)
                self.c += step

        if not is_passed:
            print(
                f"⚠️ [Warning] GM(1,1) Ratio Test failed even after translation (c={self.c}). Model accuracy might be low.")
        elif self.c > 0:
            print(f"ℹ️ [Info] Ratio Test passed with translation constant c={self.c:.2f}.")
        else:
            print("✅ [Info] Ratio Test passed directly.")

        # Use the translated data for modeling
        self.x0 = self.raw_x0 + self.c

        # --- Step 2: GM(1,1) Modeling ---
        # 1-AGO
        x1 = np.cumsum(self.x0)

        # B Matrix & Y Vector
        z1 = (x1[:-1] + x1[1:]) / 2.0
        B = np.vstack([-z1, np.ones(n - 1)]).T
        Y = self.x0[1:].reshape(-1, 1)

        # Least Squares
        try:
            self.params = np.linalg.inv(B.T @ B) @ B.T @ Y
            self.a = self.params[0][0]
            self.b = self.params[1][0]
        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix error. Data might be collinear or constant.")

        self.is_fitted = True

        # Self-Check Evaluation
        preds_all = self._forecast_steps(steps=0)
        y_pred_history = preds_all[:n]

        # Compute MAPE
        # Ignore divide by zero for metrics
        with np.errstate(divide='ignore', invalid='ignore'):
            err = np.abs((self.raw_x0 - y_pred_history) / self.raw_x0)
            err = np.nan_to_num(err)  # Handle 0/0
            self.metrics['MAPE'] = np.mean(err)

        return self

    def _forecast_steps(self, steps):
        """
        Internal forecasting logic.
        IMPORTANT: Must subtract 'c' at the end to restore original scale.
        """
        n = len(self.x0)
        total_len = n + steps

        # Time Response Function
        preds_x1 = []
        x0_1 = self.x0[0]
        val_constant = (x0_1 - self.b / self.a)

        for k in range(total_len):
            # x1(k) calculation
            val = val_constant * np.exp(-self.a * k) + self.b / self.a
            preds_x1.append(val)

        # IAGO (Inverse Accumulate)
        preds_x1 = np.array(preds_x1)
        preds_x0 = np.zeros(total_len)
        preds_x0[0] = x0_1
        for k in range(1, total_len):
            preds_x0[k] = preds_x1[k] - preds_x1[k - 1]

        # --- CRITICAL: Restore Data (Subtract c) ---
        return preds_x0 - self.c

    def as_function(self):
        """
        Convert the trained GM(1,1) model into a continuous Python function f(t).

        This allows:
        1. Interpolation (e.g., predict at t=2.5).
        2. Inverse solving (e.g., find t when value reaches threshold).

        Returns:
            function: A callable 'func(t)' where t is the time index (0, 1, 2...).
        """
        if not self.is_fitted:
            raise Exception("Model is not fitted. Cannot convert to function.")

        # Pre-calculate constants for closure to improve performance
        x0_1 = self.x0[0]
        val_constant = (x0_1 - self.b / self.a)
        a = self.a
        b = self.b
        c = self.c

        def func(t):
            """
            Predict value at time t (can be float).
            """
            # Handle list/array input (take first element)
            if hasattr(t, '__len__') and not isinstance(t, str):
                t = t[0]

            k = float(t)

            # 1. Calculate accumulated value at k
            x1_k = val_constant * np.exp(-a * k) + b / a

            # 2. Calculate accumulated value at k-1 (Discrete derivative approximation)
            if k == 0:
                pred_val = x0_1
            else:
                x1_k_minus_1 = val_constant * np.exp(-a * (k - 1)) + b / a
                pred_val = x1_k - x1_k_minus_1

            # 3. Restore translation
            return pred_val - c

        return func

    def plot(self):
        """Plot with translation handling hidden from user."""
        if not self.is_fitted:
            return

        # Predict history
        preds = self._forecast_steps(0)

        plt.figure(figsize=(10, 5))
        plt.plot(self.raw_x0, 'b-o', label='Actual Data')
        plt.plot(preds, 'r--', label='GM(1,1) Fitted')
        plt.title(f"GM(1,1) Prediction (MAPE={self.metrics['MAPE']:.4f})\nTranslation c={self.c:.2f}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()