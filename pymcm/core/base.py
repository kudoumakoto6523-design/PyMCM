from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    Abstract Base Class for all PyMCM models.

    Updated to support both:
    1. Regression Prediction (requires X)
    2. Time Series Forecasting (requires steps)
    """

    def __init__(self, name="BaseModel"):
        self.name = name
        self.is_fitted = False
        self.metrics = {}

    @abstractmethod
    def fit(self, mcm_data):
        """Train the model."""
        pass

    def _predict_single(self, x_input):
        """Optional: Helper for regression single sample prediction."""
        raise NotImplementedError("This model does not support single-sample prediction.")

    def _forecast_steps(self, steps):
        """Optional: Helper for time-series forecasting."""
        raise NotImplementedError("This model does not support time-series forecasting.")

    def predict(self, X=None, steps=None):
        """
        Universal Prediction Interface.

        Args:
            X (list/np.array, optional): Input features for Regression tasks.
            steps (int, optional): Number of future steps for Time Series tasks.

        Returns:
            np.array: Prediction results.
        """
        if not self.is_fitted:
            raise Exception(f"Model '{self.name}' is not fitted. Please call .fit() first.")

        # Scenario A: Time Series Forecasting (Prioritize if steps is given)
        if steps is not None:
            if steps < 0:
                raise ValueError("Steps must be non-negative.")
            return self._forecast_steps(steps)

        # Scenario B: Regression Prediction (requires X)
        if X is not None:
            X = np.array(X)
            if X.ndim == 1:
                return self._predict_single(X)
            return np.array([self._predict_single(x) for x in X])

        raise ValueError("You must provide either 'X' (for regression) or 'steps' (for time series).")

    def as_function(self):
        """
        Converts the model to a function f(x).
        Only works for Regression models (requires X input).
        """
        if not self.is_fitted:
            raise Exception("Model not fitted.")

        def func(x):
            x_arr = np.array(x).flatten()
            return self._predict_single(x_arr)

        return func

    def report(self):
        print(f"--- {self.name} Report ---")
        for k, v in self.metrics.items():
            print(f"{k}: {v:.4f}")

    def plot(self):
        print(f"[{self.name}] Plotting not implemented.")