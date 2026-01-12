import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 引入我们的父类
from pymcm.core.base import BaseModel


class Regressor(BaseModel):
    """
    Universal Regressor for PyMCM.

    Supports:
    - Linear Regression ('linear')
    - Polynomial Regression ('poly')
    - Ridge Regression ('ridge')

    It automatically handles scaling, training, metric calculation, and visualization.
    """

    def __init__(self, method='linear', degree=2, alpha=1.0):
        """
        Args:
            method (str): 'linear', 'poly', 'ridge', or 'lasso'.
            degree (int): Degree for polynomial regression (only used if method='poly').
            alpha (float): Regularization strength (only for 'ridge'/'lasso').
        """
        super().__init__(name=f"Regressor({method})")
        self.method = method
        self.degree = degree

        # Build the model pipeline based on user choice
        if method == 'linear':
            self.model = LinearRegression()
        elif method == 'poly':
            # Polynomial requires generating features -> then linear regression
            self.model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            self.name = f"PolyRegressor(degree={degree})"
        elif method == 'ridge':
            self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        elif method == 'lasso':
            self.model = make_pipeline(StandardScaler(), Lasso(alpha=alpha))
        else:
            raise ValueError(f"Unknown method: {method}")

    def fit(self, mcm_data):
        """
        Train the model using MCMData container.
        """
        # 1. Get Data
        X = mcm_data.get_X()
        y = mcm_data.get_y()

        if y is None:
            raise ValueError("Regression requires a target (y). Please specify 'target' in MCMData.")

        # 2. Train
        self.model.fit(X, y)
        self.is_fitted = True

        # 3. Evaluate immediately (Self-Check)
        y_pred = self.model.predict(X)

        # Store metrics
        self.metrics['R2'] = r2_score(y, y_pred)
        self.metrics['MSE'] = mean_squared_error(y, y_pred)
        self.metrics['RMSE'] = np.sqrt(self.metrics['MSE'])
        self.metrics['MAE'] = mean_absolute_error(y, y_pred)

        # Store data for plotting later
        self._X_train = X
        self._y_train = y
        self._y_pred_train = y_pred

        return self  # Allow chain calling

    def _predict_single(self, x_input):
        # x_input is 1D array, reshape to (1, n_features)
        return self.model.predict(x_input.reshape(1, -1))[0]

    def plot(self):
        """
        Visualize the regression results.
        Draws two charts:
        1. Actual vs Predicted (Fit Check)
        2. Residuals (Error Distribution)
        """
        if not self.is_fitted:
            print("Model not fitted. Cannot plot.")
            return

        plt.figure(figsize=(12, 5))

        # Plot 1: Actual vs Predicted
        plt.subplot(1, 2, 1)
        plt.scatter(range(len(self._y_train)), self._y_train, label='Actual', color='blue', alpha=0.6)
        plt.plot(range(len(self._y_train)), self._y_pred_train, label='Predicted', color='red', linewidth=2)
        plt.title(f"{self.name} - Fit Result")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        # Plot 2: Residuals vs Predicted (Check for patterns)
        residuals = self._y_train - self._y_pred_train
        plt.subplot(1, 2, 2)
        plt.scatter(self._y_pred_train, residuals, color='green', alpha=0.6)
        plt.axhline(y=0, color='black', linestyle='--')
        plt.title("Residual Analysis")
        plt.xlabel("Predicted Value")
        plt.ylabel("Residuals (Actual - Pred)")
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()