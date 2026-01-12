import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
from scipy.stats import spearmanr, rankdata, t, norm
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA as StatsARIMA  # Updated import

from pymcm.core.base import BaseModel


class ARIMAModel(BaseModel):
    """
    ARIMA Model Wrapper (Supports Auto-ARIMA & Manual-ARIMA).

    Features:
    - Automatic Trend/Stability Check (Daniel Test).
    - Auto-Tuning via pmdarima (AIC minimization).
    - Manual Parameter support (p, d, q).
    - Visualization with Confidence Intervals.
    """

    def __init__(self, p=None, d=None, q=None, alpha=0.05):
        """
        Args:
            p, d, q (int, optional): Order for manual ARIMA. If None, Auto-ARIMA is used.
            alpha (float): Significance level for trend test and confidence intervals.
        """
        super().__init__(name='ARIMAModel')
        self.p = p
        self.d = d
        self.q = q
        self.alpha = alpha
        self.model_res = None  # Stores the fitted model object
        self.auto_mode = False  # Flag to track if we used Auto-ARIMA
        self.history = None  # Store original data for plotting
        self.metrics = {}  # Initialize metrics dict

    def _trend_check(self, Y):
        """
        Performs Daniel Trend Test (Spearman Rank Correlation).
        checks if the sequence has a significant trend (Not Stable).
        """
        y = np.array(Y).flatten()
        n = len(y)
        time_idx = np.arange(n)

        # 1. Calculate Spearman Correlation (r)
        # r is equivalent to the Daniel statistic q
        stat_r, p_val_scipy = spearmanr(y, time_idx)

        self.metrics['Spearman_r'] = stat_r

        # 2. Manual T-Test Verification (For academic rigor)
        # t = r * sqrt((n-2) / (1-r^2))
        if abs(stat_r) == 1.0:
            t_val = float('inf')
        else:
            t_val = stat_r * np.sqrt((n - 2) / (1 - stat_r ** 2))

        df = n - 2
        # Two-tailed P-value calculation using t-distribution
        p_val_manual = 2 * (1 - t.cdf(abs(t_val), df))

        self.metrics['P-Value'] = p_val_manual

        print(f"--- Trend Check (Daniel Test) ---")
        print(f"Statistic(r): {stat_r:.4f}, T-Score: {t_val:.4f}")
        print(f"P-Value: {p_val_manual:.4f} (Threshold: {self.alpha})")

        # 3. Conclusion
        # If P < 0.05, we reject H0 (No Trend). Thus, data is NOT stable.
        if p_val_manual < self.alpha:
            print("⚠️ Result: Significant Trend detected (Sequence is NOT stable).")
            print("   -> ARIMA will likely need differencing (d >= 1).")
        else:
            print("✅ Result: No significant trend detected (Sequence appears stable).")

    def fit(self, mcm_data):
        """
        Train the model.
        """
        # 1. Get Data
        Y = mcm_data.get_y()
        # Fallback if Y is not set
        if Y is None:
            raw_X = mcm_data.get_X()
            if raw_X.shape[1] > 0:
                Y = raw_X[:, 0]
            else:
                raise ValueError("ARIMA requires a 1D target sequence.")

        self.history = np.array(Y).flatten()

        # 2. Check Data Stability
        self._trend_check(self.history)

        # 3. Model Training
        print(f"\n[{self.name}] Start Fitting...")

        # --- Mode A: Auto-ARIMA (pmdarima) ---
        if self.p is None and self.d is None and self.q is None:
            print("-> Mode: Auto-ARIMA (Searching for optimal params...)")
            self.auto_mode = True

            # Use pmdarima to find best model
            self.model_res = auto_arima(
                self.history,
                start_p=1, start_q=1,
                max_p=5, max_q=5,
                m=1,  # Set m=1 for non-seasonal data by default
                seasonal=False,  # Set True if seasonality is suspected
                d=None,  # Let auto_arima find 'd'
                trace=True,  # Print search progress
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            print("-> Best Model Found:")
            print(self.model_res.summary())

        # --- Mode B: Manual ARIMA (statsmodels) ---
        else:
            print(f"-> Mode: Manual ARIMA with order ({self.p}, {self.d}, {self.q})")
            self.auto_mode = False

            # Validate input
            if None in [self.p, self.d, self.q]:
                raise ValueError("For Manual Mode, p, d, and q must all be integers.")

            # Use statsmodels
            model = StatsARIMA(self.history, order=(self.p, self.d, self.q))
            self.model_res = model.fit()
            print(self.model_res.summary())

        self.is_fitted = True
        return self

    def predict(self, X= None, steps=5):
        """
        Forecast future steps.
        Handles API differences between pmdarima and statsmodels.
        """
        if not self.is_fitted:
            raise Exception("Model not fitted.")

        # Unified interface for prediction
        if self.auto_mode:
            # pmdarima .predict() returns value and intervals
            forecast, conf_int = self.model_res.predict(n_periods=steps, return_conf_int=True, alpha=self.alpha)
            self._last_conf_int = conf_int
            return forecast
        else:
            # statsmodels .get_forecast()
            forecast_res = self.model_res.get_forecast(steps=steps)
            forecast = forecast_res.predicted_mean
            self._last_conf_int = forecast_res.conf_int(alpha=self.alpha)
            return forecast

    def _predict_single(self, x):
        return 0  # Not used for time series

    def plot(self, steps=5):
        """
        Visualize History, Fit, Forecast, and Confidence Intervals.
        """
        if not self.is_fitted:
            print("Model not fitted.")
            return

        # 1. Generate Forecast
        future_pred = self.predict(steps=steps)
        conf_int = self._last_conf_int

        # 2. Get In-Sample Fitted Values (History)
        if self.auto_mode:
            # pmdarima provides in_sample_preds
            hist_pred = self.model_res.predict_in_sample()
            # Often the first 'd' points are 0 or NaN, we handle alignment in plot
        else:
            hist_pred = self.model_res.fittedvalues

        # 3. Setup Plot Indices
        n = len(self.history)
        hist_idx = np.arange(n)
        future_idx = np.arange(n, n + steps)

        plt.figure(figsize=(12, 6))

        # A. Plot Actual Data
        plt.plot(hist_idx, self.history, 'k.-', label='Observed Data', linewidth=1.5)

        # B. Plot Historical Fit
        # Usually skip the first point due to differencing
        plt.plot(hist_idx[1:], hist_pred[1:], 'b--', label='In-Sample Fit', alpha=0.7)

        # C. Plot Future Forecast
        plt.plot(future_idx, future_pred, 'r.-', label=f'Forecast ({steps} steps)', linewidth=2)

        # D. Plot Confidence Intervals (The "Gray Area")
        # conf_int is usually [[low, high], [low, high]...]
        lower_bound = conf_int[:, 0]
        upper_bound = conf_int[:, 1]
        plt.fill_between(future_idx, lower_bound, upper_bound, color='red', alpha=0.15,
                         label=f'{int((1 - self.alpha) * 100)}% Confidence Interval')

        plt.title(
            f"ARIMA Forecast Results\nMode: {'Auto-ARIMA' if self.auto_mode else f'Manual ({self.p},{self.d},{self.q})'}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()