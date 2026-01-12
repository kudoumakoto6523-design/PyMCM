import numpy as np
import matplotlib.pyplot as plt


class SensitivityAnalyzer:
    """
    Universal Sensitivity Analyzer (Black-box approach).

    Logic:
    1. Input a 'wrapper function' model_func(param_val).
    2. Vary param_val within a certain range.
    3. Repeatedly call model_func and record the results.
    """

    def __init__(self, model_func, x_name="Parameter", y_name="Result"):
        """
        Args:
            model_func (callable): The wrapper function.
                                   Input: float (current parameter value)
                                   Output: float (model result metric)
            x_name (str): Name of the parameter (X-axis label)
            y_name (str): Name of the result (Y-axis label)
        """
        self.func = model_func
        self.x_name = x_name
        self.y_name = y_name
        self.results = []
        self.param_values = []

    def analyze(self, base_value, change_rate=0.1, steps=10):
        """
        Args:
            base_value (float): The baseline value of the parameter.
            change_rate (float): Fluctuation ratio (0.1 means +/- 10%).
            steps (int): Number of sampling steps (how many times to re-run the model).
        """
        # Generate parameter range
        low = base_value * (1 - change_rate)
        high = base_value * (1 + change_rate)
        self.param_values = np.linspace(low, high, steps)
        self.results = []

        print(f"[Sensitivity] Analyzing '{self.x_name}' -> '{self.y_name}'")
        print(f"   Range: [{low:.4f}, {high:.4f}], Steps: {steps}")

        # --- Core Loop: Re-run the model repeatedly ---
        for i, val in enumerate(self.param_values):
            # self.func is the user-defined function to re-run the whole model
            try:
                res = self.func(val)
                self.results.append(res)
                # Print progress because re-running models might be slow
                print(f"   Step {i + 1}/{steps}: {self.x_name}={val:.4f} => {self.y_name}={res:.4f}")
            except Exception as e:
                print(f"   Step {i + 1}/{steps}: Failed ({e})")
                self.results.append(None)  # Placeholder for failure

        return self.param_values, self.results

    def plot(self):
        if not self.results: return

        # Filter out None values (failed runs)
        valid_x = [x for x, r in zip(self.param_values, self.results) if r is not None]
        valid_y = [r for r in self.results if r is not None]

        if len(valid_y) < 2:
            print("Not enough valid results to plot.")
            return

        # Calculate Coefficient of Variation (CV) as a sensitivity metric
        # High CV -> Sensitive (Small change in param leads to huge change in result)
        # Low CV -> Robust (Result is stable)
        mean_val = np.mean(valid_y)
        if mean_val == 0: mean_val = 1e-6

        cv = np.std(valid_y) / mean_val

        plt.figure(figsize=(8, 5))
        plt.plot(valid_x, valid_y, 'o-', linewidth=2, color='teal')

        # Mark the baseline (center point)
        mid_idx = len(valid_x) // 2
        plt.plot(valid_x[mid_idx], valid_y[mid_idx], 'r*', markersize=15, label='Base Value')

        plt.title(f"Sensitivity: {self.x_name} vs {self.y_name}\n(Sensitivity Score / CV: {cv:.4f})")
        plt.xlabel(f"{self.x_name} Value")
        plt.ylabel(f"{self.y_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()