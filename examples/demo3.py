'''
Scenario: Model CO2 growth using a differential equation based on the data, and analyze how sensitive the final 2024 prediction
is to the growth rate parameter. Modules Used: SensitivityAnalyzer, SmartODESolver (conceptually), scipy.
'''
import sys
import os
import pandas as pd
import numpy as np
from scipy.integrate import odeint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pymcm.utils.sensitivity import SensitivityAnalyzer
# We import SmartODESolver just to show it's part of the stack
from pymcm.mod.predict.ode import SmartODESolver

print("=== Integrated Demo 3: CO2 Growth Simulation & Sensitivity ===")

# 1. Load Data
csv_path = 'test_clean.csv'
if not os.path.exists(csv_path):
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'test_clean.csv')

df = pd.read_csv(csv_path)
initial_co2 = df['CO2'].iloc[0]  # Year 2000
final_co2_actual = df['CO2'].iloc[-1]  # Year 2024
print(f"Actual CO2 in 2000: {initial_co2}")
print(f"Actual CO2 in 2024: {final_co2_actual}")


# 2. Define Simulation Wrapper
# Model: dC/dt = r * C (Exponential Growth)
# Goal: See how growth rate 'r' affects the predicted value in year 2024
def simulate_co2(rate):
    # This wrapper function simulates the whole model run
    # Simple exponential growth model
    def model(y, t):
        return rate * y

    t = np.linspace(0, 24, 25)  # 0 to 24 years (2000-2024)
    y0 = [initial_co2]
    ret = odeint(model, y0, t)
    return ret[-1][0]  # Return value at year 2024


# 3. Baseline Run
# Estimate 'r' roughly: 203 * exp(r*24) = 397 -> r ~ 0.028
baseline_rate = 0.028
pred_baseline = simulate_co2(baseline_rate)
print(f"Baseline Prediction (r={baseline_rate:.3f}): {pred_baseline:.2f}")

# 4. Sensitivity Analysis
print("\n[Step 2] Sensitivity Analysis of Growth Rate...")
analyzer = SensitivityAnalyzer(
    model_func=simulate_co2,
    x_name="Growth Rate (r)",
    y_name="Predicted CO2 (2024)"
)

# Analyze +/- 20% fluctuation
analyzer.analyze(base_value=baseline_rate, change_rate=0.2, steps=10)
analyzer.plot()