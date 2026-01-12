'''
Scenario: Analyze historical data, predict future economic indicators,
 and evaluate which future year offers the best balance of growth and sustainability. Modules Used: ARIMAModel (Prediction), Topsis (Evaluation), Visualize (Analysis).
'''
import sys
import os
import pandas as pd
import numpy as np

# Ensure pymcm is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymcm.core.data import MCMData
from pymcm.utils.visualize import plot_correlation_heatmap
from pymcm.mod.predict.arima import ARIMAModel
from pymcm.mod.eval.topsis import Topsis

print("=== Integrated Demo 1: Sustainability Analysis Pipeline ===")

# 1. Load Data
csv_path = 'test_clean.csv'
# Handle path if running from examples folder
if not os.path.exists(csv_path):
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'test_clean.csv')

print(f"[Step 1] Loading {csv_path}...")
df = pd.read_csv(csv_path)
mcm_data = MCMData(df)
print(df.head())

# 2. Correlation Analysis
print("\n[Step 2] Analyzing Correlation...")
# Visualize relationships between GDP, Energy, and CO2
try:
    plot_correlation_heatmap(df[['GDP', 'Energy', 'CO2']], title="Economic Indicators")
    print("   -> Heatmap displayed.")
except Exception as e:
    print(f"   -> Could not plot: {e}")

# 3. Predict Future (2025-2029)
print("\n[Step 3] Predicting Future (2025-2029)...")
future_years = range(2025, 2030)
future_df = pd.DataFrame({'Year': future_years})

for col in ['GDP', 'Energy', 'CO2']:
    print(f"  -> Training ARIMA for {col}...")
    model = ARIMAModel(1, 1, 1)
    sub_data = MCMData(df[[col]])
    model.fit(sub_data)
    preds = model.predict(steps=5)
    future_df[col] = preds

print("\nPredicted Data:")
print(future_df)

# 4. Evaluate Sustainability
print("\n[Step 4] Evaluating Future Scenarios...")
# Criteria:
# GDP -> High is better (+1)
# Energy -> Low is better (-1)
# CO2 -> Low is better (-1)
eval_data = MCMData(future_df[['GDP', 'Energy', 'CO2']])
directions = [1, -1, -1]

topsis = Topsis(weights=None) # Auto Entropy Weights
topsis.fit(eval_data, directions=directions)

future_df['Score'] = topsis.scores
future_df['Rank'] = topsis.rankings

print("\nFinal Evaluation (Rank 1 is Best):")
print(future_df.sort_values(by='Rank'))