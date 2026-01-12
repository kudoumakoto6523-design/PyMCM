# PyMCM: The Ultimate Mathematical Modeling Arsenal 🚀

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![MCM](https://img.shields.io/badge/MCM%2FICM-Ready-orange)

**PyMCM** is a comprehensive, "out-of-the-box" Python library designed specifically for Mathematical Modeling Competitions (COMAP MCM/ICM, CUMCM). 

It encapsulates complex algorithms—from Differential Equations and Neural Networks to Heuristic Optimization and Evaluation Models—into **simple, human-readable APIs**. Stop writing spaghetti code from scratch; focus on modeling, not debugging.

---

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/kudoumakoto/PyMCM.git](https://github.com/yourusername/PyMCM.git)
   cd PyMCM
Install dependencies:
```bash
pip install -r requirements.txt
```
🔥 Quick StartSolve a TOPSIS Evaluation problem in just 3 lines:
```Python
from pymcm.core.data import MCMData
from pymcm.mod.eval.topsis import Topsis
import pandas as pd
```
### 1. Load your data
```
df = pd.DataFrame([[90, 80], [60, 60], [95, 95]], columns=['Math', 'English'])
data = MCMData(df)
```
### 2. Initialize and Run (Auto Entropy Weights)
```
model = Topsis(weights=None) 
model.fit(data).report()
```
### 3. Visualization

```
model.plot()
```
## 🛠️ Features Overview
PyMCM is organized into four main modules covering 90% of competition scenarios.

1. 🔮 Prediction  
 (`pymcm.mod.predict`)Predict the future using time series, regression, or differential equations.  

 | Class | Description |
| :--- | :--- |
| `SmartODESolver` | **Killer Feature**. Solves ODEs symbolically (formulas first). Auto-switches to Numerical (RK45) if symbolic fails. Supports high-order equations. |
| `LSTMModel` | Deep Learning (PyTorch) for complex non-linear time series. |
| `RandomForest` | Robust regression with **Feature Importance** analysis. |
| `ARIMAModel` | Classic time series forecasting. |
| `GreyModel` | GM(1,1) for small datasets. |
| `MarkovChain` | State transition probabilities. |     

 2. ⚖️ Evaluation (`pymcm.mod.eval`)  
 Rank objects or determine weights scientifically.
 
 | Class | Description |
| :--- | :--- |
| `Topsis` | Multi-criteria decision making. Supports **Entropy Weight Method** automatically. |
| `AHP` | Analytic Hierarchy Process for subjective weighting. Includes Consistency Check. |
| `PCAModel` | Principal Component Analysis for dimension reduction and scoring. |
| `FuzzyEval` | Fuzzy Comprehensive Evaluation for qualitative metrics. |

3. 🎯 Optimization (`pymcm.mod.opt`)  
Find the global optimum for complex functions.

| Class | Description |
| :--- | :--- |
| `MCMOptimizer` | A unified interface for **GA (Genetic Algorithm)**, **PSO (Particle Swarm)**, and **SA (Simulated Annealing)**. |
| **Constraint** | Supports complex, non-linear constraints using the **Penalty Function Method**. |

4. 🧩 Utilities & Others (`pymcm.mod.* / pymcm.utils`)
- Clustering: KMeansModel (Includes Elbow Method for finding optimal $K$).
- Graph Theory: GraphModel (Dijkstra Shortest Path, Minimum Spanning Tree).Sensitivity: 
- SensitivityAnalyzer (Analyze robustness of ANY model parameter).
- Visualization: plot_correlation_heatmap (Publication-ready heatmaps).

## 💡 Advanced Usage Examples
### A. Solving Differential Equations (Smart Mode)
PyMCM tries to find the math formula first. If it's too hard, it solves it numerically.Pythonfrom pymcm.mod.predict.ode import SmartODESolver
```python
model = SmartODESolver()

# Define equation: y'' + 2y' + 5y = 0
# Initial conditions: y(0)=2, y'(0)=0
eq = "diff(y, t, 2) + 2*diff(y, t) + 5*y = 0"
ics = [2, 0] 

model.solve(eq, ics, t_span=(0, 10))
model.plot() # Plots the curve and displays the formula if found
```
### B. Sensitivity AnalysisCheck how sensitive your model is to a specific parameter (e.g., Infection Rate $\beta$).
```Python
from pymcm.utils.sensitivity import SensitivityAnalyzer

# 1. Define a wrapper function for your model
def my_model_simulation(beta):
    # ... Re-run your model with new beta ...
    # ... Return the key metric (e.g., max infected) ...
    return result

# 2. Analyze
sa = SensitivityAnalyzer(my_model_simulation, x_name="Beta", y_name="Max Infected")
sa.analyze(base_value=0.5, change_rate=0.2) # Fluctuate +/- 20%
sa.plot()
```
### C. Heuristic Optimization with Constraints Minimize a function with complex logic constraints.
```Python
from pymcm.mod.opt.optimizer import MCMOptimizer

def obj_func(vars):
    x, y = vars
    return x**2 + y**2

# Constraint: x + y > 5  =>  5 - (x + y) <= 0
def constraint_1(vars):
    return 5 - (vars[0] + vars[1])

optimizer = MCMOptimizer()
optimizer.run(obj_func, lb=[-10, -10], ub=[10, 10], 
              method='pso', constraints=[constraint_1])
            
```

## Project Structure

    📂 Project StructurePlaintextpymcm/
    ├── core/             # Base classes and Data wrappers
    ├── mod/              # Main Models
    │   ├── predict/      # LSTM, ARIMA, ODE, RF, Grey, Markov
    │   ├── eval/         # TOPSIS, AHP, PCA, Fuzzy, Entropy
    │   ├── opt/          # GA, PSO, SA Optimizer
    │   ├── cluster/      # K-Means
    │   └── graph/        # Dijkstra, MST
    ├── utils/            # Sensitivity, Visualization tools
    └── examples/         # Demo scripts for modules

