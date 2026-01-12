'''
Scenario: Classify years into different "Development Stages" based on GDP and Energy, then find the optimal representative center for the "Developed" stage. Modules Used: KMeansModel (Clustering), MCMOptimizer (Optimization).
'''
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pymcm.core.data import MCMData
from pymcm.mod.cluster.kmeans import KMeansModel
from pymcm.mod.opt.optimizer import MCMOptimizer

print("=== Integrated Demo 2: Economic Clustering & Center Optimization ===")

# 1. Load Data
csv_path = 'test_clean.csv'
if not os.path.exists(csv_path):
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'test_clean.csv')

print(f"[Step 1] Loading {csv_path}...")
df = pd.read_csv(csv_path)

# Use GDP and Energy as development features
X_df = df[['GDP', 'Energy']]
mcm_data = MCMData(X_df)

# 2. Clustering
print("\n[Step 2] Clustering Years into 3 Development Stages...")
kmeans = KMeansModel(n_clusters=3)
kmeans.fit(mcm_data)
df['Cluster'] = kmeans.labels
print("Cluster Centers (GDP, Energy):")
print(kmeans.centers)

# 3. Optimization
# Task: Find a 'Ideal Point' (Center of Mass) for the 'Most Developed' cluster
# We define 'Most Developed' as the cluster with highest average GDP
cluster_means = df.groupby('Cluster')['GDP'].mean()
target_cluster = cluster_means.idxmax()
target_points = df[df['Cluster'] == target_cluster][['GDP', 'Energy']].values

print(f"\n[Step 3] Optimizing Center for Developed Cluster (ID={target_cluster})...")

def dist_func(vars):
    x, y = vars
    # Objective: Minimize sum of distances to all points in this cluster
    dists = np.sqrt(np.sum((target_points[:, 0] - x)**2 + (target_points[:, 1] - y)**2))
    return dists

optimizer = MCMOptimizer()
# Search bounds based on data range
lb = [X_df['GDP'].min(), X_df['Energy'].min()]
ub = [X_df['GDP'].max(), X_df['Energy'].max()]

best_loc, min_dist = optimizer.run(dist_func, lb, ub, method='pso', pop_size=30)
print(f"  -> Optimal Center: GDP={best_loc[0]:.2f}, Energy={best_loc[1]:.2f}")