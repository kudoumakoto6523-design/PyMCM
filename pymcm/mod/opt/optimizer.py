import numpy as np
import matplotlib.pyplot as plt
from pymcm.core.base import BaseModel


class MCMOptimizer(BaseModel):
    """
    Heuristic Optimization Toolkit (With Auto-Constraint Support).

    Supports: PSO, GA, SA.
    Constraint Logic: Uses Penalty Function Method internally.
    """

    def __init__(self):
        super().__init__(name="MCM Optimizer")
        self.history = []
        self.best_x = None
        self.best_y = np.inf
        self.method_used = ""
        self.func = None
        self.constraints = []  # Store constraint functions

    def _compute_fitness(self, x):
        """
        Internal helper: Calculates Objective + Penalty.
        Constraint convention: g(x) <= 0 is Valid. g(x) > 0 is Violation.
        """
        # 1. Calculate raw objective
        obj_val = self.func(x)

        # 2. Add penalties if constraints exist
        if not self.constraints:
            return obj_val

        penalty = 0
        penalty_factor = 1e7  # A very large number (Big-M)

        for g in self.constraints:
            violation = g(x)
            # If violation > 0, constraint is broken
            if violation > 0:
                # Penalty = Fixed Cost + Linear Cost (to guide optimizer back)
                penalty += penalty_factor + (violation * penalty_factor)

        return obj_val + penalty

    def run(self, func, lb, ub, method='pso', constraints=None, **kwargs):
        """
        Args:
            func: Objective function.
            lb, ub: Bounds.
            method: 'pso', 'ga', 'sa'.
            constraints (list): List of functions [g1, g2].
                                Each g(x) should return <= 0 if valid.
        """
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = len(lb)
        self.func = func
        self.constraints = constraints if constraints else []
        self.method_used = method.upper()
        self.history = []
        self.best_y = np.inf

        print(f"[{self.name}] Running {self.method_used} with {len(self.constraints)} constraints...")

        if method == 'pso':
            self._pso(**kwargs)
        elif method == 'ga':
            self._ga(**kwargs)
        elif method == 'sa':
            self._sa(**kwargs)
        else:
            raise ValueError("Method must be 'pso', 'ga', or 'sa'.")

        print(f"[{self.name}] Best Value: {self.best_y:.6f}")
        self.is_fitted = True
        return self.best_x, self.best_y

    # ==========================================
    # 1. PSO (Updated to use _compute_fitness)
    # ==========================================
    def _pso(self, pop_size=50, iter_num=100, w=0.8, c1=0.5, c2=0.5):
        X = np.random.uniform(self.lb, self.ub, (pop_size, self.dim))
        V = np.random.uniform(-1, 1, (pop_size, self.dim))

        # Init P_best
        p_best = X.copy()
        # Use _compute_fitness instead of func
        p_best_y = np.array([self._compute_fitness(x) for x in X])

        g_best_idx = np.argmin(p_best_y)
        self.best_x = p_best[g_best_idx].copy()
        self.best_y = p_best_y[g_best_idx]

        for _ in range(iter_num):
            r1 = np.random.rand(pop_size, self.dim)
            r2 = np.random.rand(pop_size, self.dim)
            V = w * V + c1 * r1 * (p_best - X) + c2 * r2 * (self.best_x - X)
            X = X + V
            X = np.clip(X, self.lb, self.ub)

            # Evaluate
            current_y = np.array([self._compute_fitness(x) for x in X])

            better_mask = current_y < p_best_y
            p_best[better_mask] = X[better_mask]
            p_best_y[better_mask] = current_y[better_mask]

            min_idx = np.argmin(p_best_y)
            if p_best_y[min_idx] < self.best_y:
                self.best_y = p_best_y[min_idx]
                self.best_x = p_best[min_idx].copy()
            self.history.append(self.best_y)

    # ==========================================
    # 2. GA (Updated to use _compute_fitness)
    # ==========================================
    def _ga(self, pop_size=50, iter_num=100, mutation_rate=0.1):
        pop = np.random.uniform(self.lb, self.ub, (pop_size, self.dim))

        for _ in range(iter_num):
            fitness_vals = np.array([self._compute_fitness(x) for x in pop])

            min_idx = np.argmin(fitness_vals)
            if fitness_vals[min_idx] < self.best_y:
                self.best_y = fitness_vals[min_idx]
                self.best_x = pop[min_idx].copy()
            self.history.append(self.best_y)

            # Selection & Crossover (Simplified Tournament)
            new_pop = []
            for _ in range(pop_size):
                # Pick 3 for better tournament pressure
                idx = np.random.choice(pop_size, 3, replace=False)
                winner = idx[np.argmin(fitness_vals[idx])]
                new_pop.append(pop[winner])
            new_pop = np.array(new_pop)

            # Crossover
            np.random.shuffle(new_pop)
            for j in range(0, pop_size - 1, 2):
                if np.random.rand() < 0.8:
                    alpha = np.random.rand()
                    p1, p2 = new_pop[j], new_pop[j + 1]
                    new_pop[j] = alpha * p1 + (1 - alpha) * p2
                    new_pop[j + 1] = (1 - alpha) * p1 + alpha * p2

            # Mutation
            mask = np.random.rand(pop_size, self.dim) < mutation_rate
            noise = np.random.normal(0, 0.5 * (self.ub - self.lb), (pop_size, self.dim))
            new_pop[mask] += noise[mask]
            pop = np.clip(new_pop, self.lb, self.ub)

    # ==========================================
    # 3. SA (Updated to use _compute_fitness)
    # ==========================================
    def _sa(self, iter_num=1000, T_init=100, alpha=0.98):
        x_curr = np.random.uniform(self.lb, self.ub)
        y_curr = self._compute_fitness(x_curr)

        self.best_x = x_curr.copy()
        self.best_y = y_curr
        T = T_init

        for _ in range(iter_num):
            scale = (self.ub - self.lb) * (T / T_init) * 0.5
            x_new = np.clip(x_curr + np.random.normal(0, scale), self.lb, self.ub)
            y_new = self._compute_fitness(x_new)

            if y_new < y_curr:
                x_curr, y_curr = x_new, y_new
                if y_new < self.best_y:
                    self.best_y = y_new
                    self.best_x = x_new.copy()
            else:
                p = np.exp(-(y_new - y_curr) / (T + 1e-10))
                if np.random.rand() < p:
                    x_curr, y_curr = x_new, y_new

            T *= alpha
            self.history.append(self.best_y)

    # plot and others remain same...
    def plot(self):
        if not self.is_fitted: return
        plt.figure(figsize=(10, 5))
        plt.plot(self.history, 'b-', linewidth=2)
        plt.title(f"{self.method_used} Optimization Process")
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness (Obj + Penalty)")
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.show()

    def fit(self):
        pass

    def _predict_single(self):
        return 0