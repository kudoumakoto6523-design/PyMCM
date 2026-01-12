import numpy as np
import matplotlib.pyplot as plt
import sympy
from scipy.integrate import solve_ivp
from pymcm.core.base import BaseModel


class SmartODESolver(BaseModel):
    """
    Smart ODE Solver (Robust Edition).

    Logic:
    1. Parse user equation.
    2. TRY to solve Analytically (Formula).
    3. IF anything fails (Complex roots, No formula, SymPy errors) -> CATCH & SWITCH.
    4. Solve Numerically (RK45) by reducing N-th order to system of 1st order ODEs.
    """

    def __init__(self):
        super().__init__(name="Smart ODE Solver")
        self.solution_type = None
        self.formula = None
        self.t_eval = None
        self.y_eval = None

    def solve(self, eq_str, ics_list, t_span, t_eval_points=100):
        """
        Args:
            eq_str: "diff(y, t, 8) + 2*diff(y, t) + 5*y = 0"
            ics_list: Initial conditions [y(0), y'(0), ..., y^(n-1)(0)]
            t_span: (0, 10)
        """
        # 1. Define Basic Symbols
        t = sympy.symbols('t')
        y = sympy.Function('y')(t)

        # 2. Parse Equation String
        if "=" in eq_str:
            lhs_str, rhs_str = eq_str.split("=")
        else:
            lhs_str, rhs_str = eq_str, "0"

        try:
            # Parse context
            context = {'y': y, 't': t, 'diff': sympy.diff, 'sin': sympy.sin,
                       'cos': sympy.cos, 'exp': sympy.exp, 'log': sympy.log}
            lhs = sympy.sympify(lhs_str, locals=context)
            rhs = sympy.sympify(rhs_str, locals=context)
            ode_eq = sympy.Eq(lhs, rhs)
        except Exception as e:
            raise ValueError(f"Equation parsing failed: {e}")

        # 3. Detect Order Manually (Replaces ode_order to avoid ImportError)
        # Find all derivatives in the equation
        derivs = ode_eq.atoms(sympy.Derivative)
        order = 0
        for d in derivs:
            # Check if this derivative is related to y
            if d.has(y):
                # derivative_count gets the order (e.g., diff(y, t, 2) -> 2)
                current_order = d.derivative_count
                if current_order > order:
                    order = current_order

        # If no derivatives found, it's order 0 (algebraic), but let's assume 1 for safety or error
        if order == 0: order = 1

        # Check ICs length
        if len(ics_list) != order:
            print(f"⚠️ Warning: Equation is Order-{order}, but {len(ics_list)} ICs provided.")
            # Adjust ICs list to match order for numerical solver safety
            if len(ics_list) > order:
                ics_list = ics_list[:order]
            elif len(ics_list) < order:
                # Pad with zeros if missing
                ics_list = ics_list + [0] * (order - len(ics_list))
                print(f"   -> Auto-padded ICs with zeros: {ics_list}")

        print(f"[{self.name}] Equation: {ode_eq} (Order: {order})")

        # 4. Construct SymPy ICS Dictionary
        t0 = t_span[0]
        ics_sympy = {}
        for i in range(len(ics_list)):
            if i == 0:
                ics_sympy[y.subs(t, t0)] = ics_list[i]
            else:
                ics_sympy[y.diff(t, i).subs(t, t0)] = ics_list[i]

        # --- PHASE 1: Attempt Analytical Solution (Try/Except Block) ---
        try:
            print(f"[{self.name}] Attempting Analytical Solution...")

            # Try to solve
            sol = sympy.dsolve(ode_eq, y, ics=ics_sympy)

            # If successful, store and return
            self.formula = sol
            self.solution_type = "Analytical"
            print(f"✅ Formula Found: {self.formula.rhs}")

            # Generate Data for Plotting
            func = sympy.lambdify(t, self.formula.rhs, modules='numpy')
            self.t_eval = np.linspace(t_span[0], t_span[1], t_eval_points)
            self.y_eval = func(self.t_eval)
            self.is_fitted = True
            return self

        except Exception as e:
            # Here we catch EVERYTHING: ImportError, NotImplementedError, NotAlgebraic, etc.
            print(f"⚠️ Analytical solve failed (Reason: {str(e)[:100]}...)")
            print(f"[{self.name}] -> Switching to Numerical Solution (Automatic Fallback)...")

        # --- PHASE 2: Fallback to Numerical (Reduction of Order) ---
        self.solution_type = "Numerical"

        # A. Isolate highest derivative: y^(n) = G(...)
        highest_deriv = y.diff(t, order)
        try:
            # Solve for y^(n) symbolically
            sol_highest = sympy.solve(ode_eq, highest_deriv)
            if not sol_highest:
                raise ValueError("Could not isolate highest derivative.")
            g_expr = sol_highest[0]
        except Exception as e:
            raise ValueError(f"Numerical fallback failed. Cannot isolate highest derivative: {e}")

        # B. Create the System Function for solve_ivp
        # We need to turn g_expr into a python function: g(t, y, y', ...)
        sym_args = [t] + [y.diff(t, i) for i in range(order)]

        try:
            g_func = sympy.lambdify(sym_args, g_expr, modules='numpy')
        except Exception as e:
            raise ValueError(f"Failed to compile equation to Python function: {e}")

        def ode_system(t, current_Y):
            # current_Y is [y, y', ..., y^(n-1)]
            dYdt = []
            # y' = y'
            # y'' = y'' ...
            for i in range(order - 1):
                dYdt.append(current_Y[i + 1])

            # Highest derivative
            # We must pass arguments exactly as defined in sym_args: t, y, y', ...
            last_deriv_val = g_func(t, *current_Y)
            dYdt.append(last_deriv_val)
            return dYdt

        # C. Solve using SciPy
        t_eval_arr = np.linspace(t_span[0], t_span[1], t_eval_points)

        print(f"[{self.name}] Running RK45 Numerical Solver...")
        res = solve_ivp(
            fun=ode_system,
            t_span=t_span,
            y0=ics_list,
            t_eval=t_eval_arr,
            method='RK45'  # Robust solver
        )

        if not res.success:
            print(f"❌ Numerical solver failed: {res.message}")
        else:
            self.t_eval = res.t
            self.y_eval = res.y[0]  # We only care about y (index 0)
            self.is_fitted = True
            print(f"✅ Numerical Solution Completed.")

        return self

    def plot(self):
        if not self.is_fitted: return
        plt.figure(figsize=(10, 6))

        title = f"ODE Solution ({self.solution_type})"
        if self.solution_type == "Analytical":
            try:
                title += f"\n${sympy.latex(self.formula.rhs)}$"
            except:
                pass

        plt.title(title)
        plt.plot(self.t_eval, self.y_eval, 'r-', linewidth=2, label='y(t)')
        plt.xlabel("t")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # Placeholders
    def fit(self):
        pass

    def _predict_single(self):
        return 0