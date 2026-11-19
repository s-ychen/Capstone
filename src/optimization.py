import cvxpy as cp
import numpy as np

class GoalPathBondOptimizer:
    """
    Optimizer class for the GoalPath-Bond multi-period tracking model.
    """
    def __init__(self, 
                 H, # number of periods
                 n, # number of assets
                 omega, # weight of each period
                 c, # transaction cost
                 F, # covariance factors for each period
                 psi, # residual variances
                 b, # benchmark weights
                 R, # return matrices for each period
                 r_b, # benchmark returns
                 x_init, # initial portfolio weights
                 D, # factor loadings for each period
                 K, # risk factor exposures
                 D_spr, # sparse factor loadings
                 C, # cardinality constraints
                 delta_D, # max error limits for D
                 delta_K, # max exposure limits for K
                 delta_S, # max exposure limits for D_spr
                 delta_C, 
                 mu, 
                 eps2, 
                 Theta_half, 
                 r_req, 
                 l, 
                 E, 
                 k_max):
        
        # Dimensions
        self.H = H
        self.n = n
        
        # Objective parameters
        self.omega = omega
        self.c = c
        self.F = F
        self.psi = psi
        
        # Benchmark and drift parameters
        self.b = b
        self.R = R
        self.r_b = r_b
        self.x_init = x_init
        
        # Risk factor exposures and bounds
        self.D = D
        self.K = K
        self.D_spr = D_spr
        self.C = C
        self.delta_D = delta_D
        self.delta_K = delta_K
        self.delta_S = delta_S
        self.delta_C = delta_C
        
        # Robust return parameters
        self.mu = mu
        self.eps2 = eps2
        self.Theta_half = Theta_half
        self.r_req = r_req
        
        # Portfolio and cardinality constraints
        self.l = l
        self.E = E
        self.k_max = k_max
        
        # Placeholders for optimization state and results
        self.prob = None
        self.x_val = None
        self.u_plus_val = None
        self.u_minus_val = None
        self.z_val = None

    def optimize(self, verbose=True):
        """
        Formulates and solves the mixed-integer CVXPY problem using GUROBI.
        """
        # 1. Decision Variables
        x = [cp.Variable(self.n, nonneg=True) for _ in range(self.H)]
        z = [cp.Variable(self.n, boolean=True) for _ in range(self.H)]
        u_plus = [cp.Variable(self.n, nonneg=True) for _ in range(self.H)]
        u_minus = [cp.Variable(self.n, nonneg=True) for _ in range(self.H)]
        
        # Intermediate variables
        a = [cp.Variable(self.n) for _ in range(self.H)]       # active weights
        x_tilde = [cp.Variable(self.n) for _ in range(self.H)] # drifted weights
        
        # 2. Objective Function
        objective_terms = []
        for h in range(self.H):
            active_risk = self.omega[h] * (
                cp.sum_squares(self.F[h].T @ a[h])
                + cp.sum(cp.multiply(self.psi, cp.square(a[h])))
            )
            transaction_costs = self.c.T @ (u_plus[h] + u_minus[h])
            objective_terms.append(active_risk + transaction_costs)
            
        objective = cp.Minimize(cp.sum(objective_terms))
        
        # 3. Constraints
        constraints = []
        
        for h in range(self.H):
            # Active weights definition
            constraints.append(a[h] == x[h] - self.b[h])
            
            # Drift constraints
            if h == 0:
                constraints.append(x_tilde[h] == (self.R[h] @ self.x_init) / (1 + self.r_b[h]))
            else:
                constraints.append(x_tilde[h] == (self.R[h] @ x[h-1]) / (1 + self.r_b[h]))
                
            # Trade balance
            constraints.append(x[h] - x_tilde[h] == u_plus[h] - u_minus[h])
            
            # Factor exposure limits (Absolute value constraints)
            constraints.append(cp.abs(self.D[h].T @ a[h]) <= self.delta_D)
            constraints.append(cp.abs(self.K[h].T @ a[h]) <= self.delta_K)
            constraints.append(cp.abs(self.D_spr[h].T @ a[h]) <= self.delta_S)
            constraints.append(cp.abs(self.C[h].T @ a[h]) <= self.delta_C)
            
            # Robust return goal (Second-Order Cone constraint)
            robust_penalty = self.eps2 * cp.norm(self.Theta_half[h] @ x[h], 2)
            constraints.append(self.mu[h].T @ x[h] - robust_penalty >= self.r_req[h])
            
            # Minimum lot and indicator constraints
            constraints.append(cp.multiply(self.l[h], z[h]) <= x[h])
            constraints.append(x[h] <= z[h])
            
            # Eligibility constraint
            if len(self.E[h]) > 0:
                for i in self.E[h]:
                    constraints.append(z[h][i] == 0)
                    
            # Cardinality constraint
            constraints.append(cp.sum(z[h]) <= self.k_max)
            
            # Budget and no-shorting constraints
            constraints.append(cp.sum(x[h]) == 1)

        # 4. Formulate and Solve Problem
        self.prob = cp.Problem(objective, constraints)
        if cp.GUROBI not in cp.installed_solvers():
            raise RuntimeError(
                "GUROBI is required to solve this mixed-integer problem. "
                "Install gurobipy and configure a valid Gurobi license."
            )
        self.prob.solve(solver=cp.GUROBI, verbose=verbose)
        
        if self.prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Solver did not find an optimal solution. Status: {self.prob.status}")
            
        # 5. Extract and store results
        self.x_val = [x[h].value for h in range(self.H)]
        self.u_plus_val = [u_plus[h].value for h in range(self.H)]
        self.u_minus_val = [u_minus[h].value for h in range(self.H)]
        self.z_val = [z[h].value for h in range(self.H)]
        
        return self.prob.value, self.x_val, self.u_plus_val, self.u_minus_val, self.z_val

    def optimize_sequential(self, verbose=True):
        """Solve periods in sequence, using each solution as the next period's x_init.

        Returns per-period objective values, weights, buys, sells, and selection flags.
        The optimizer's x_init is updated to the final period's solution.
        """
        initial_weights = np.asarray(self.x_init, dtype=float).copy()
        if initial_weights.shape != (self.n,):
            raise ValueError(
                f"x_init must have shape ({self.n},), got {initial_weights.shape}"
            )

        objective_values = []
        x_values = []
        u_plus_values = []
        u_minus_values = []
        z_values = []

        for h in range(self.H):
            period_optimizer = GoalPathBondOptimizer(
                H=1,
                n=self.n,
                omega=[self.omega[h]],
                c=self.c,
                F=[self.F[h]],
                psi=self.psi,
                b=[self.b[h]],
                R=[self.R[h]],
                r_b=[self.r_b[h]],
                x_init=initial_weights,
                D=[self.D[h]],
                K=[self.K[h]],
                D_spr=[self.D_spr[h]],
                C=[self.C[h]],
                delta_D=self.delta_D,
                delta_K=self.delta_K,
                delta_S=self.delta_S,
                delta_C=self.delta_C,
                mu=[self.mu[h]],
                eps2=self.eps2,
                Theta_half=[self.Theta_half[h]],
                r_req=[self.r_req[h]],
                l=[self.l[h]],
                E=[self.E[h]],
                k_max=self.k_max,
            )
            value, period_x, period_u_plus, period_u_minus, period_z = (
                period_optimizer.optimize(verbose=verbose)
            )

            if not period_x or period_x[0] is None:
                raise RuntimeError(f"Optimizer returned no weights for period {h}.")

            initial_weights = np.asarray(period_x[0], dtype=float).copy()
            objective_values.append(value)
            x_values.append(initial_weights)
            u_plus_values.append(period_u_plus[0])
            u_minus_values.append(period_u_minus[0])
            z_values.append(period_z[0])

            self.x_init = initial_weights.copy()
            self.prob = period_optimizer.prob
            self.x_val = x_values
            self.u_plus_val = u_plus_values
            self.u_minus_val = u_minus_values
            self.z_val = z_values

        return objective_values, x_values, u_plus_values, u_minus_values, z_values

    def plot_wealth(self, initial_wealth, ax=None, show=True, include_transaction_costs=True):
        """Plot expected portfolio wealth using optimized weights and forecast returns.

        If enabled, transaction costs are deducted at each rebalance before that
        period's forecast return is applied. Returns wealth at periods 0 through H.
        """
        if self.x_val is None or self.u_plus_val is None or self.u_minus_val is None:
            raise RuntimeError("Run optimize() or optimize_sequential() before plotting wealth.")
        if len(self.x_val) != self.H:
            raise ValueError(f"Expected {self.H} optimized holdings, got {len(self.x_val)}.")
        if not np.isfinite(initial_wealth) or initial_wealth <= 0:
            raise ValueError("initial_wealth must be a positive finite number.")

        wealth = np.empty(self.H + 1, dtype=float)
        wealth[0] = initial_wealth
        for h in range(self.H):
            portfolio_return = float(np.asarray(self.mu[h]) @ np.asarray(self.x_val[h]))
            transaction_cost = 0.0
            if include_transaction_costs:
                trades = np.asarray(self.u_plus_val[h]) + np.asarray(self.u_minus_val[h])
                transaction_cost = float(np.asarray(self.c) @ trades)
                if not np.isfinite(transaction_cost) or transaction_cost >= 1:
                    raise ValueError(
                        f"Transaction cost for period {h} must be finite and less than 100%."
                    )

            gross_return = 1 + portfolio_return
            if not np.isfinite(gross_return) or gross_return < 0:
                raise ValueError(f"Forecast return for period {h} implies negative wealth.")
            wealth[h + 1] = wealth[h] * (1 - transaction_cost) * gross_return

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(np.arange(self.H + 1), wealth, marker="o")
        ax.set(
            xlabel="Rebalance period",
            ylabel="Portfolio wealth",
            title="Expected portfolio wealth",
        )
        ax.grid(True, alpha=0.3)
        if show:
            plt.show()
        return wealth

    def calculate_bond_risk_metrics(df, risk_free_rate=0.0, face_value=100.0):
        """
        Calculates duration, spread duration, and convexity for a dataframe of bonds.
        
        Assumes annual compounding and integer years to maturity.
        
        Args:
            df (pd.DataFrame): Dataframe containing 'bond', 'maturity' (in years), 
                            'coupon' (decimal), and 'credit_spread' (decimal).
            risk_free_rate (float): Base rate to add to the credit spread to get YTM.
            face_value (float): The par value of the bonds.
            
        Returns:
            pd.DataFrame: A copy of the original dataframe with price, duration, and convexity metrics.
        """
        results = df.copy()
        
        prices = []
        mac_durations = []
        mod_durations = []
        convexities = []
        
        for _, row in results.iterrows():
            maturity = int(np.ceil(row['maturity']))
            coupon_rate = row['coupon']
            ytm = risk_free_rate + row['credit_spread']
            
            if maturity <= 0:
                prices.append(face_value)
                mac_durations.append(0.0)
                mod_durations.append(0.0)
                convexities.append(0.0)
                continue
                
            # 1. Generate Cash Flows
            t_periods = np.arange(1, maturity + 1)
            cash_flows = np.full(maturity, coupon_rate * face_value)
            cash_flows[-1] += face_value
            
            # 2. Calculate Price
            discount_factors = (1 + ytm) ** t_periods
            pv_cash_flows = cash_flows / discount_factors
            price = np.sum(pv_cash_flows)
            
            # 3. Calculate Macaulay Duration
            # Formula: (1 / Price) * sum [ t * CF_t / (1+y)^t ]
            mac_dur = np.sum(t_periods * pv_cash_flows) / price
            
            # 4. Calculate Modified Duration (and Spread Duration)
            # Formula: Macaulay Duration / (1 + ytm)
            mod_dur = mac_dur / (1 + ytm)
            
            # 5. Calculate Convexity
            # Formula: (1 / Price) * sum [ (CF_t * t * (t+1)) / (1+y)^(t+2) ]
            convexity_terms = (cash_flows * t_periods * (t_periods + 1)) / ((1 + ytm) ** (t_periods + 2))
            convexity = np.sum(convexity_terms) / price
            
            prices.append(price)
            mac_durations.append(mac_dur)
            mod_durations.append(mod_dur)
            convexities.append(convexity)
            
        results['price'] = prices
        results['macaulay_duration'] = mac_durations
        results['modified_duration'] = mod_durations
        results['convexity'] = convexities
        
        return results