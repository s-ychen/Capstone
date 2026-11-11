import cvxpy as cp
import numpy as np

class GoalPathBondOptimizer:
    """
    Optimizer class for the GoalPath-Bond multi-period tracking model.
    """
    def __init__(self, 
                 H, # number of periods
                 n, # number of assets
                 omega, # multi-period weights
                 c, # transaction cost
                 Q, # covariance matrices for each period
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
        self.Q = Q
        
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
        Formulates and solves the CVXPY problem using GUROBI based on instance properties.
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
            active_risk = self.omega[h] * cp.quad_form(a[h], self.Q[h])
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
        self.prob.solve(solver=cp.GUROBI, verbose=verbose)
        
        if self.prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Solver did not find an optimal solution. Status: {self.prob.status}")
            
        # 5. Extract and store results
        self.x_val = [x[h].value for h in range(self.H)]
        self.u_plus_val = [u_plus[h].value for h in range(self.H)]
        self.u_minus_val = [u_minus[h].value for h in range(self.H)]
        self.z_val = [z[h].value for h in range(self.H)]
        
        return self.prob.value, self.x_val, self.u_plus_val, self.u_minus_val, self.z_val

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