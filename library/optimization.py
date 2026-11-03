'''
Optimization strategies:
1. Min difference in weights
2. Min difference returns - covariance matrix of historical bond returns 
3. Min ex-ante tracking error - covariance matrix of historical factor returns

Constraints: 
1. Expected return 
2. KRD matching
3. Roll-down matching
4. Multi-duration matching 

Multi-period matching: 
1. Min total tracking error of same weights over multiple periods 
2. Min total factor exposure of same weights at each period 
3. Scenario-based - credit spreads, curve shifts with associated probabilities 
'''
import cvxpy as cp 
import utils

class Optimizer: 
    def __init__(self):
        pass

    def min_weights(self, index_weights, expected_return, historical_returns, covariance):
        x = cp.Variable(len(index_weights))
        krd = utils.KRD(historical_returns)
        constraints = [cp.sum(x) == 1, # full portfolio 
                       x >= 0, # no shorting 
                       abs(historical_returns @ x - expected_return) < 0.01, # return matching
                                  
                       ]
        opt_eqn = cp.quad_form(x-index_weights, covariance)
        prob = cp.Problem(cp.Minimize(opt_eqn), self.constraints)