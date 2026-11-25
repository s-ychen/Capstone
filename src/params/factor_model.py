"""Factor form of the bond covariance, Q = F F' + diag(psi) (D3 eq. 3, Appendix A).

F = [K Gamma_p Lambda_p^{1/2}, B L]: p curve factors, then one credit factor per corporate sector. B_ig = -D_spr_i if
bond i is in credit group g (else 0), L L' is the Ledoit-Wolf covariance of the monthly credit factor returns.

Q is never formed for the solver: the tracking variance of active weights a is ||F'a||^2 + sum(psi a^2).
"""
from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf

from .covariance import estimate_sigma, pca


@dataclass
class FactorModel:
    Sigma: np.ndarray       # m x m key-rate covariance (monthly)
    Gamma: np.ndarray       # m x p
    Lambda: np.ndarray      # p
    explained: np.ndarray   # p variance shares
    psi: np.ndarray         # n residual variances
    n_obs: int              # observations behind Sigma (T_s)
    credit_groups: tuple = ()           # credit factor names (corporate sectors), empty if no credit factors
    credit_L: np.ndarray | None = None  # g x g Cholesky factor of the credit factor covariance
    credit_n_obs: int = 0               # months behind credit_L


def fit_factor_model(dY, psi: np.ndarray, method: str = "ledoit_wolf", p: int = 3, halflife: float = 12,
                     scale: float = 1.0, credit: np.ndarray | None = None, credit_groups=()) -> FactorModel:
    """Estimate Sigma from key-rate changes dY, keep p principal components, attach residual variances psi.

    credit (g x T, NaN allowed) are the credit factor returns from residual.credit_factors. Their covariance is
    Ledoit-Wolf over the months where every group has a value. Fewer than 3 such months leaves the credit factors out.
    """
    Sigma = estimate_sigma(dY, method, halflife, scale)
    pc = pca(Sigma, p)
    fm = FactorModel(Sigma, pc.Gamma, pc.Lambda, pc.explained, np.asarray(psi, float), len(dY))
    if credit is not None and len(credit_groups):
        X = credit[:, np.all(np.isfinite(credit), axis=0)].T
        if len(X) >= 3:
            cov = LedoitWolf().fit(X).covariance_
            fm.credit_groups = tuple(credit_groups)
            fm.credit_L = np.linalg.cholesky(cov + 1e-12 * np.eye(len(cov)))
            fm.credit_n_obs = len(X)
    return fm


def credit_exposures(D_spr: np.ndarray, group: np.ndarray, fm: FactorModel) -> np.ndarray:
    """B, n x g: B_ig = -D_spr_i if bond i is in credit group g, else 0."""
    B = np.zeros((len(D_spr), len(fm.credit_groups)))
    for j, g in enumerate(fm.credit_groups):
        m = group == g
        B[m, j] = -D_spr[m]
    return B


def loadings(K: np.ndarray, fm: FactorModel, D_spr: np.ndarray | None = None, group: np.ndarray | None = None
             ) -> np.ndarray:
    """F = [K Gamma_p Lambda_p^{1/2}, B L], n x (p + g). Without D_spr and group (or credit factors) only the curve part.
    Rows with NaN K stay NaN."""
    F = K @ fm.Gamma * np.sqrt(fm.Lambda)
    if fm.credit_L is None or D_spr is None or group is None:
        return F
    return np.hstack([F, credit_exposures(D_spr, group, fm) @ fm.credit_L])


def dense_Q(F: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Full n x n covariance. For tests and small universes only (13,000 bonds is about 180 million entries)."""
    return F @ F.T + np.diag(psi)


def tracking_variance(F: np.ndarray, psi: np.ndarray, a: np.ndarray) -> float:
    """a'Qa without forming Q."""
    return float(np.sum((F.T @ a) ** 2) + np.sum(psi * a ** 2))


def theta_factor(F: np.ndarray, psi: np.ndarray, T_s: int) -> tuple[np.ndarray, np.ndarray]:
    """Factor form of the estimation-error covariance Theta = Q / T_s (covariance of a sample mean).

    Assumption: D2 and D3 do not define Theta. ||Theta^{1/2} x||_2 = sqrt(||F_th' x||^2 + sum(psi_th x^2)).
    Returns (F / sqrt(T_s), psi / T_s).
    """
    return F / np.sqrt(T_s), psi / T_s
