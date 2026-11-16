"""Factor form of the bond covariance, Q = F F' + diag(psi) with F = K Gamma_p Lambda_p^{1/2} (D3 eq. 3, Appendix A).

Q is never formed for the solver: the tracking variance of active weights a is ||F'a||^2 + sum(psi a^2).
"""
from dataclasses import dataclass

import numpy as np

from .covariance import estimate_sigma, pca


@dataclass
class FactorModel:
    Sigma: np.ndarray       # m x m key-rate covariance (monthly)
    Gamma: np.ndarray       # m x p
    Lambda: np.ndarray      # p
    explained: np.ndarray   # p variance shares
    psi: np.ndarray         # n residual variances
    n_obs: int              # observations behind Sigma (T_s)


def fit_factor_model(dY, psi: np.ndarray, method: str = "ledoit_wolf", p: int = 3, halflife: float = 12,
                     scale: float = 1.0) -> FactorModel:
    """Estimate Sigma from key-rate changes dY, keep p principal components, attach residual variances psi."""
    Sigma = estimate_sigma(dY, method, halflife, scale)
    pc = pca(Sigma, p)
    return FactorModel(Sigma, pc.Gamma, pc.Lambda, pc.explained, np.asarray(psi, float), len(dY))


def loadings(K: np.ndarray, fm: FactorModel) -> np.ndarray:
    """F = K Gamma_p Lambda_p^{1/2}, n x p. Rows with NaN K stay NaN."""
    return K @ fm.Gamma * np.sqrt(fm.Lambda)


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
