"""Key-rate covariance and its principal components (Deliverable 3, Section 2.1 steps 1 and 3)."""
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from data_cleaning.treasury_curve import KEY_COLS

TRADING_DAYS_PER_MONTH = 21
METHODS = ("sample", "ewma", "ledoit_wolf")


def key_rate_changes(curve: pd.DataFrame, end=None, freq: str = "M", window: int | None = None) -> pd.DataFrame:
    """One-period changes of the 8 key-rate zero yields (decimal). freq "M" uses month-end curves, "D" daily curves.

    end drops later dates (no look-ahead); window keeps only the last `window` changes. With freq="D" pass
    scale=TRADING_DAYS_PER_MONTH to estimate_sigma to get a monthly covariance.
    """
    y = curve.set_index("date")[KEY_COLS].sort_index()
    if end is not None:
        y = y[y.index <= pd.Timestamp(end)]
    if freq == "M":
        y = y.groupby([y.index.year, y.index.month]).tail(1)
    elif freq != "D":
        raise ValueError("freq must be 'M' or 'D'")
    dy = y.diff().dropna()
    return dy.tail(window) if window else dy


def estimate_sigma(dY, method: str = "ledoit_wolf", halflife: float = 12, scale: float = 1.0) -> np.ndarray:
    """m x m covariance of key-rate changes (rows are periods, oldest first).

    sample: np.cov (ddof 1). ewma: exponentially weighted with the given halflife in periods, newest weighted most,
    mean removed. ledoit_wolf: sklearn Ledoit-Wolf shrinkage toward a scaled identity. scale multiplies the result
    (e.g. 21 to turn a daily covariance into a monthly one).
    """
    X = np.asarray(dY, float)
    if method == "sample":
        S = np.cov(X, rowvar=False)
    elif method == "ewma":
        w = 0.5 ** (np.arange(len(X))[::-1] / halflife)
        w /= w.sum()
        Xc = X - w @ X
        S = (Xc * w[:, None]).T @ Xc / (1 - (w ** 2).sum())
    elif method == "ledoit_wolf":
        S = LedoitWolf().fit(X).covariance_
    else:
        raise ValueError(f"method must be one of {METHODS}")
    return (S + S.T) / 2 * scale


class PCA(NamedTuple):
    Gamma: np.ndarray       # m x p eigenvectors (columns), each signed so its largest-magnitude loading is positive
    Lambda: np.ndarray      # p eigenvalues, descending
    explained: np.ndarray   # p variance shares of the total trace


def pca(Sigma: np.ndarray, p: int = 3) -> PCA:
    """Top-p eigendecomposition of Sigma (D2 Appendix B.1): level, slope, curvature for p=3."""
    vals, vecs = np.linalg.eigh(Sigma)
    order = np.argsort(vals)[::-1][:p]
    G, L = vecs[:, order], np.clip(vals[order], 0, None)
    G = G * np.sign(G[np.abs(G).argmax(axis=0), np.arange(p)])
    return PCA(G, L, L / np.trace(Sigma))
