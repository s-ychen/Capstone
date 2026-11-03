import numpy as np
import pandas as pd


def consecutive_differences(historical_yields):
    """
    Return the change in each maturity between consecutive dates.

    The input is expected to be a 2D array with rows as dates and columns as
    maturities. The output has one fewer row than the input and contains the
    element-wise difference across time for each maturity column.
    """
    yields_array = np.asarray(historical_yields, dtype=float)

    if yields_array.ndim != 2:
        raise ValueError("historical_yields must be a 2D array-like object")

    if yields_array.shape[0] < 2:
        return np.empty((0, yields_array.shape[1]), dtype=float)

    return np.diff(yields_array, axis=0)


def yield_covariance(historical_yields):
    """
    Returns the covariance matrix of the yield curve changes.

    Each maturity column is differenced across consecutive dates before
    computing the covariance matrix.

    :param historical_yields: date x maturities matrix
    :return: covariance matrix of consecutive yield changes
    """
    historical_yields = np.asarray(historical_yields, dtype=float)
    
    if historical_yields.shape[0] < 2:
        return np.zeros((historical_yields.shape[1], historical_yields.shape[1]), dtype=float)

    deltas = consecutive_differences(historical_yields)
    covariance = np.cov(deltas, rowvar=False)

    if np.isscalar(covariance):
        return np.array([[float(covariance)]], dtype=float)

    return covariance