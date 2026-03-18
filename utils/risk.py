import numpy as np
import pandas as pd

def rolling_cov_matrix(returns, window):
    """
    Compute rolling covariance matrices for returns.
    Returns:
        covs: (n_steps, n_assets, n_assets) array
    """
    n_steps, n_assets = returns.shape
    covs = np.zeros((n_steps, n_assets, n_assets))
    for t in range(window-1, n_steps):
        covs[t] = np.cov(returns[t-window+1:t+1].T, ddof=0)
    # Fill initial with first valid
    for t in range(window-1):
        covs[t] = covs[window-1]
    return covs

def compute_risk_contributions(weights, covs):
    """
    Compute portfolio volatility, marginal and total risk contributions.
    Returns:
        sigma_p: (n_steps,) array
        mcr: (n_steps, n_assets) array
        rc: (n_steps, n_assets) array
    """
    n_steps, n_assets = weights.shape
    sigma_p = np.zeros(n_steps)
    mcr = np.zeros((n_steps, n_assets))
    rc = np.zeros((n_steps, n_assets))
    for t in range(n_steps):
        w = weights[t]
        cov = covs[t]
        sigma = np.sqrt(w @ cov @ w)
        sigma_p[t] = sigma
        mcr_t = (cov @ w) / sigma if sigma > 0 else np.zeros_like(w)
        mcr[t] = mcr_t
        rc[t] = w * mcr_t
    return sigma_p, mcr, rc
