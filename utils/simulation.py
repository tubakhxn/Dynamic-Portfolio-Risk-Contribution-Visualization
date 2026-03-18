import numpy as np

def simulate_correlated_gbm(n_assets, n_steps, dt, mu, base_cov, vol_cluster_alpha=0.97, seed=None):
    """
    Simulate correlated GBM returns with time-varying volatility (volatility clustering).
    Returns:
        returns: (n_steps, n_assets) array
        prices: (n_steps+1, n_assets) array
    """
    if seed is not None:
        np.random.seed(seed)
    n_assets = int(n_assets)
    n_steps = int(n_steps)
    # Cholesky for correlation
    L = np.linalg.cholesky(base_cov)
    # Initial volatilities
    vols = np.sqrt(np.diag(base_cov))
    # Simulate time-varying vol (GARCH-like)
    returns = np.zeros((n_steps, n_assets))
    prices = np.ones((n_steps+1, n_assets))
    current_vol = vols.copy()
    for t in range(n_steps):
        # Volatility clustering: slow mean reversion to base vol
        current_vol = np.sqrt(vol_cluster_alpha * current_vol**2 + (1-vol_cluster_alpha) * vols**2)
        z = np.random.randn(n_assets)
        correlated_z = L @ z
        ret = mu * dt + current_vol * np.sqrt(dt) * correlated_z
        returns[t] = ret
        prices[t+1] = prices[t] * np.exp(ret)
        # Update vol with squared return (GARCH flavor)
        current_vol = np.sqrt(vol_cluster_alpha * current_vol**2 + (1-vol_cluster_alpha) * (ret**2))
    return returns, prices
