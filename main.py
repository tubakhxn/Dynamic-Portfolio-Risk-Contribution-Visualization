import numpy as np
import pandas as pd
from utils.simulation import simulate_correlated_gbm
from utils.portfolio import generate_dynamic_weights
from utils.risk import rolling_cov_matrix, compute_risk_contributions
from utils.visualization import animate_stacked_risk_contributions

# --- PARAMETERS ---
N_ASSETS = 6
N_STEPS = 600
DT = 1/252
WINDOW = 60
SEED = 42
ASSET_NAMES = [f"Asset {chr(65+i)}" for i in range(N_ASSETS)]

# --- SIMULATE CORRELATED RETURNS ---
np.random.seed(SEED)
mu = np.random.uniform(0.03, 0.12, N_ASSETS)
# Random positive-definite covariance
A = np.random.randn(N_ASSETS, N_ASSETS)
base_cov = np.dot(A, A.T)
base_cov /= np.max(np.abs(base_cov))
base_cov *= np.random.uniform(0.08, 0.18, (N_ASSETS, 1))

returns, prices = simulate_correlated_gbm(
    n_assets=N_ASSETS,
    n_steps=N_STEPS,
    dt=DT,
    mu=mu,
    base_cov=base_cov,
    vol_cluster_alpha=0.97,
    seed=SEED
)

# --- DYNAMIC PORTFOLIO WEIGHTS ---
weights = generate_dynamic_weights(N_ASSETS, N_STEPS, smooth=0.985, seed=SEED)

# --- ROLLING COVARIANCE ---
covs = rolling_cov_matrix(returns, window=WINDOW)

# --- RISK CONTRIBUTIONS ---
sigma_p, mcr, rc = compute_risk_contributions(weights, covs)

# --- ANIMATE STACKED RISK CONTRIBUTIONS ---
if __name__ == "__main__":
    animate_stacked_risk_contributions(
        rc=rc,
        sigma_p=sigma_p,
        asset_names=ASSET_NAMES,
        window=180,  # longer trailing window for smoother look
        fps=60,      # higher FPS for smoother animation
        highlight=True
    )
