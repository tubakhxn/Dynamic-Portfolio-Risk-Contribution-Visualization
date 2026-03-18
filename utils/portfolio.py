import numpy as np

def generate_dynamic_weights(n_assets, n_steps, smooth=0.98, seed=None):
    """
    Generate dynamic portfolio weights that evolve smoothly over time.
    Returns:
        weights: (n_steps, n_assets) array, rows sum to 1
    """
    if seed is not None:
        np.random.seed(seed+100)
    weights = np.zeros((n_steps, n_assets))
    w = np.random.dirichlet(np.ones(n_assets))
    for t in range(n_steps):
        # Small random walk in weight space
        noise = 0.01 * np.random.randn(n_assets)
        w = smooth * w + (1-smooth) * np.random.dirichlet(np.ones(n_assets)) + noise
        w = np.clip(w, 0, None)
        w = w / w.sum()
        weights[t] = w
    return weights
