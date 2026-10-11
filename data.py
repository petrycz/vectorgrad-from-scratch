import numpy as np

def make_moons(n=100, noise=0.10, seed=0):
    rng = np.random.default_rng(seed)
    n_out = n // 2; n_in = n - n_out
    to = np.linspace(0, np.pi, n_out); outer = np.c_[np.cos(to), np.sin(to)]
    ti = np.linspace(0, np.pi, n_in);  inner = np.c_[1 - np.cos(ti), 1 - np.sin(ti) - 0.5]
    X = np.vstack([outer, inner]); y = np.array([0]*n_out + [1]*n_in)
    X = X + rng.normal(scale=noise, size=X.shape)
    X = (X - X.mean(0)) / X.std(0)
    return X, y
