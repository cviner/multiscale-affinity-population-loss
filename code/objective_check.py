"""A second implementation of the objective and gradient of P, separate from msld, for checking saved fits."""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def recompute(data, x, a, b, channels):
    """The objective and the largest gradient component at saved parameters, summing over ordered pairs
    and writing the coordinate gradient in Laplacian form."""
    n = len(x)
    off = ~np.eye(n, dtype=bool)
    input_d2 = squareform(pdist(data['features'], 'sqeuclidean'))
    map_d2 = squareform(pdist(x, 'sqeuclidean'))
    f = 0.0
    ga = np.zeros(n)
    gb = []
    coef = np.zeros((n, n))
    for j, channel in enumerate(channels):
        s = data['h2'][channel]
        g = np.exp(-input_d2 / (2 * s) - data['log_norm'][channel]) * off
        eta = b[j] + a[:, None] + a[None, :] - map_d2 / (2 * s)
        exp_eta = np.exp(np.minimum(eta, 700.0))
        mean_term = exp_eta * (1.0 + np.maximum(eta - 700.0, 0.0))
        f += np.sum((mean_term - g * eta)[off])
        resid = (g - exp_eta) * off
        ga -= 2 * resid.sum(axis=1)
        gb.append(-resid.sum())
        coef += resid / s
    gx = 2 * (coef.sum(axis=1)[:, None] * x - coef @ x)
    scale = len(channels) * n * (n - 1)
    return float(f / scale), float(np.max(np.abs(np.r_[gx.ravel(), ga, gb] / scale)))
