"""Model P on the RNA affinity channels.

P_A is the joint fit to channel set A, with the msld objective divided by S n (n - 1). Two starts,
the input's leading principal components with a small perturbation and a Gaussian configuration,
share one seeded generator, and the start with the lower objective is kept.
"""

import time

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.special import logsumexp

import msld
from rna import CHANNEL_SETS, kernels

P_CHANNELS = {f'P_{name}': channels for name, channels in CHANNEL_SETS.items()}
OPTIONS = {'maxiter': 10000, 'maxfun': 30000, 'ftol': 1e-10, 'gtol': 1e-7}


def fit_p(z, gs, widths, seed, dimension=5):
    """Fit P to the affinity channels gs from both starts; returns (map, record, parameters of both starts)."""
    n = len(z)
    off = ~np.eye(n, dtype=bool)
    scale = len(gs) * n * (n - 1)
    rng = np.random.default_rng(seed)
    starts = [z[:, :dimension] + rng.normal(size=(n, dimension)) * 1e-3, rng.normal(size=(n, dimension))]
    objective, units = msld.scaled_objective(gs, widths, n, dimension=dimension, normalizer=scale, off=off)
    records, solutions = [], []
    rms = np.sqrt(np.mean(pdist(z, 'sqeuclidean')))
    for start_name, x0 in zip(['pc_perturbed', 'gaussian'], starts):
        x0 -= x0.mean(0)
        x0 *= rms / np.sqrt(np.mean(pdist(x0, 'sqeuclidean')))
        d2 = squareform(pdist(x0, 'sqeuclidean'))
        # Each intercept matches the channel's mean predicted affinity to its observed mean.
        b0 = np.array(
            [
                np.log(g[off].mean()) - logsumexp(-d2[off] / (2 * s)) + np.log(off.sum())
                for g, s in zip(gs, widths)
            ]
        )
        theta0 = msld.pack(x0, np.zeros(n), b0)
        tic = time.perf_counter()
        result = minimize(objective, theta0 / units, jac=True, method='L-BFGS-B', options=OPTIONS)
        elapsed = time.perf_counter() - tic
        x, a, b = msld.gauge(*msld.unpack(result.x * units, n, len(gs), dimension))
        f, g = objective(msld.pack(x, a, b) / units)
        records.append(
            {
                'start': start_name,
                'objective': float(f),
                'message': str(result.message),
                'iterations': int(result.nit),
                'gradient_inf_mean_loss': float(np.max(np.abs(g))),
                'seconds': elapsed,
            }
        )
        solutions.append((x, a, b))
    chosen = int(np.argmin([r['objective'] for r in records]))
    coordinates, effects, intercepts = zip(*solutions)
    parameters = dict(
        start_coordinates=np.stack(coordinates),
        start_node_effects=np.stack(effects),
        start_intercepts=np.stack(intercepts),
    )
    return coordinates[chosen], {'chosen_start': chosen, 'starts': records}, parameters


def embed(z, h2, log_norm, method, seed, dimension=5):
    """Fit the P configuration named by method to the features z; returns (map, record, start parameters)."""
    channels = P_CHANNELS[method]
    gs = kernels(z, h2, log_norm)
    x, record, parameters = fit_p(z, [gs[j] for j in channels], h2[channels], seed, dimension)
    record['channels'] = channels
    return x, record, parameters
