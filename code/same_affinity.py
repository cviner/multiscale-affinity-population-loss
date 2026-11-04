"""Same-affinity control F_A: exact t-SNE on the calibrated affinities that P receives.

The channels of set A are pooled with equal weight into one normalized matrix and fitted with an
exact five-dimensional Cauchy objective and the multiscale t-SNE schedule.
"""

import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from openTSNE import affinity

from exact_tsne import optimize
from rna import CHANNEL_SETS

CHANNELS = {f'F_{name}': channels for name, channels in CHANNEL_SETS.items()}


def probabilities(z, h2, log_norm, method):
    """The channels of the set, pooled with equal weight and normalized to sum one."""
    channels = CHANNELS[method]
    weight = 1.0 / len(channels)
    d2 = squareform(pdist(z, 'sqeuclidean'))
    pool = np.zeros_like(d2)
    for j in channels:
        g = np.exp(-d2 / (2 * h2[j]) - log_norm[j])
        np.fill_diagonal(g, 0.0)
        pool += weight * g
    return sparse.csr_matrix(pool / pool.sum())


def fit(z, h2, log_norm, method, seed):
    """Exact five-dimensional Cauchy embedding of the pooled affinities; returns (map, record)."""
    p = affinity.PrecomputedAffinities(probabilities(z, h2, log_norm, method), normalize=False)
    e = optimize(z, p, seed)
    return np.array(e), {'kl_divergence': float(e.kl_divergence)}
