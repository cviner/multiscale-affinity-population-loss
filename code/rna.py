"""RNA representation and Gaussian affinity channels.

Counts are normalized to 10,000 per cell and log1p-transformed. On the training cells, the
2,000 highest-variance genes (detected in at least three cells, ties broken by gene order) are
standardized and reduced to 30 unwhitened principal components, whose scores are divided by
their root-mean-square pairwise distance. Each of four affinity widths is set so that the median
perplexity of the row-normalized affinities equals a fixed fraction of the n - 1 other cells.

The refitted design learns everything from the cells of each arm:
    features = transform(logx, train_representation(logx, solver))
The fixed design learns the representation on calibration set A, applies it to set B and to the
evaluation cells, learns the widths on set B, and keeps all of it across arms.
"""

import numpy as np
from scipy import sparse
from scipy.optimize import brentq
from scipy.spatial.distance import pdist, squareform
from scipy.special import logsumexp
from sklearn.decomposition import PCA

FRACTIONS = np.array([0.025, 0.1, 0.4, 0.8])
# The smallest n at which the narrowest perplexity target, FRACTIONS[0] (n - 1), exceeds one.
MIN_CELLS = 42
PCA_SEED = 20261022
# The channel set A of P_A and of its same-affinity control F_A.
CHANNEL_SETS = {
    '0': [0],
    '1': [1],
    '2': [2],
    '12': [1, 2],
    '012': [0, 1, 2],
    '3': [3],
    '23': [2, 3],
    '123': [1, 2, 3],
    '0123': [0, 1, 2, 3],
}


def normalize_counts(counts):
    """Counts per 10,000 per cell, log1p-transformed (sparse)."""
    x = counts.astype(float).tocsr(copy=True)
    lib = np.asarray(x.sum(1)).ravel()
    if not np.all(lib > 0):
        raise ValueError('every cell needs at least one count')
    x = sparse.diags(1e4 / lib) @ x
    x.data = np.log1p(x.data)
    return x.tocsr()


def train_representation(logx, solver='randomized'):
    """Gene selection, standardization and 30 principal components learned on these cells.

    solver='exact' computes a full singular value decomposition, as in the refitted design with
    exact PCA; solver='randomized' is used by the fixed design and by the refitted design with
    randomized PCA.
    """
    mean = np.asarray(logx.mean(0)).ravel()
    var = np.asarray(logx.power(2).mean(0)).ravel() - mean**2
    detected = np.asarray((logx > 0).sum(0)).ravel()
    eligible = np.flatnonzero((detected >= 3) & (var > 1e-12))
    chosen = eligible[np.lexsort((eligible, -var[eligible]))[:2000]]
    sd = np.sqrt(var[chosen])
    x = (logx[:, chosen].toarray() - mean[chosen]) / sd
    if solver == 'exact':
        pca = PCA(n_components=30, svd_solver='full').fit(x)
    elif solver == 'randomized':
        pca = PCA(
            n_components=30,
            svd_solver='randomized',
            random_state=PCA_SEED,
            n_oversamples=20,
            iterated_power=7,
        ).fit(x)
    else:
        raise ValueError(f'unknown solver {solver!r}')
    z = pca.transform(x)
    rms = np.sqrt(np.mean(pdist(z, 'sqeuclidean')))
    return {
        'genes': chosen,
        'mean': mean[chosen],
        'sd': sd,
        'pca_mean': pca.mean_,
        'components': pca.components_,
        'rms': rms,
    }


def transform(logx, model):
    """Scaled PC scores of cells under a learned representation, after standardizing the chosen genes."""
    x = (logx[:, model['genes']].toarray() - model['mean']) / model['sd']
    return ((x - model['pca_mean']) @ model['components'].T) / model['rms']


def perplexities(d2, h2):
    """Perplexity of each row of Gaussian affinities with squared width h2, excluding the diagonal."""
    logits = -d2 / (2 * h2)
    np.fill_diagonal(logits, -np.inf)
    lp = logits - logsumexp(logits, axis=1, keepdims=True)
    p = np.exp(lp)
    # The diagonal contributes no entropy; setting its log to zero avoids 0 * -inf.
    lp[np.diag_indices_from(lp)] = 0
    return np.exp(-(p * lp).sum(1))


def learn_widths(z):
    """Four squared widths h2 and log normalizers, by Brent search in log h2 over [-20, 10]."""
    d2 = squareform(pdist(z, 'sqeuclidean'))
    h2, log_norm = [], []
    off = ~np.eye(len(z), dtype=bool)
    for frac in FRACTIONS:
        target = frac * (len(z) - 1)
        logh2 = brentq(lambda h: np.median(perplexities(d2, np.exp(h))) - target, -20, 10, xtol=1e-12)
        h2.append(np.exp(logh2))
        log_norm.append(logsumexp(-d2[off] / (2 * h2[-1])) - np.log(off.sum()))
    return np.asarray(h2), np.asarray(log_norm)


def kernels(z, h2, log_norm):
    """The four Gaussian affinity channels C^(s), each with mean off-diagonal value one at calibration."""
    d2 = squareform(pdist(z, 'sqeuclidean'))
    gs = [np.exp(-d2 / (2 * s) - norm) for s, norm in zip(h2, log_norm)]
    for g in gs:
        np.fill_diagonal(g, 0.0)
    assert all(np.all(np.isfinite(g)) for g in gs)
    return gs
