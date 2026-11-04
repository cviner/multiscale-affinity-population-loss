"""Held-out surface-protein prediction from map neighbours within a population and in the opposite fold."""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata


def neighbors(x, labels, folds, pop):
    """Cells of population pop and, for each, its ten nearest map neighbours in the other fold."""
    ix = np.flatnonzero(labels == pop)
    f = folds[ix]
    counts = np.bincount(f, minlength=2)
    if min(counts) < 10:
        return ix, None
    local = np.empty((len(ix), 10), dtype=int)
    for fold in range(2):
        query = np.flatnonzero(f == fold)
        train = np.flatnonzero(f != fold)
        d = cdist(x[ix[query]], x[ix[train]], 'sqeuclidean')
        local[query] = train[np.argsort(d, axis=1, kind='stable')[:, :10]]
    return ix, local


def predict(y, nn, folds, constant=False):
    """Neighbour-mean prediction, or with constant=True the other fold's population mean."""
    if not constant:
        return y[nn].mean(1)
    pred = np.empty_like(y)
    for fold in range(2):
        pred[folds == fold] = y[folds != fold].mean(0)
    return pred


def correlation(y, pred):
    """Spearman correlation of each marker between observed and predicted values."""
    yr = rankdata(y, axis=0)
    pr = rankdata(pred, axis=0)
    yr -= yr.mean(0)
    pr -= pr.mean(0)
    den = np.sqrt((yr**2).sum(0) * (pr**2).sum(0))
    result = np.full(y.shape[1], np.nan)
    np.divide((yr * pr).sum(0), den, out=result, where=den > 0)
    return result
