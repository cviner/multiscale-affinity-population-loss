"""Map geometry scores: deformation and ten-neighbour retention."""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def clean(value):
    """Plain JSON types for NumPy values; nonfinite numbers become None."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def unique_ids(ids):
    """Cell identities as strings, required to be one-dimensional and unique."""
    ids = np.asarray(ids).astype(str)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('Cell IDs must be one-dimensional and unique')
    return ids


def deformation(f, s):
    """E = ||c s - f|| / ||f|| for reference and candidate pair distances f and s, with the least-squares c."""
    fnorm, snorm = np.linalg.norm(f), np.linalg.norm(s)
    scale = float(np.dot(f / fnorm, s / snorm) * (fnorm / snorm))
    return float(np.linalg.norm(scale * s - f) / fnorm)


def nearest(x, k):
    """Each row's k nearest other rows, with ties broken by row order."""
    d = squareform(pdist(x))
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1, kind='stable')[:, :k]


def retention(reference, candidate, k=10):
    """Mean fraction of each cell's k nearest neighbours in the reference that are also among its k
    nearest in the candidate; both arrays list the same cells in the same order."""
    k = min(k, len(reference) - 1)
    shared = sum(len(set(a) & set(b)) for a, b in zip(nearest(reference, k), nearest(candidate, k)))
    return shared / (len(reference) * k)


def geometry(reference, candidate):
    """Deformation and ten-neighbour retention of a candidate map."""
    return dict(
        shape_error=deformation(pdist(reference), pdist(candidate)),
        neighbor_recall=retention(reference, candidate),
    )
