"""The comparators, on the same 30-dimensional RNA input as P: fresh PCA, UMAP with spectral or
PC-based initialization, and multiscale t-SNE with an exact five-dimensional Cauchy objective."""

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

UMAP_NEIGHBORS = (15, 60, 150, 300)
METHODS = (
    ['PCA', 'TSNE_multiscale']
    + [f'UMAP_{k}' for k in UMAP_NEIGHBORS]
    + [f'UMAP_pc_{k}' for k in UMAP_NEIGHBORS]
)


def embed_comparator(z, method, seed, dimension=5):
    """Fit one comparator to the RNA features z; returns (coordinates, record)."""
    if method == 'PCA':
        return PCA(n_components=dimension, svd_solver='full').fit(z).transform(z), {}
    if method == 'TSNE_multiscale':
        from exact_tsne import fit

        return fit(z, seed, dimension)
    if method.startswith('UMAP_'):
        import umap

        init = 'spectral'
        if method.startswith('UMAP_pc_'):
            # The leading input components with a seeded perturbation, at the input's RMS pair distance.
            init = z[:, :dimension] + np.random.default_rng(seed).normal(size=(len(z), dimension)) * 0.001
            init -= init.mean(0)
            init *= np.sqrt(np.mean(pdist(z, 'sqeuclidean')) / np.mean(pdist(init, 'sqeuclidean')))
        model = umap.UMAP(
            n_components=dimension,
            n_neighbors=min(int(method.rsplit('_', 1)[1]), len(z) - 1),
            min_dist=0.1,
            n_epochs=500,
            metric='euclidean',
            init=init,
            random_state=seed,
            n_jobs=1,
        )
        return model.fit_transform(z).astype(float), {}
    raise ValueError(method)
