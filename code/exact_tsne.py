"""t-SNE in five dimensions with an exact Cauchy objective.

openTSNE runs the optimization, but its Barnes-Hut gradient is not meant for more than three
output dimensions and its FFT gradient allows at most two. Its negative-gradient callback is
therefore given scikit-learn's exact Kullback-Leibler divergence and gradient.
"""

import numpy as np
from scipy.spatial.distance import squareform
from sklearn.manifold._t_sne import _kl_divergence
from openTSNE import TSNEEmbedding, affinity, initialization


def exact_objective(embedding, P, dof=1, reference_embedding=None, should_eval_error=False, **unused):
    """KL divergence and gradient of a Cauchy embedding, evaluated exactly, in openTSNE's callback form."""
    if reference_embedding is not None:
        raise ValueError('Full refitting only')
    if dof != 1:
        raise ValueError('Only the Cauchy kernel (dof=1) is supported')
    n, d = embedding.shape
    probabilities = squareform(P.toarray(), checks=False)
    error, gradient = _kl_divergence(
        np.asarray(embedding).ravel(), probabilities, 1, n, d, compute_error=should_eval_error
    )
    # openTSNE omits the common factor four and uses correspondingly larger
    # learning rates, so the gradient is divided by four to match its convention.
    return error, gradient.reshape(n, d) / 4


def optimize(z, affinities, seed, dimension=5, iterations=(250, 500)):
    """From the PCA start of z: 250 iterations with exaggeration 12 and momentum 0.5, then 500 without
    exaggeration and with momentum 0.8. Returns the embedding."""
    initial = initialization.pca(z, n_components=dimension, random_state=seed)
    em = TSNEEmbedding(
        initial,
        affinities=affinities,
        negative_gradient_method=exact_objective,
        dof=1,
        n_jobs=1,
        random_state=seed,
    )
    em = em.optimize(n_iter=iterations[0], exaggeration=12, momentum=0.5, learning_rate='auto', inplace=True)
    return em.optimize(n_iter=iterations[1], exaggeration=1, momentum=0.8, learning_rate='auto', inplace=True)


def fit(z, seed, dimension=5):
    """Multiscale t-SNE (perplexities 15 and 60, exact neighbours); returns (map, record)."""
    a = affinity.Multiscale(
        z, perplexities=[15, 60], method='exact', metric='euclidean', n_jobs=1, random_state=seed
    )
    em = optimize(z, a, seed, dimension)
    return np.array(em), {'kl_divergence': float(em.kl_divergence)}
