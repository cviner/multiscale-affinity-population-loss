"""Tests of the RNA channels, of P against objective_check, and of the validation of saved maps."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT)]
from objective_check import recompute
from reconstruct import embed
from rna import FRACTIONS, kernels, learn_widths, perplexities
from score_maps import joined


def features(n=60, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=3, size=(4, 30))
    return centres[rng.integers(4, size=n)] + rng.normal(size=(n, 30))


def test_widths_meet_their_perplexity_targets():
    z = features()
    h2, log_norm = learn_widths(z)
    d2 = squareform(pdist(z, 'sqeuclidean'))
    for f, s in zip(FRACTIONS, h2):
        assert np.isclose(np.median(perplexities(d2, s)), f * (len(z) - 1), rtol=1e-8)
    off = ~np.eye(len(z), dtype=bool)
    assert all(np.isclose(g[off].mean(), 1.0) for g in kernels(z, h2, log_norm))


def test_two_range_fit_agrees_with_independent_objective():
    z = features(n=50, seed=1)
    h2, log_norm = learn_widths(z)
    x, record, starts = embed(z, h2, log_norm, 'P_12', seed=0)
    data = dict(features=z, h2=h2, log_norm=log_norm)
    for i, start in enumerate(record['starts']):
        objective, gradient = recompute(
            data,
            starts['start_coordinates'][i],
            starts['start_node_effects'][i],
            starts['start_intercepts'][i],
            [1, 2],
        )
        assert abs(objective - start['objective']) < 1e-10
        assert abs(gradient - start['gradient_inf_mean_loss']) < 1e-10
    assert x.shape == (50, 5) and np.isclose(x.mean(0), 0).all()


def test_maps_are_joined_by_identity_and_invalid_maps_rejected():
    rng = np.random.default_rng(2)
    ids = np.array([f'cell{i}' for i in range(20)])
    x = rng.normal(size=(20, 5))
    order = rng.permutation(20)
    assert np.array_equal(joined(ids, {'ids': ids[order], 'coords': x[order]}), x)
    for bad in [
        {'ids': np.repeat(ids[0], 20), 'coords': x},
        {'ids': ids, 'coords': np.zeros((20, 5))},
        {'ids': ids, 'coords': np.full((20, 5), np.nan)},
        {'ids': ids, 'coords': x[:, :3]},
        {'ids': ids[:19], 'coords': x[:19]},
    ]:
        with pytest.raises(ValueError):
            joined(ids, bad)


def test_command_line_from_counts_to_scores(tmp_path):
    rng = np.random.default_rng(3)
    counts = sparse.csr_matrix(rng.poisson(rng.gamma(0.3, 2, size=(60, 2600))) + np.eye(60, 2600, dtype=int))
    ids = [f'cell{i}' for i in range(60)]

    def run(script, *args):
        subprocess.run(
            [sys.executable, str(ROOT / script), *map(str, args)],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    for name, rows in [('all', slice(None)), ('arm', slice(0, 50))]:
        sparse.save_npz(tmp_path / f'{name}.npz', counts[rows])
        (tmp_path / f'{name}.txt').write_text('\n'.join(ids[rows]) + '\n')
        run(
            'prepare_input.py',
            'refitted',
            '--counts',
            f'{name}.npz',
            '--ids',
            f'{name}.txt',
            '--solver',
            'exact',
            '--out',
            f'{name}_input.npz',
        )
        run('fit.py', '--input', f'{name}_input.npz', '--method', 'PCA', '--out', f'{name}_map.npz')
    with np.load(tmp_path / 'all_input.npz') as z:
        np.savez(tmp_path / 'reference.npz', ids=z['ids'][:50], fixed_features=z['features'][:50])
    run(
        'score_maps.py',
        '--intact',
        'all_map.npz',
        '--refit',
        'arm_map.npz',
        '--reference',
        'reference.npz',
        '--out',
        'scores.json',
    )
    scores = json.loads((tmp_path / 'scores.json').read_text())
    assert scores['n_retained'] == 50 and scores['n_removed'] == 10
    assert 0 <= scores['neighbor_recall'] <= 1 and scores['shape_error'] >= 0
