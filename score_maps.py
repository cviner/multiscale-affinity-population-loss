"""Score retained-cell geometry, RNA fidelity and held-out protein prediction, joining maps by cell ID.

Two controls need no fit: fixed_PC5 is the first five components of the fixed RNA reference, and
constant predicts each cell's markers by the population mean of the opposite fold.
"""

import os

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'code'))
import argparse
import json
import numpy as np
from scipy.spatial.distance import pdist
from metrics import clean, geometry, retention, unique_ids
from protein import correlation, neighbors, predict


def load(path):
    """All arrays of an .npz file, without pickled objects."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def joined(expected, saved, subset=False):
    """Coordinates of a saved map for the expected cells, in their order; rejects invalid maps."""
    if not {'ids', 'coords'} <= set(saved):
        raise ValueError('A map must contain ids and coords')
    ids, expected = unique_ids(saved['ids']), unique_ids(expected)
    x = np.asarray(saved['coords'], float)
    if x.shape != (len(ids), 5) or not np.isfinite(x).all():
        raise ValueError('Map must contain finite five-dimensional coordinates')
    if not set(expected) <= set(ids) or (not subset and set(expected) != set(ids)):
        raise ValueError('Map identities do not match the requested cell set')
    index = {v: i for i, v in enumerate(ids)}
    x = x[[index[v] for v in expected]]
    if np.linalg.norm(pdist(x)) <= np.finfo(float).tiny:
        raise ValueError('Collapsed map')
    return x


def fixed_pc5(reference):
    """The fixed-PC5 map of the cells of an RNA reference file."""
    return {'ids': reference['ids'], 'coords': reference['fixed_features'][:, :5]}


def protein_scores(views, ids, protein_reference, constant=False):
    """Within-population prediction of each held-out marker from opposite-fold neighbours in each view.

    views: {name: coordinates of the cells ids}; constant=True predicts the opposite fold's mean.
    """
    source_ids = unique_ids(protein_reference['ids'])
    if not set(ids) <= set(source_ids):
        raise ValueError('Missing protein reference cell identities')
    index = {v: i for i, v in enumerate(source_ids)}
    ix = np.array([index[v] for v in ids])
    populations, folds = protein_reference['labels'][ix], protein_reference['folds'][ix]
    if not set(np.unique(folds)) <= {0, 1}:
        raise ValueError('Protein reference must contain the two folds')
    panel = protein_reference['panel'].astype(str)
    rows = []
    for view, x in views.items():
        for pop in np.unique(populations):
            local, nn = neighbors(x, populations, folds, pop)
            if nn is None:
                continue
            for mode in ['raw', 'adjusted']:
                target = protein_reference[mode][ix][local]
                prediction = predict(target, nn, folds[local], constant=constant)
                rho = np.full(len(panel), np.nan) if constant else correlation(target, prediction)
                error = prediction - target
                for j, marker in enumerate(panel):
                    rows.append(
                        dict(
                            view=view,
                            population=int(pop),
                            marker=marker,
                            mode=mode,
                            n=len(local),
                            mse=float(np.mean(error[:, j] ** 2)),
                            spearman=float(rho[j]),
                        )
                    )
    return rows


def score(intact, refit, reference, protein_reference=None, constant=False):
    """Scores of the map of one arm (refit) against the intact map, the RNA references and the proteins."""
    ids = unique_ids(reference['ids'])
    full, reduced = joined(ids, intact, subset=True), joined(ids, refit)
    result = dict(n_retained=len(ids), n_removed=len(intact['ids']) - len(ids), **geometry(full, reduced))
    for field, name in [('fixed_features', 'fixed_rna'), ('own_features', 'own_rna')]:
        if field in reference:
            result[f'{name}_neighbor_recall'] = retention(reference[field], reduced)
    if protein_reference is not None:
        views = {'refit': reduced, 'restricted_intact': full}
        result['protein'] = protein_scores(views, ids, protein_reference, constant)
    return clean(result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--intact',
        type=Path,
        required=True,
        help='map fitted to all cells, or with --control the RNA reference of the intact arm',
    )
    ap.add_argument('--refit', type=Path, help='map fitted to the retained cells (.npz)')
    ap.add_argument(
        '--control', choices=['fixed_PC5', 'constant'], help='score a control instead of a fitted map'
    )
    ap.add_argument(
        '--reference', type=Path, required=True, help='RNA reference of the retained cells (.npz)'
    )
    ap.add_argument('--protein-reference', type=Path, help='populations, folds and protein targets (.npz)')
    ap.add_argument('--out', type=Path, required=True, help='new JSON file')
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError('The output exists; choose a new path')
    if (args.refit is None) == (args.control is None):
        raise ValueError('Give either --refit or --control')
    reference = load(args.reference)
    if args.control:
        intact, refit = fixed_pc5(load(args.intact)), fixed_pc5(reference)
    else:
        intact, refit = load(args.intact), load(args.refit)
    protein = load(args.protein_reference) if args.protein_reference else None
    result = score(intact, refit, reference, protein, constant=args.control == 'constant')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(args.out)


if __name__ == '__main__':
    main()
