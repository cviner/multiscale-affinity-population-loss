"""Prepare a fit input (cell IDs, 30 RNA features, four squared widths, and log normalizers), with a
.json record of the design, the PCA solver, and the selected genes.

Every count matrix is a SciPy sparse .npz of already quality-filtered raw RNA counts, cells by
genes, with the same genes in the same order. Each ID file has one unique cell ID per line,
in the order of the rows of its counts.

  refitted design: learn the representation and the widths from the cells of one arm
      python prepare_input.py refitted --counts arm.npz --ids arm.txt --solver exact --out arm_input.npz
  fixed design: learn the representation on calibration set A, the widths on calibration set B
  projected through it, and project the evaluation cells; an arm keeps the rows of its retained
  cells and the same widths
      python prepare_input.py fixed --calibration-a a.npz --calibration-b b.npz
          --counts evaluation.npz --ids evaluation.txt [--retain arm.txt] --out arm_input.npz
"""

import os

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault('NUMBA_CACHE_DIR', str(Path(tempfile.gettempdir()) / 'population-loss-numba'))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'code'))
import argparse
import json
import numpy as np
from scipy import sparse
from rna import MIN_CELLS, normalize_counts, train_representation, transform, learn_widths


def load_counts(path):
    counts = sparse.load_npz(path).tocsr()
    if not np.isfinite(counts.data).all() or np.any(counts.data < 0):
        raise ValueError(f'{path}: counts must be finite and nonnegative')
    return counts


def load_ids(path, n):
    ids = np.asarray(Path(path).read_text().splitlines())
    if len(set(ids)) != len(ids) or len(ids) != n or np.any(ids == ''):
        raise ValueError(f'{path}: IDs must be unique, nonempty and match the count rows')
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('design', choices=['refitted', 'fixed'], help='preprocessing design')
    ap.add_argument(
        '--counts', type=Path, required=True, help='cells to represent (one arm, or all evaluation cells)'
    )
    ap.add_argument(
        '--ids', type=Path, required=True, help='one cell ID per line, in the row order of --counts'
    )
    ap.add_argument(
        '--solver',
        choices=['exact', 'randomized'],
        help='principal components: exact SVD or randomized PCA '
        '(required for the refitted design; the fixed design defaults to randomized)',
    )
    ap.add_argument('--calibration-a', type=Path, help='fixed design: counts that train the representation')
    ap.add_argument('--calibration-b', type=Path, help='fixed design: disjoint counts that set the widths')
    ap.add_argument('--retain', type=Path, help='fixed design: IDs of the retained cells of one arm')
    ap.add_argument(
        '--out', type=Path, required=True, help='new .npz file; a .json record is written beside it'
    )
    args = ap.parse_args()
    if args.out.suffix != '.npz' or args.out.exists() or args.out.with_suffix('.json').exists():
        raise ValueError('The output exists; choose a new path')
    counts = load_counts(args.counts)
    ids = load_ids(args.ids, counts.shape[0])
    if args.design == 'refitted':
        if args.solver is None or args.calibration_a or args.calibration_b or args.retain:
            raise ValueError('The refitted design takes --solver and no calibration sets')
        logx = normalize_counts(counts)
        model = train_representation(logx, args.solver)
        features = transform(logx, model)
        h2, log_norm = learn_widths(features)
        keep = np.ones(len(ids), dtype=bool)
        solver = args.solver
    else:
        if not (args.calibration_a and args.calibration_b):
            raise ValueError('The fixed design needs --calibration-a and --calibration-b')
        solver = args.solver or 'randomized'
        a, b = load_counts(args.calibration_a), load_counts(args.calibration_b)
        if not a.shape[1] == b.shape[1] == counts.shape[1]:
            raise ValueError('All count matrices must have the same genes')
        model = train_representation(normalize_counts(a), solver)
        h2, log_norm = learn_widths(transform(normalize_counts(b), model))
        features = transform(normalize_counts(counts), model)
        keep = np.ones(len(ids), dtype=bool)
        if args.retain:
            retained = set(Path(args.retain).read_text().splitlines())
            if not retained <= set(ids):
                raise ValueError('Retained IDs must be evaluation cells')
            keep = np.isin(ids, list(retained))
    if keep.sum() < MIN_CELLS:
        raise ValueError(f'At least {MIN_CELLS} cells are required')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, ids=ids[keep], features=features[keep], h2=h2, log_norm=log_norm)
    record = dict(design=args.design, solver=solver, n=int(keep.sum()), genes=model['genes'].tolist())
    args.out.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
    print(args.out)


if __name__ == '__main__':
    main()
