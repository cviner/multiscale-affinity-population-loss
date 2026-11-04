"""Fit one configuration to a fit input (cell IDs, 30 RNA features, h2 and log_norm).

Output: a new .npz map (cell IDs and five-dimensional coordinates, plus the parameters of both
starts for P) and a .json record of the fit.
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
import time
import numpy as np
from comparators import METHODS as COMPARATORS, embed_comparator
from reconstruct import P_CHANNELS, embed
from rna import MIN_CELLS

F_METHODS = ['F' + m[1:] for m in P_CHANNELS]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True, help='fit input written by prepare_input.py (.npz)')
    ap.add_argument(
        '--out', type=Path, required=True, help='new .npz file; a .json record is written beside it'
    )
    ap.add_argument('--method', choices=list(P_CHANNELS) + F_METHODS + COMPARATORS, required=True)
    ap.add_argument(
        '--seed', type=int, default=0, help='seed of the starts and of the stochastic comparators'
    )
    args = ap.parse_args()
    if args.out.suffix != '.npz' or args.out.exists() or args.out.with_suffix('.json').exists():
        raise ValueError('The output exists; choose a new path')
    with np.load(args.input, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if set(data) != {'ids', 'features', 'h2', 'log_norm'}:
        raise ValueError('A fit input holds ids, features, h2 and log_norm')
    ids, features = data['ids'].astype(str), np.asarray(data['features'], float)
    if features.shape != (len(ids), 30) or len(set(ids)) != len(ids) or len(ids) < MIN_CELLS:
        raise ValueError(f'Expected at least {MIN_CELLS} uniquely identified cells with 30 RNA components')
    if not all(np.isfinite(data[k]).all() for k in ['features', 'h2', 'log_norm']):
        raise ValueError('Nonfinite fit input')
    if (
        data['h2'].shape != (4,)
        or data['log_norm'].shape != (4,)
        or not np.all(np.diff(data['h2']) > 0)
        or data['h2'][0] <= 0
    ):
        raise ValueError('Four ordered positive squared widths and four normalizers are required')
    began = time.perf_counter()
    parameters = {}
    if args.method in P_CHANNELS:
        x, info, parameters = embed(features, data['h2'], data['log_norm'], args.method, args.seed)
    elif args.method in F_METHODS:
        from same_affinity import fit

        x, info = fit(features, data['h2'], data['log_norm'], args.method, args.seed)
    else:
        x, info = embed_comparator(features, args.method, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, ids=data['ids'], coords=x, **parameters)
    record = dict(method=args.method, seed=args.seed, n=len(ids), seconds=time.perf_counter() - began, **info)
    args.out.with_suffix('.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(args.out)


if __name__ == '__main__':
    main()
