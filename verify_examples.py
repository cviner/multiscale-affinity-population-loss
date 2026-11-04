"""Check the saved example against the result tables of the refitted design with exact PCA.

The script checks every file against its recorded hash. For every saved map and the fixed-PC5 and
constant controls, it recomputes the scores of score_maps.py and compares them with the tables; for
every P map, it recomputes the objective and gradient at both saved starts. It fits nothing.
"""

import os
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'code'))
import argparse
import csv
import gzip
import hashlib
import json

import numpy as np
import pandas as pd

from score_maps import fixed_pc5, load, score
from objective_check import recompute

ROOT = Path(__file__).resolve().parent
TOLERANCE = 1e-10
# pandas 3 parses floats exactly by default; pandas 2 needs the round-trip parser.
EXACT = {} if int(pd.__version__.split('.')[0]) >= 3 else {'float_precision': 'round_trip'}
GEOMETRY = ['shape_error', 'neighbor_recall']
FIDELITY = ['fixed_rna_neighbor_recall', 'own_rna_neighbor_recall']


def check_files():
    """Every example file and result table against the hashes in its manifest; returns the example manifest."""
    manifest = json.loads((ROOT / 'examples/MANIFEST.json').read_text())
    for path, digest in manifest['files'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    with (ROOT / 'results/MANIFEST.csv').open() as handle:
        for row in csv.DictReader(handle):
            data = gzip.decompress((ROOT / row['file']).read_bytes())
            assert hashlib.sha256(data).hexdigest() == row['sha256_uncompressed'], row['file']
            assert data.count(b'\n') == int(row['rows']) + 1, row['file']
    return manifest


def one(table, **keys):
    selected = table[np.logical_and.reduce([table[k] == v for k, v in keys.items()])]
    assert len(selected) == 1, keys
    return selected.iloc[0]


def main():
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    manifest = check_files()

    def read(name):
        table = pd.read_csv(ROOT / f'results/refitted_exact_pca/{name}.csv.gz', **EXACT)
        return table[table.seed == 0]

    geometry, fidelity, protein = read('geometry'), read('rna_fidelity'), read('protein_prediction')
    counts = dict(maps=0, controls=0, starts=0, geometry_values=0, protein_values=0)
    gaps = [0.0]

    def check(actual, expected):
        if actual is None or np.isnan(expected):
            # The constant control has no rank correlation.
            assert actual is None and np.isnan(expected), (actual, expected)
            return
        gaps.append(abs(float(actual) - float(expected)))
        assert gaps[-1] < TOLERANCE, (actual, expected)

    for cohort in manifest['cohorts']:
        protein_reference = load(ROOT / f'examples/protein/{cohort}.npz')
        intact_reference = load(ROOT / f'examples/references/{cohort}/intact.npz')
        for arm in manifest['arms']:
            reference = load(ROOT / f'examples/references/{cohort}/{arm}.npz')
            inputs = load(ROOT / f'examples/inputs/{cohort}/{arm}.npz')
            for method in manifest['methods'] + ['fixed_PC5', 'constant']:
                if method in ('fixed_PC5', 'constant'):
                    counts['controls'] += 1
                    intact, refit = fixed_pc5(intact_reference), fixed_pc5(reference)
                else:
                    counts['maps'] += 1
                    path = ROOT / f'examples/results/{cohort}/{arm}/{method}/seed_0.npz'
                    intact, refit = (
                        load(ROOT / f'examples/results/{cohort}/intact/{method}/seed_0.npz'),
                        load(path),
                    )
                scored = score(intact, refit, reference, protein_reference, constant=method == 'constant')
                if method.startswith('P_'):
                    record = json.loads(path.with_suffix('.json').read_text())
                    for i, start in enumerate(record['starts']):
                        objective, gradient = recompute(
                            inputs,
                            refit['start_coordinates'][i],
                            refit['start_node_effects'][i],
                            refit['start_intercepts'][i],
                            record['channels'],
                        )
                        check(objective, start['objective'])
                        check(gradient, start['gradient_inf_mean_loss'])
                        counts['starts'] += 1
                if method != 'constant':
                    # The intact arm has no removal, so only its RNA fidelity is scored.
                    tables = (
                        [(fidelity, FIDELITY)]
                        if arm == 'intact'
                        else [(geometry, GEOMETRY), (fidelity, FIDELITY)]
                    )
                    for table, columns in tables:
                        row = one(table, cohort=cohort, arm=arm, method=method)
                        for column in columns:
                            check(scored[column], row[column])
                            counts['geometry_values'] += 1
                expected = protein[
                    (protein.cohort == cohort) & (protein.arm == arm) & (protein.method == method)
                ]
                values = [v for v in scored['protein'] if arm != 'intact' or v['view'] == 'refit']
                assert len(values) == len(expected), (cohort, arm, method)
                for value in values:
                    row = one(expected, **{k: value[k] for k in ['view', 'population', 'marker', 'mode']})
                    assert row['n'] == value['n']
                    for column in ['mse', 'spearman']:
                        check(value[column], row[column])
                        counts['protein_values'] += 1
    print(json.dumps(dict(complete=True, **counts, max_abs_gap=max(gaps))))


if __name__ == '__main__':
    main()
