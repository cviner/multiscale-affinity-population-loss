# Multiscale affinity reconstruction under cell population loss

## Installation

```
pip install -r requirements.txt
```

## Checking the saved maps

```
python verify_examples.py
```

Recomputes the scores of the 42 supplied maps and compares them with `results/refitted_exact_pca/`. It does not refit anything, and any mismatch stops it with an assertion error.

The saved maps of the full study and the count matrices are not included.

## Running the analysis

```
python prepare_input.py refitted --counts arm.npz --ids arm.txt --solver exact --out arm_input.npz
python prepare_input.py fixed --calibration-a a.npz --calibration-b b.npz --counts evaluation.npz --ids evaluation.txt --retain arm.txt --out arm_input.npz
python fit.py --input arm_input.npz --method P_12 --seed 0 --out maps/P_12.npz
python score_maps.py --intact intact/P_12.npz --refit maps/P_12.npz --reference reference.npz --protein-reference protein.npz --out scores.json
```

`prepare_input.py` takes quality-filtered raw RNA counts as a SciPy sparse `.npz` matrix, with cells in rows and the same genes in the same order in every matrix, and a text file of cell IDs, one per line in row order. An input needs at least 42 cells.

`--method` is one of:

- `P_0`, `P_1`, `P_2`, `P_12`, `P_012`, `P_3`, `P_23`, `P_123`, or `P_0123`, model P on those channels, from 0 (narrowest) to 3 (broadest)
- `F_` with the same suffixes, the same-affinity control
- `UMAP_k` or `UMAP_pc_k`, with spectral or principal-component initialization and k = 15, 60, 150, or 300
- `TSNE_multiscale` or `PCA`

`examples/references/` and `examples/protein/` show the reference formats.

## Result tables

`results/` has one folder per design. Arms are `intact`, `remove_g` (population `g` removed), and `random_g_k` (random mask `k` of the size of population `g`).

- In `geometry.csv.gz`, `shape_error` is the deformation E and `neighbor_recall` is ten-neighbour retention.
- In `protein_prediction.csv.gz`, `view` is `refit` or `restricted_intact`, the intact map restricted to the same cells.
- `refitted_exact_pca/gene_selection_turnover.csv.gz` holds the gene counts of Table S8.

## Data

`examples/` is derived from two public CITE-seq datasets:

- Cord blood mononuclear cells: Stoeckius et al., *Nature Methods* 2017 ([doi:10.1038/nmeth.4380](https://doi.org/10.1038/nmeth.4380)), GEO [GSE100866](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100866). NCBI places no restrictions on the use or distribution of GEO data.
- Peripheral blood mononuclear cells: 10x Genomics, *10k PBMCs from a Healthy Donor - Gene Expression with a Panel of TotalSeq-B Antibodies* (2018, sample `pbmc_10k_protein_v3`), under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

`examples/inputs/` and `examples/references/` hold RNA principal-component scores computed from these data. `examples/protein/` holds population labels, prediction folds, and normalized CD4, CD16, and CD45RA levels. Supplementary Methods S1 of the paper lists the source files.

## Licence

The code is under the GNU General Public License, version 3 (see `LICENSE`). The GPL does not cover `examples/inputs/`, `examples/references/`, or `examples/protein/`, which stay under the terms of their sources above.
