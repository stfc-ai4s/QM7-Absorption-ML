# Reproducing the figures

Every figure can be redrawn from data committed in this repository — no model
weights, no density cubes, no download:

```bash
bash scripts/reproduce_figures.sh
```

On the pinned `environment.yml` all five PNGs come out byte-identical to the
committed ones. Write to a scratch directory to check without overwriting:

```bash
bash scripts/reproduce_figures.sh /tmp/check
for f in 3 4 5 6 7; do cmp /tmp/check/figure_$f.png figures/figure_$f.png; done
```

Regenerating the *inputs* to those plots is a separate, heavier step, described
per figure below. For Figures 3, 4 and 5 it needs both released records, the
weights and the dataset — see [DATA.md](DATA.md). The Figure 6/7 summary CSV
regenerates from committed data alone.

## Overview

| Figure | Shows | Plotted from | Drawn by |
|---|---|---|---|
| 3 | CNN spectra vs TDDFT for three molecules | `data/figures_3_4/spectrum_prediction_M*.dat` | `make_figures_3_4_5.py` |
| 4 | Two similar molecules, true vs predicted | the same four tables | `make_figures_3_4_5.py` |
| 5 | Predicted vs true normalisation constant | `data/figure_5/val_preds_true_raw.npz` | `make_figures_3_4_5.py` |
| 6 | Median Pearson correlation by chemical category | `data/plotting/figure_6/category_delta_true_minus_false_pearson.csv` | `make_figures_6_7.py` |
| 7 | Difference in median Pearson, present minus absent | the same CSV | `make_figures_6_7.py` |

Figures 6 and 7 share one summary CSV, which is why both come from the
`figure_6` data directory.

## Figures 3 and 4

```bash
python scripts/plotting/make_figures_3_4_5.py              # figures 3 and 4
python scripts/plotting/make_figures_3_4_5.py --figures 3 4 5
```

Outputs `figures/figure_3.png`, `figures/figure_4.png` and
`figures/figures_3_4_metrics.csv`.

### Inputs

| Item | Path |
|---|---|
| Molecule ids, panel assignment, SMILES | `data/figures_3_4/molecules.json` |
| Prediction tables | `data/figures_3_4/spectrum_prediction_M{481,972,3935,4170}.dat` |
| Provenance of the tables | `data/figures_3_4/prediction_run.json` |

`molecules.json` drives everything: `figure_3` lists the three panel molecules,
`figure_4` the two compared ones, and `molecules` maps each id to the SMILES
used for the inset structure drawing.

### What the plotter does and does not do

Saved values and the saved energy column are plotted directly. No
interpolation, no smoothing, no renormalisation. Tables must therefore already
follow the convention in [INFERENCE.md](INFERENCE.md#the-output-table): three
columns `energy_au reference prediction`, 900 bins, each intensity normalised
to unit sum within that window. An old table that stretched 2000 output bins
across 0–0.45 a.u. cannot be fixed by relabelling axes; regenerate it on the
correct grid.

`figures_3_4_metrics.csv` holds the Pearson correlation and MAE of each
**displayed molecule**, plus spectrum sums, bin counts and energy endpoints.
These are per-example numbers, not validation-set averages. Pearson here is
NumPy's `corrcoef`; the Figure 6/7 analysis uses its own epsilon-regularised
definition and the two are not interchangeable.

### Regenerating the four tables

Needs the CNN checkpoint and the eight density/reference files listed in
[DATA.md](DATA.md#the-one-exception-figure-3-and-4-inputs).

```bash
python scripts/analysis/generate_figures_3_4_data.py \
    --model /your/scratch/qm7-spectra/weights/cnn_density_spectrum.pt
python scripts/plotting/make_figures_3_4_5.py
```

Or read the dataset directly instead of copying inputs in:

```bash
python scripts/analysis/generate_figures_3_4_data.py \
    --model /your/scratch/qm7-spectra/weights/cnn_density_spectrum.pt \
    --dataset-root /your/scratch/qm7-spectra/dataset
```

The model is loaded once and the four tables are overwritten in place.
`prediction_run.json` records the checkpoint's SHA-256 and the preprocessing
options. To compare against the archived tables before replacing them, write
somewhere else and plot that directory:

```bash
python scripts/analysis/generate_figures_3_4_data.py \
    --model .../cnn_density_spectrum.pt --output-dir /tmp/new_tables
python scripts/plotting/make_figures_3_4_5.py --data-dir /tmp/new_tables --output-dir /tmp/new_figs
```

### Known label discrepancy: M4170

`--check-labels` verifies `molecules.json` against
`data/figure_6/validation_molecule_numbers_ordered_with_smiles.csv`. It is off
by default because M4170 fails it:

| Source | SMILES |
|---|---|
| `molecules.json` (paper label) | `C=C(CC)NC=[N]` |
| validation CSV (from the geometry) | `C=C(CC)NC=N` |

They differ by one hydrogen on the terminal nitrogen — a nitrene radical versus
an imine. M481, M972 and M3935 all match. The published label is kept so
regenerated figures stay comparable with the paper; only the printed inset
label is affected, never a spectrum. Resolve it against the original molecule
metadata before a final release, then turn the check on.

Figure 4's molecule assignment is also worth knowing about: the original
Figure 3 mapped M481 to `C#CC#CCOC` while Figure 4 attached a different SMILES
to the same spectrum. The manifest now orders Figure 4 as M972 then M481,
consistent with Figure 3.

## Figure 5

```bash
python scripts/plotting/make_figures_3_4_5.py --figures 5
```

Reads `data/figure_5/val_preds_true_raw.npz`, which holds aligned `pred` and
`true` arrays of raw spectrum normalisation constants over the validation set,
and plots a parity scatter with R² beside an error histogram. Needs
scikit-learn. `--normalization-npz` points at a different file; extra keys in
the NPZ are ignored, only `pred` and `true` are read.

### What is being plotted

The spectrum models all end in a softmax, so they predict a unit-sum shape and
nothing about intensity. A separate model predicts that missing scale: the sum
of the first 900 points of the reference spectrum, one scalar per molecule.
Figure 5 is how well it does. The 1031 points are the same 1031 validation
molecules the Figure 6/7 analysis uses — the normalization model is trained by
`scripts/training/train_density_normconst_from_cache.py` with the same
`--seed 0 --val-frac 0.15` split as every other model here.

### Regenerating the input

```bash
python scripts/inference/predict_normconst_from_density.py validation \
    --model   /your/scratch/qm7-spectra/weights/normconst_model.pt \
    --config  /your/scratch/qm7-spectra/runs/normconst/config.json \
    --split   /your/scratch/qm7-spectra/runs/normconst/split.json \
    --targets /your/scratch/qm7-spectra/runs/normconst/targets_sum900.npz \
    --output  /tmp/val_preds_true_raw.npz
python scripts/plotting/make_figures_3_4_5.py --figures 5 \
    --normalization-npz /tmp/val_preds_true_raw.npz --output-dir /tmp
```

This needs the cache shards as well as the checkpoint, because it reads the
validation densities out of the cache. See [DATA.md](DATA.md).

One caveat if you compare against the committed file: the original run saved
its parity data at the **final** epoch, while the training script now saves it
at the same epoch as the best checkpoint. Predictions from the released
checkpoint therefore need not match `data/figure_5/val_preds_true_raw.npz`
point for point.

## Figures 6 and 7

```bash
python scripts/analysis/generate_figure_6_data.py    # rebuild the summary CSV
python scripts/plotting/make_figures_6_7.py          # draw both figures
```

The CSV is committed, so the plotting step alone is enough; the analysis step
regenerates it from the committed validation predictions and reproduces it
exactly.

### Inputs

| Item | Path |
|---|---|
| Validation predictions (`pred`, `true`, `val_idx`) | `data/figure_6/val_preds_true.npz` |
| Split of the run that produced them | `data/figure_6/indices.json` |
| Cache index mapping positions to molecules | `data/figure_6/cache_index.json` |
| Molecule id → SMILES | `data/figure_6/validation_molecule_numbers_ordered_with_smiles.csv` |
| Geometries for element flags | `data/figure_6/xyz/<id>/GEO_M<id>.xyz` |

The NPZ's `val_idx` fixes the row order; `indices.json` must agree with it and
the script errors out if it does not. Molecule identity comes from
`samples[val_idx[i]]["dens_npz"]` in the cache index. There is no fall-back to
row order for the SMILES lookup — a missing or unparseable molecule is an
error, never silently assigned to the category-absent group.

### The analysis

Spectra are cast to float64, clipped at zero and normalised to unit sum before
a Pearson correlation is taken per molecule, with the original epsilon of
1e-12. Zero-total spectra become uniform; constant spectra give Pearson zero.

Each molecule then gets boolean chemistry flags: element presence from the XYZ
atoms, and bond and functional-group flags from RDKit on the SMILES. For each
category the script reports the median Pearson correlation of the molecules
where it is present and where it is absent, and the difference between them.
Categories overlap — every row is an independent present-versus-absent
comparison, not a partition. A category with an empty group is dropped.

Output columns: `category, n_false, n_true, median_false, median_true,
delta_median_true_minus_false`, sorted by the difference. True means the
category is present. A negative difference means molecules containing that
feature are predicted less well.

### Using a different run

```bash
python scripts/analysis/generate_figure_6_data.py \
    --run-dir /your/scratch/qm7-spectra/runs/cnn/exp_003 \
    --cache-index /your/scratch/qm7-spectra/cache/index.json \
    --output /tmp/summary.csv
python scripts/plotting/make_figures_6_7.py --data-file /tmp/summary.csv --output-dir /tmp
```

The CNN sweep writes `val_preds_true.npz` (with `val_idx`) and `indices.json`
into each experiment directory, so a run directory can be used as `--run-dir`
unchanged. The `--xyz-root` and `--smiles-csv` defaults still point into
`data/figure_6/`; override them for molecules not covered there.

## Common failures

**`Missing paper table: .../spectrum_prediction_M481.dat`** — the four tables
are committed; you are probably pointing `--data-dir` somewhere else.

**`Wrong column order`, or `must contain three finite columns`** — the table
was written with the old `energy prediction reference` ordering. Regenerate it
with the current predictor.

**`Median differences disagree with the group medians`** — the summary CSV was
rounded. Use the full-precision file, or regenerate it.

**`Figure 5 requires ...`** — pass `--normalization-npz`, or omit `5` from
`--figures`.

**RDKit cannot parse a SMILES** — Figures 3, 4 and 6/7 all draw or analyse
structures from SMILES strings. Fix the entry in `molecules.json` or in the
validation CSV rather than working around it in code.
