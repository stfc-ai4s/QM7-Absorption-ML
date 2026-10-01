# Validation

Two rounds of checking, recorded in the order they happened.

## Round 2: the repository reorganization

This round had the real data: the committed figure tables, the published
validation predictions, and the split of the published CNN run.

**Byte-identical figure reproduction.** A baseline of all five PNGs and both
CSVs was taken before any edit and re-checked after every batch of changes.
`bash scripts/reproduce_figures.sh /tmp/check` reproduces `figure_3.png`
through `figure_7.png`, `figures/figures_3_4_metrics.csv` and
`data/plotting/figure_6/category_delta_true_minus_false_pearson.csv` byte for
byte on the pinned `environment.yml`.

**The train/validation split.** `build_density_spectrum_cache.py` drew its
split with the legacy `np.random.seed(0); np.random.shuffle(...)` API, while
the other four training scripts used `np.random.default_rng(0)`. The two
produce different permutations, so the CNN sweep was holding out a different
set of molecules from every other model family.

Which one the published run used was settled empirically against
`data/figure_6/indices.json`: `default_rng(0)` reproduces both its `val_idx`
and its `train_idx` exactly, the legacy API does not. The CNN builder was
changed to `default_rng`, so all five scripts now agree with each other and
with the published split — 5843 train / 1031 validation for the 6874-molecule
cache at `--seed 0 --val-frac 0.15`. Sharding involves no randomness at all:
position in the sorted density glob determines shard and offset.

**The Figure 6 loop is closed.** The CNN sweep now writes `indices.json` and a
`val_preds_true.npz` carrying `val_idx`, at the same epoch as the best
checkpoint — exactly the two files `generate_figure_6_data.py` consumes. A
fresh run's experiment directory can be passed to `--run-dir` unchanged.

**The normalization-constant model.** Two scripts added late
(`train_normconst_sum900_from_cache.py`, `predict_normconst_from_pt.py`) were
renamed to `scripts/training/train_density_normconst_from_cache.py` and
`scripts/inference/predict_normconst_from_density.py`. The trainer's
hardcoded `/home/ubuntu/datasets/...` cache path and its module-level
`mkdir` were replaced by argparse, with every default left at the published
value.

Its split was checked to be the shared one: `default_rng(0).permutation(6874)`
sliced at `round(0.15 * 6874) = 1031` reproduces `data/figure_6/indices.json`
exactly, for both the validation and the training half. The committed
`data/figure_5/val_preds_true_raw.npz` has 1031 rows, confirming the run
covered the full cache — so Figure 5's molecules are the Figure 6/7 molecules.

Two behaviours changed deliberately. The script now saves
`val_preds_true_raw.npz` at the best epoch rather than the last, so the saved
parity data describes the saved checkpoint, and it records `val_ds_idx` and
`epoch` alongside `pred`/`true`. It also exports a TorchScript
`normconst_model.pt`, which the predictor expected but nothing previously
produced; the predictor now additionally accepts the `best_model.pth` state
dict, rebuilding the architecture from `config.json`.

Exercised on a synthetic 16³ cache: forward pass, TorchScript export,
TorchScript load and state-dict load agree to zero absolute difference; target
un-standardization inverts correctly; 3-D, 4-D and 5-D single-density inputs
are all accepted and mismatched shapes rejected; the `validation` mode runs
end to end and its output rows follow `val_ds_idx` in order. Not run against
real weights or densities.

**Imports.** All three spectrum predictors import their shared
`spectrum_io` when loaded as modules, not only when run as scripts, so they
always write the same table format. A `KeyError` in
`predict_spectrum_from_xyz_schnet.py`, raised when the module was loaded
through an explicit loader rather than by name, was fixed and the import path
re-checked.

**Not re-verified here.** Training was not re-run: the split fix is verified
against the recorded indices of the published run, not by retraining. The
geometry predictors (SchNet/DimeNet++, MACE) were import-checked only —
running them needs checkpoints from the eData record.

## Round 1: the earlier script consolidation

All numeric checks in this round used synthetic densities and reference
spectra and a temporary random CNN checkpoint outside the deliverable. None of
those arrays, weights, PNGs or reported test correlations are paper data.

- Parsed all six Python scripts successfully.
- Compared the revised density preprocessing and CNN forward output with the
  uploaded script using identical parameters and input. Values matched.
- Checked 3-D, first-channel and last-channel single-density inputs.
- Ran the complete four-molecule generator with a temporary 2000-bin CNN,
  retaining the leading 900 bins; verified the original energy spacing and
  the 0.4495 a.u. last selected value for a 0.0005 a.u. fixture grid.
- Compared the output reference smoothing, within-window normalisation and
  prediction values with independent numerical calculations.
- Checked arbitrary density filenames and prediction-only two-column output.
- Loaded state-dict bundles, plain state dictionaries with explicit config,
  and a legacy `__main__.CNNModel` full-object checkpoint.
- Ran Figures 3 and 4 from a working directory outside the staged repository.
  Visually inspected their test renders and verified every exported example
  Pearson/MAE against its input arrays.
- Verified Figure 5 is optional, missing requested Figure 5 data gives a clear
  error, and an explicitly supplied synthetic NPZ produces its plot.
- Verified mismatched molecule-ID/SMILES labels and incompatible energy grids
  are rejected.
- Compared all SchNet/DimeNet++ and MACE model-class ASTs against the uploaded
  versions: the model classes are unchanged.
- Exercised the MACE output-writing function separately: common DAT column
  order, and numeric NPZ arrays without an object placeholder for a missing
  reference.

Limits noted at the time, and what became of them:

- No real density/reference pairs, trained checkpoint or original paper DAT
  tables were attached, so the published numbers were not reproduced.
  *Round 2 reproduced every figure from the committed tables; regenerating
  those tables from the density cubes still needs the released dataset.*
- The original training preprocessing code was not attached, so the density
  centring/scaling and reference-smoothing choices could not be confirmed
  against the training pipeline. *Round 2 had the training scripts; the
  preprocessing in the cache builder and in `predict_spectrum_from_density.py`
  agree — divide by maximum absolute amplitude, integer centre-of-mass
  alignment, Gaussian sigma of 10 bins, unit-sum normalisation.*
- The real molecule-ID/SMILES CSV was not attached, so the Figure 4
  inconsistency was resolved relative to the Figure 3 labels. *Round 2 had the
  CSV. Three of the four labels match it; M4170 does not, and the paper label
  is kept deliberately — see
  [FIGURES.md](FIGURES.md#known-label-discrepancy-m4170).*
- PyTorch Geometric / MACE / e3nn environments and trained geometry
  checkpoints were unavailable. *Still true.*
