# Inference

Three spectrum predictors, one per input type, all writing the same table
format:

| Script | Input | Weights flag |
|---|---|---|
| `scripts/inference/predict_spectrum_from_density.py` | density `.npz` | `--model` |
| `scripts/inference/predict_spectrum_from_xyz_schnet.py` | geometry `.xyz` | `--model` |
| `scripts/inference/predict_mace_spectrum.py` | geometry `.xyz` | `--weights` |

Plus one scalar predictor, which supplies the overall scale those models do
not predict:

| Script | Input | Output |
|---|---|---|
| `scripts/inference/predict_normconst_from_density.py` | density `.npz` | one normalization constant |

`scripts/inference/spectrum_io.py` is shared by the three spectrum predictors
and must stay beside them. Download the checkpoints from the
[eData record](https://edata.stfc.ac.uk/items/2d60d176-fe1f-4f16-a59e-6487224230be)
and keep them outside the checkout; see [DATA.md](DATA.md).

The examples below write into `predictions/`, which is gitignored, so your own
inference output never lands in a commit.

## The output table

```text
# energy_au reference prediction
# {"model": "...", "selected_bins": 900, ...}
0.000000000000e+00 1.494961781072e-05 1.127144206314e-05
...
```

- With a reference: `energy_au reference prediction`. Without one:
  `energy_au prediction`. Column order is fixed; do not mix these tables with
  older files that used `energy prediction reference`.
- The prediction is the model's **leading `--length` bins** (default 900),
  renormalised to unit sum within that window. A longer model output is cropped,
  never stretched onto a different energy range.
- Energy values are never inferred from the number of bins. They come from the
  reference's first column, from `--energy-grid`, or from an explicit
  `--energy-step-au` counting up from zero. If both a reference and a grid are
  given and they disagree, that is an error.
- A 900-bin grid at 0.0005 a.u. spacing ends at **0.4495 a.u.**, not 0.45. The
  scripts keep the true endpoint.

Any committed paper table works as an energy grid for a model trained on the
same target:

```bash
--energy-grid data/figures_3_4/spectrum_prediction_M481.dat
```

## Predict from an electron density

```bash
python scripts/inference/predict_spectrum_from_density.py \
    --model /your/scratch/qm7-spectra/weights/cnn_density_spectrum.pt \
    --npz  /your/scratch/qm7-spectra/dataset/1/density_M1.npz \
    --ref  /your/scratch/qm7-spectra/dataset/1/tddft_spectrum_gamma_150meV_M1.dat \
    --outdir predictions/density_M1
```

`--ref` is optional. If it is omitted and the density is named
`density_M<id>.npz`, the script looks for `tddft_spectrum_gamma_150meV_M<id>.dat`
beside it. Without any reference, supply `--energy-grid` or `--energy-step-au`.

The filename does not have to follow the QM7 convention. The density NPZ must
carry the volume under the key `data`, as a 3-D array or a 4-D array with a
single leading or trailing channel.

Preprocessing matches training: divide by the maximum absolute amplitude, then
align the integer centre of mass within the model's input grid, padding or
cropping as needed. The grid comes from the checkpoint's `input_shape` when it
records one, otherwise `(1, 155, 147, 143)`. The model is not invariant to
different voxel spacings or orientations — the density must be sampled the same
way the training densities were.

Reference handling: column 1 by default (`--reference-column`), smoothed with a
Gaussian of 10 bins before truncation (`--reference-sigma`, `0` disables it).
These defaults reproduce the committed Figure 3/4 tables.

## Predict from a geometry with SchNet or DimeNet++

```bash
python scripts/inference/predict_spectrum_from_xyz_schnet.py \
    --model /your/scratch/qm7-spectra/weights/schnet_spectrum.pt \
    --xyz   /your/scratch/qm7-spectra/dataset/1/GEO_M1.xyz \
    --reference /your/scratch/qm7-spectra/dataset/1/tddft_spectrum_gamma_150meV_M1.dat \
    --outdir predictions/schnet_M1
```

Writes the table, a metadata JSON and a quick PNG (`--no-plot` to skip it).
The reference is optional; without it, pass `--energy-grid` or
`--energy-step-au`. The reference intensity column defaults to the **last**
column here, and no smoothing is applied — both are the conventions these
models were trained and evaluated with.

## Predict from a geometry with MACE

```bash
python scripts/inference/predict_mace_spectrum.py \
    --weights /your/scratch/qm7-spectra/weights/mace_spectrum.pth \
    --xyz     /your/scratch/qm7-spectra/dataset/1/GEO_M1.xyz \
    --reference /your/scratch/qm7-spectra/dataset/1/tddft_spectrum_gamma_150meV_M1.dat \
    --outdir predictions/mace_M1
```

Output directory contents: `prediction_spectrum.dat`,
`prediction_spectrum.npz`, `prediction_metadata.json`, and
`prediction_spectrum.png` when matplotlib is available. With a reference, the
metadata also carries MAE and Pearson correlation.

`--out-dim` is the **trained** head size; `--length` is the evaluation window.
They differ only if the checkpoint predicts more bins than you want to score.
The script reads `core.atomic_numbers` out of the checkpoint to rebuild the
z-table; pass `--atomic-numbers 1,6,7,8,16,17` if the checkpoint lacks it.

## Predict the normalization constant

Every spectrum model ends in a softmax, so its output is unit-sum and carries
no information about how intense the spectrum actually is. This model predicts
that missing scale — the sum of the first 900 points of the reference spectrum
— from the same density cube:

```bash
python scripts/inference/predict_normconst_from_density.py density \
    --model   /your/scratch/qm7-spectra/weights/normconst_model.pt \
    --config  /your/scratch/qm7-spectra/weights/normconst_config.json \
    --density /your/scratch/qm7-spectra/dataset/1/density_M1.npz
```

Multiply a unit-sum predicted spectrum by the printed constant to get an
absolute one. `--output result.npz` saves it instead of only printing it.

**`--config` is not optional.** The model was trained on a standardized
target, so `config.json` — written by the training run — is what converts the
raw output back into spectrum-sum units. Without the matching config the
number is meaningless.

**Preprocessing is your responsibility in this mode.** The density must be
prepared exactly as the cache builder prepares one: divided by its maximum
absolute amplitude, then centre-of-mass aligned onto the `cube_shape` recorded
in `config.json`. Shapes `(X,Y,Z)`, `(C,X,Y,Z)` and `(1,C,X,Y,Z)` are all
accepted, but a mismatch against `cube_shape` is an error rather than a
silent resize.

The second mode scores a whole validation set straight from the cache shards,
where the densities are already prepared:

```bash
python scripts/inference/predict_normconst_from_density.py validation \
    --model   /your/scratch/qm7-spectra/weights/normconst_model.pt \
    --config  /your/scratch/qm7-spectra/runs/normconst/config.json \
    --split   /your/scratch/qm7-spectra/runs/normconst/split.json \
    --targets /your/scratch/qm7-spectra/runs/normconst/targets_sum900.npz \
    --output  /tmp/val_preds_true_raw.npz
```

It prints MAE, RMSE and Pearson correlation in raw units and writes `pred`,
`true` and `val_ds_idx`. That file is exactly what Figure 5 plots, so it can
be passed to `make_figures_3_4_5.py --figures 5 --normalization-npz`.

## Checkpoint formats

**Density CNN** accepts three forms:

1. A full pickled `CNNModel` (`torch.save(model)`), including legacy ones saved
   under `__main__`.
2. A bundle dict containing `model_kwargs` and `state_dict`.
3. A plain state dict, which then needs `--config config.json` giving
   `num_layers`, `kernel_size`, `base_filters`, `dropout_p`, `input_shape` and
   `out_features`. The architecture is never guessed from defaults.

**SchNet / DimeNet++** accepts a full saved `.pt` model or a portable
checkpoint bundle. **MACE** takes a `best_model.pth` state dict; the script
matches the architecture against the checkpoint's tensor shapes.

**Normalization-constant CNN** accepts a TorchScript export
(`normconst_model.pt`) or the plain `best_model.pth` state dict. For a state
dict the architecture is rebuilt from the `arch` and `cube_shape` entries of
`config.json`, never from defaults.

Full-model pickles execute code on load. Only load checkpoints you produced or
obtained from the eData record.

## Environment

Plotting needs NumPy, Matplotlib and RDKit. Density inference adds PyTorch and
SciPy. SchNet/DimeNet++ inference needs PyTorch Geometric; MACE inference needs
`mace-torch` and e3nn. `environment.yml` pins a single environment with all of
them. Use the versions the model was trained with — `environment.yml` is that
environment.
