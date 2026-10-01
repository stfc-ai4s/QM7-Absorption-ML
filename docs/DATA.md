# Data and model weights

Two things are deliberately **not** in this repository, and they live in two
separate records:

| What | Where |
|---|---|
| **Trained model weights** (`.pt` / `.pth`) | eData (STFC): <https://edata.stfc.ac.uk/items/2d60d176-fe1f-4f16-a59e-6487224230be> |
| **The electron-density and geometry dataset** — roughly 6 GB of density cubes, TDDFT reference spectra and geometries | PSDI Data Collections: <https://data-collections.psdi.ac.uk/records/pk3n6-s4778> |

`.gitignore` excludes `*.pt`, `*.pth` and the dataset directories, so neither
can be committed by accident.

> **Before publishing:** add both records' DOIs here in place of
> `<EDATA_DOI>` and `<PSDI_DOI>`, for the citation in the README.

Nothing on either record is needed to redraw the paper figures — see
[FIGURES.md](FIGURES.md). You need a download only to train a model, to run
inference on a new molecule, or to regenerate the figure input tables from
scratch. Training needs both records; inference on a molecule you already have
needs only the weights.

## Weights

| File | Model | Used by |
|---|---|---|
| `cnn_density_spectrum.pt` | 3D CNN, density → spectrum | `predict_spectrum_from_density.py`, `generate_figures_3_4_data.py` |
| `schnet_spectrum.pt` | SchNet, geometry → spectrum | `predict_spectrum_from_xyz_schnet.py` |
| `dimenetpp_spectrum.pt` | DimeNet++, geometry → spectrum | `predict_spectrum_from_xyz_schnet.py` |
| `mace_spectrum.pth` | MACE, geometry → spectrum | `predict_mace_spectrum.py` |
| `normconst_model.pt` | 3D CNN, density → normalization constant | `predict_normconst_from_density.py` |

`normconst_model.pt` is a TorchScript export. The predictor also accepts the
raw `best_model.pth` state dict, but either way it needs the run's
`config.json` beside it — that file carries the target transform and the
standardization constants without which the output is in arbitrary units.
Keep `config.json`, `split.json` and `targets_sum900.npz` together with the
checkpoint.

The CNN checkpoint behind Figures 3 and 4 is the `exp_003` model of the
900-bin CNN sweep. `data/figures_3_4/prediction_run.json` records its SHA-256,
so you can confirm you have the same file:

```bash
sha256sum /path/to/weights/cnn_density_spectrum.pt
python -c "import json;print(json.load(open('data/figures_3_4/prediction_run.json'))['model_sha256'])"
```

Checkpoint formats accepted by the predictors are described in
[INFERENCE.md](INFERENCE.md#checkpoint-formats).

## Dataset

From the PSDI Data Collections record. One directory per QM7 molecule id,
holding up to three files:

```text
<id>/
  density_M<id>.npz                          ground-state electron density
  tddft_spectrum_gamma_150meV_M<id>.dat      TDDFT reference spectrum
  GEO_M<id>.xyz                              Cartesian geometry
```

The density NPZ stores the volume under the key `data`, with shape
`(nx, ny, nz)` or `(nx, ny, nz, 1)`. The spectrum `.dat` holds an energy column
in atomic units followed by an intensity column.

## Where to put the download

Keep both downloads **outside the repository checkout**. Every script takes
the location as a command-line argument, so no path is hard-coded and nothing
large can drift into git. The two records unpack side by side; a layout that
works with all the default templates:

```text
/your/scratch/qm7-spectra/
  weights/                                   from the eData record
    cnn_density_spectrum.pt
    schnet_spectrum.pt
    dimenetpp_spectrum.pt
    mace_spectrum.pth
    normconst_model.pt
    normconst_config.json                    keep beside normconst_model.pt
  dataset/                                   from the PSDI record
    1/
      density_M1.npz
      tddft_spectrum_gamma_150meV_M1.dat
      GEO_M1.xyz
    2/
      ...
  cache/                                     created by the cache builder
```

With that layout the defaults line up as:

| Script argument | Value |
|---|---|
| `--root` (cache builder) | `/your/scratch/qm7-spectra/dataset` |
| `--cache-dir` | `/your/scratch/qm7-spectra/cache` |
| `--cache-index` | `/your/scratch/qm7-spectra/cache/index.json` |
| `--xyz-root` | `/your/scratch/qm7-spectra/dataset` |
| `--model` / `--weights` | a file under `/your/scratch/qm7-spectra/weights/` |

If your densities, spectra and geometries are laid out differently, do not edit
the code — pass `--density-glob`, `--spectrum-template` and `--xyz-template`.
See [TRAINING.md](TRAINING.md#other-directory-layouts).

## The one exception: Figure 3 and 4 inputs

Regenerating the four Figure 3/4 prediction tables reads eight files from
`data/figures_3_4/inputs/<id>/` by default. That path is gitignored, so you can
copy the eight files in without risking a commit:

```bash
for mid in 481 972 3935 4170; do
    mkdir -p "data/figures_3_4/inputs/${mid}"
    cp /your/scratch/qm7-spectra/dataset/${mid}/density_M${mid}.npz \
       /your/scratch/qm7-spectra/dataset/${mid}/tddft_spectrum_gamma_150meV_M${mid}.dat \
       "data/figures_3_4/inputs/${mid}/"
done
```

Or skip the copy and point the generator straight at the dataset with
`--dataset-root /your/scratch/qm7-spectra/dataset`.

## What is committed, and why

| Path | Size | Why it is here |
|---|---|---|
| `data/figures_3_4/*.dat` | ~200 KB | The four prediction tables Figures 3 and 4 plot |
| `data/figure_5/val_preds_true_raw.npz` | ~20 KB | Validation normalisation constants for Figure 5 |
| `data/figure_6/val_preds_true.npz` | ~7 MB | Validation predictions for the Figure 6/7 analysis |
| `data/figure_6/cache_index.json` | ~3 MB | Maps validation rows back to molecule ids |
| `data/figure_6/xyz/` | ~54 MB | Geometries used for element-presence flags |
| `data/plotting/figure_6/*.csv` | ~2 KB | The chemistry summary Figures 6 and 7 plot |

Together these make every figure reproducible with no download at all. The
density cubes, which dominate the dataset size, are never needed for plotting.

`data/figure_6/cache_index.json` records the absolute paths of the machine the
cache was built on. Those paths are never opened — only the molecule id is
parsed out of them — so they are harmless and are kept unchanged so the index
still matches the run that produced it.
