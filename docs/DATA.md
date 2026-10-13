# Data and model weights

Two things are deliberately **not** in this repository, and they live in two
separate records:

| What | Where |
|---|---|
| **Trained model weights** (`.pt` / `.pth`) | eData (STFC): <https://doi.org/10.5286/edata/987> |
| **The electron-density and geometry dataset** — roughly 6 GB of density cubes, TDDFT reference spectra and geometries | PSDI Data Collections: <https://data-collections.psdi.ac.uk/records/pk3n6-s4778> |

`.gitignore` excludes `*.pt`, `*.pth` and the dataset directories, so neither
can be committed by accident.

Nothing on either record is needed to redraw the paper figures — see
[FIGURES.md](FIGURES.md). You need a download only to train a model, to run
inference on a new molecule, or to regenerate the figure input tables from
scratch. Training needs both records; inference on a molecule you already have
needs only the weights.

## Weights

| File | Model | Used by |
|---|---|---|
| `cnn_model_spectra.pt` | 3D CNN, density → spectrum | `predict_spectrum_from_density.py`, `generate_figures_3_4_data.py` |
| `schnet_model_spectra.pt` | SchNet, geometry → spectrum | `predict_spectrum_from_xyz_schnet.py` |
| `dimnetpp_model_spectra.pt` | DimeNet++, geometry → spectrum | `predict_spectrum_from_xyz_schnet.py` |
| `mace_model_spectra.pth` | MACE, geometry → spectrum | `predict_mace_spectrum.py` |
| `cnn_model_normconst.pt` | 3D CNN, density → normalization constant | `predict_normconst_from_density.py` |

These are the file names on the eData record. Note the spelling `dimnetpp`.

`cnn_model_normconst.pt` is a TorchScript export. The predictor also accepts the
raw `best_model.pth` state dict, but either way it needs the training run's
`config.json`, which carries the target transform and the standardization
constants without which the output is in arbitrary units. That file is small,
so it is committed here as `configs/normconst_config.json` rather than shipped
with the weights, together with the run's train/validation split
(`configs/normconst_split.json`) and its targets
(`configs/normconst_targets_sum900.npz`), which the predictor's `validation`
mode uses to regenerate Figure 5.

The CNN checkpoint behind Figures 3 and 4 is the `exp_003` model of the
900-bin CNN sweep.

Checkpoint formats accepted by the predictors are described in
[INFERENCE.md](INFERENCE.md#checkpoint-formats).

## Dataset

From the PSDI Data Collections record. One directory per QM7 molecule id:

```text
<id>/
  density_M<id>.cube                         ground-state electron density (Gaussian cube)
  elf_M<id>.cube                             electron localization function (not used)
  GEO_M<id>.xyz                              Cartesian geometry
  tddft_spectrum_gamma_150meV_M<id>.dat      TDDFT reference spectrum — the training target
  tddft_spectrum_gamma_{50,500}meV_M<id>.dat same spectrum at other broadenings
  ks_spectrum_gamma_{50,150,500}meV_M<id>.dat Kohn–Sham spectra (not used)
  tddft_spectrum_list_<id>.dat               excitation list
```

Every model here was trained on the **150 meV TDDFT** spectra. A spectrum
`.dat` has an energy column in atomic units followed by `<xx> <yy> <zz> <f>`
columns.

### Convert the density cubes once

The density predictors and the cache builder read `density_M<id>.npz`, not the
cube. Convert every cube in the dataset in place (each NPZ is written beside
its cube; about 4 MB per molecule):

```bash
python scripts/convert_cubes_to_npz.py /your/scratch/qm7-spectra/dataset
```

The NPZ stores the volume under the key `data` as an `(nx, ny, nz)` float32
array in cube order. Already-converted molecules are skipped, so the command
can be re-run after a partial download.

## Where to put the download

Keep both downloads **outside the repository checkout**. Every script takes
the location as a command-line argument, so no path is hard-coded and nothing
large can drift into git. (`weights/` and `datasets/` inside the checkout are
also gitignored, if you prefer to keep them there.) The two records unpack side
by side; a layout that works with all the default templates:

```text
/your/scratch/qm7-spectra/
  weights/                                   from the eData record
    cnn_model_spectra.pt
    schnet_model_spectra.pt
    dimnetpp_model_spectra.pt
    mace_model_spectra.pth
    cnn_model_normconst.pt
  dataset/                                   from the PSDI record
    1/
      density_M1.cube
      density_M1.npz                         written by convert_cubes_to_npz.py
      tddft_spectrum_gamma_150meV_M1.dat
      GEO_M1.xyz
      ...
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
