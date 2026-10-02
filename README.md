# QM7 Absorption Spectra

Machine-learned prediction of molecular absorption spectra for the QM7 molecule
set, from either a ground-state electron density (3D CNN) or a molecular
geometry (SchNet, DimeNet++, MACE). Every model predicts the same target: a
fixed-length, unit-sum absorption spectrum on an energy grid in atomic units.
A separate density CNN predicts the scalar normalization constant that turns
a unit-sum spectrum back into an absolute one.

This repository holds the training code, the inference code, and everything
needed to regenerate the paper figures. Trained model weights and the full
electron-density dataset are too large to host here and are distributed
separately: the weights on
eData,
the dataset on
[PSDI Data Collections](https://data-collections.psdi.ac.uk/records/pk3n6-s4778).
See [docs/DATA.md](docs/DATA.md).

## Environment

Everything runs in one conda environment, `spectra-ml`, pinned in
`environment.yml`:

```bash
conda env create -f environment.yml
conda activate spectra-ml
```

That is the environment the published models were trained and evaluated in:
Python 3.10, PyTorch 2.8.0, PyTorch Geometric 2.6.1, mace-torch 0.3.15,
e3nn 0.4.4, RDKit 2025.3.5, NumPy 1.23.5, SciPy 1.11.4, scikit-learn 1.3.2,
Matplotlib 3.10.8. Check it resolved:

```bash
python -c "import torch, numpy, matplotlib, rdkit; print(torch.__version__, torch.cuda.is_available())"
```

**CUDA.** The pinned PyTorch is the CUDA 12.6 build, and `environment.yml`
carries the two pip index lines needed to fetch it and the matching
`torch-cluster` wheel. For a different CUDA version, or for CPU only, edit
those two lines and the `torch*`, `torch-cluster` pins to match — see the
[PyTorch](https://pytorch.org/get-started/locally/) and
[PyG](https://data.pyg.org/whl/) wheel indexes.

**You do not need all of it.** Plotting the figures needs only NumPy,
Matplotlib, RDKit and scikit-learn, none of which need a GPU. PyTorch is
needed for any inference, PyTorch Geometric for the SchNet/DimeNet++ models,
and `mace-torch` plus e3nn for MACE. Training the density CNN needs a GPU in
practice; the other scripts will run on CPU, slowly.

Installing the full environment anyway is the safest route to reproducing
published numbers, since results can shift with library versions.

## Quick start: reproduce the figures

The figures are built from small derived tables that are committed here. They
need neither the model weights nor the density dataset:

```bash
bash scripts/reproduce_figures.sh
```

This overwrites `figures/figure_3.png` … `figure_7.png`. To check reproduction
without touching the committed images, write elsewhere and compare:

```bash
bash scripts/reproduce_figures.sh /tmp/check
cmp /tmp/check/figure_3.png figures/figure_3.png
```

On the pinned environment in `environment.yml` all five figures come out
byte-identical to the committed PNGs. [docs/FIGURES.md](docs/FIGURES.md)
explains each figure, its inputs, and how to regenerate those inputs from the
weights and dataset.

## Repository layout

```text
scripts/
  training/     model training and the shared density/spectrum cache builder
  inference/    predict one spectrum from a density cube or an XYZ geometry
  analysis/     turn model outputs into the small tables the figures plot
  plotting/     draw the figures from those tables
  reproduce_figures.sh
data/
  figures_3_4/  the four CNN prediction tables behind Figures 3 and 4
  figure_5/     validation normalisation constants behind Figure 5
  figure_6/     validation predictions and molecule metadata for Figures 6 and 7
  plotting/     the chemistry summary CSV that Figures 6 and 7 plot
figures/        the published PNGs and the example-metrics CSV
docs/           documentation
environment.yml pinned conda environment
```

## Documentation

| Document | Contents |
|---|---|
| [docs/DATA.md](docs/DATA.md) | The two released records, and exactly where each downloaded file goes |
| [docs/TRAINING.md](docs/TRAINING.md) | Cache building, the four training scripts, and the fixed train/validation split |
| [docs/INFERENCE.md](docs/INFERENCE.md) | Predicting a spectrum with each model family, and the table format |
| [docs/FIGURES.md](docs/FIGURES.md) | Figure-by-figure reproduction, from committed tables and from scratch |
| [docs/VALIDATION.md](docs/VALIDATION.md) | Checks performed when this code was consolidated, and their limits |

## Conventions that hold everywhere

- **Target.** A non-negative spectrum of 900 bins on a 0.0005 a.u. grid running
  from 0.0 to 0.4495 a.u., normalised to unit sum. Every spectrum model ends in
  a softmax, so its output is already unit-sum. The absolute scale is predicted
  separately — see
  [docs/TRAINING.md](docs/TRAINING.md#normalization-constant-density--scalar).
- **Molecule identity.** Files are linked by the molecule id in their name:
  `GEO_M<id>.xyz`, `density_M<id>.npz`, `tddft_spectrum_gamma_150meV_M<id>.dat`.
- **Molecule ordering.** The cache `index.json` produced by
  `scripts/training/build_density_spectrum_cache.py` fixes the ordering for
  every model family. Positions in that index are what the splits refer to.
- **Split.** `--seed 0` and `--val-frac 0.15` define the published split. All
  six training scripts derive it identically; see
  [docs/TRAINING.md](docs/TRAINING.md#the-fixed-trainvalidation-split). Do not
  change the seed if you want results comparable with the paper.
- **Prediction tables.** `energy_au reference prediction` with a reference, or
  `energy_au prediction` without one, plus `#` comment lines carrying metadata.

## Citation and license

Copyright (c) 2026 stfc-ai4s.

## Citation and license

If you use this work, please cite the associated paper.

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

If you use this code, the released weights or the dataset, please cite the
accompanying paper and the records listed in [docs/DATA.md](docs/DATA.md).
