# Training

All training is cache-first. One script converts the raw dataset into
memory-mappable shards plus an `index.json`; every model family then trains from
that same index, so molecule ordering and the train/validation split are
identical across model families and directly comparable.

Prerequisites: the environment from `environment.yml`, a GPU, and the dataset
downloaded from
[PSDI Data Collections](https://data-collections.psdi.ac.uk/records/pk3n6-s4778)
— see [DATA.md](DATA.md). The examples below assume the layout recommended
there:

```bash
DATA=/your/scratch/qm7-spectra/dataset
CACHE=/your/scratch/qm7-spectra/cache
RUNS=/your/scratch/qm7-spectra/runs
```

## The fixed train/validation split

This is the part to get right if you want numbers comparable with the paper.

- The split is over **positions in the cache index**, not molecule ids. The
  cache builder discovers density files with a sorted glob and writes them in
  that order, so `index.json` fixes the ordering once and for all. Shard
  membership follows directly: sample `i` lives at offset `i % shard_size` of
  shard `i // shard_size`. No randomness is involved in sharding.
- Given that ordering, all six training scripts compute the split the same
  way:

  ```python
  idx = np.arange(n_samples)
  np.random.default_rng(seed).shuffle(idx)
  n_val = int(round(val_frac * n_samples))
  val_idx, train_pool = idx[:n_val], idx[n_val:]
  ```

- **The published runs used `--seed 0` and `--val-frac 0.15`**, which are the
  defaults everywhere. For the 6874-molecule dataset that gives 5843 training
  and 1031 validation molecules. `data/figure_6/indices.json` is the split of
  the published CNN run; you can check any new run against it.
  Two scripts spell the shuffle as `rng.permutation(n)` rather than
  `rng.shuffle(arange(n))`. Those are the same operation in NumPy, so the
  split is identical.
- Size-scaling runs (`--train-fraction`, or `--fractions` for
  `train_mace_spectrum_from_cache.py`) take a leading prefix of the shuffled
  training pool. The validation set never changes, so runs at different
  training-set sizes stay comparable.
- Changing `--seed`, `--val-frac`, or the set of molecules in the cache changes
  which molecules are held out. Do not do it accidentally.

Every training script writes its split to disk (`split.json`, or
`split_seed0_val0.15.json` for `train_mace_spectrum_from_cache.py`) so a run is
always self-describing.

> Before this code was consolidated, `build_density_spectrum_cache.py` drew its
> split with the legacy `np.random.seed(0); np.random.shuffle(...)` API, which
> produces a *different* permutation from `default_rng(0)` and so a different
> held-out set from the one the published CNN was trained against. It now uses
> `default_rng`, matching the other four scripts and reproducing
> `data/figure_6/indices.json` exactly.

## Step 1: build the cache

```bash
python scripts/training/build_density_spectrum_cache.py \
  --root "$DATA" \
  --cache-dir "$CACHE" \
  --prepare-only
```

This writes `$CACHE/index.json` along with `cubes_shard_XXXX.npy` and
`specs_shard_XXXX.npy`. Keep them together. The index records absolute paths, so
if you move the dataset later, either rebuild the cache or pass the
`--spectrum-template` / `--xyz-template` options downstream.

What the builder does to each sample:

- **Density.** Divide by the maximum absolute amplitude, then paste onto a
  common canvas sized to the largest cube in the dataset, aligning the integer
  centre of mass to the canvas centre. Stored as `float16` by default
  (`--cube-dtype float32` to keep full precision).
- **Spectrum.** Clip negatives at zero, interpolate onto the reference energy
  grid if needed, apply a Gaussian of `--spec-sigma` bins (default 10), and
  normalise to unit sum.

Useful options: `--shard-size` (default 128), `--target-points` (default 900,
the model head size).

## Step 2: train

### 3D CNN, density → spectrum

Run the same script without `--prepare-only`. It reuses an existing cache.

```bash
python scripts/training/build_density_spectrum_cache.py \
  --root "$DATA" --cache-dir "$CACHE" \
  --outdir "$RUNS/cnn" \
  --experiments 20 --epochs 200
```

Each experiment samples a random architecture (layers, kernel size, base
filters, dropout) and writes to `$RUNS/cnn/exp_NNN/`. `--train-fraction 0.5`
runs a size-scaling point against the same validation set.

### SchNet and DimeNet++ sweep

```bash
python scripts/training/sweep_schnet_dimenetpp_spectrum_from_cache.py \
  --cache-index "$CACHE/index.json" \
  --xyz-root "$DATA" \
  --outdir "$RUNS/graph" \
  --model both --runs 10 --epochs 100
```

### Four-model graph sweep

SchNet, SchNet with attention pooling, DimeNet++, and a DimeNet++ graph-head
variant:

```bash
python scripts/training/sweep_graph_spectrum_models_from_cache.py \
  --cache-index "$CACHE/index.json" \
  --xyz-root "$DATA" \
  --outdir "$RUNS/graph4" \
  --runs-per-model 5 --epochs 200
```

### MACE, single run or size scaling

```bash
python scripts/training/train_mace_spectrum_from_cache.py \
  --cache-index "$CACHE/index.json" \
  --xyz-root "$DATA" \
  --outdir "$RUNS/mace" \
  --epochs 200

# size scaling against the same validation set
python scripts/training/train_mace_spectrum_from_cache.py \
  --cache-index "$CACHE/index.json" \
  --xyz-root "$DATA" \
  --outdir "$RUNS/mace-size" \
  --fractions 0.25 0.50 1.0 --epochs 200
```

### MACE hyperparameter sweep

```bash
python scripts/training/train_mace_spectrum_sweep_from_cache.py \
  --cache-index "$CACHE/index.json" \
  --xyz-root "$DATA" \
  --outdir "$RUNS/mace-sweep" \
  --target-source cache --epochs 400
```

`--target-source raw` reads the spectrum files recorded in the cache index
instead of the cached spectrum shards.

### Normalization constant, density → scalar

Every spectrum model ends in a softmax, so none of them predicts the overall
scale of a spectrum. This model predicts that scale separately: one number per
molecule, the sum of the first 900 points of the reference spectrum. Multiply
a predicted unit-sum spectrum by it to get an absolute spectrum. It is the
model behind Figure 5.

```bash
python scripts/training/train_density_normconst_from_cache.py \
  --cache-index "$CACHE/index.json" \
  --outdir "$RUNS/normconst"
```

It reuses the spectrum CNN backbone with `out_features=1` and no softmax. The
targets are read once from the `spec_dat` files the cache index records —
last column, clipped at zero, leading `--spec-points` summed — and cached as
`targets_sum900.npz` in the run directory, so a second run starts straight
into training.

The defaults are the published ones: `--num-layers 7 --kernel-size 15
--base-filters 8 --dropout-p 0.2`, L1 loss, AdamW at `5e-5`, 200 epochs,
batch 16, target standardized with the training mean and standard deviation.
`--target-transform log` fits `log(sum)` instead and inverts it for reporting.

Alongside the usual outputs it writes `normconst_model.pt`, a TorchScript
trace of the best checkpoint, and a `config.json` carrying `y_mean`, `y_std`
and the transform. Inference needs that `config.json`: without it the model's
output is in standardized units, not raw ones.

## Run outputs

```text
<outdir>/
  split.json                   the train/validation split actually used
  summary.csv                  one row per experiment
  summary.json                 where the script writes one
  <experiment>/
    config.json                hyperparameters
    train_log.csv              per-epoch losses and metrics
    best_model.pth             state dict at the best validation loss
    val_preds_true.npz         validation predictions at that same epoch
    indices.json               CNN sweep only: train_idx / val_idx
```

The normalization-constant script has no per-experiment subdirectory — it is a
single run, so `config.json`, `split.json`, `train_log.csv`, `best_model.pth`,
`normconst_model.pt`, `targets_sum900.npz` and `val_preds_true_raw.npz` all sit
directly in `<outdir>`. Its split keys are `train_ds_idx` / `val_ds_idx`.

`val_preds_true.npz` holds `pred` and `true` with one row per validation
molecule, **in `val_idx` order** — the validation loader never shuffles. The
CNN sweep also stores `val_idx` in the NPZ itself; for the other scripts, read
it from the run's `split.json` (the SchNet/DimeNet++ sweep additionally stores a
`mid` array of molecule ids). Mapping a row back to a molecule means indexing
`samples[val_idx[i]]["dens_npz"]` in the cache index — exactly what
`scripts/analysis/generate_figure_6_data.py` does.

## Other directory layouts

Do not edit the scripts. Pass templates instead.

Flat dataset directory, all molecules in one folder:

```bash
--density-glob "{root}/density_M*.npz" \
--spectrum-template "{density_dir}/tddft_spectrum_gamma_150meV_M{mid}.dat"
```

Flat or unusual geometry directory:

```bash
--xyz-root /path/to/geometries --xyz-template "{xyz_root}/GEO_M{mid}.xyz"
--xyz-template "{xyz_root}/molecules/{mid}/geometry.xyz"
```

Available fields: `{root}`, `{density_dir}`, `{xyz_root}`, `{mid}`,
`{dens_npz}`, `{spec_dat}`, depending on the option.

## Troubleshooting

**The cache index points at paths that no longer exist.** Rebuild the cache at
the new location, or pass `--spectrum-template` and `--xyz-template` so the
downstream scripts reconstruct paths from molecule ids.

**The cache builder says the cache already exists.** It refuses to overwrite.
Delete `<cache-dir>/index.json` to force a rebuild.

**A new run's validation metrics are not comparable with the paper.** Check
`split.json` against `data/figure_6/indices.json`, and confirm the cache covers
the same 6874 molecules — adding or removing molecules changes every index.
