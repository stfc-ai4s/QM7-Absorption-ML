#!/usr/bin/env python3
"""Prepare sharded density/spectrum caches and optionally run a CNN sweep.

This is the cache-producing script for the density-to-spectrum experiments. It
finds pairs of ground-state electron-density cubes and absorption spectra,
centres/pads the density on a common 3D canvas, writes memory-mappable shards,
and stores an ``index.json`` file. Downstream graph and MACE scripts should be
given this same index so that the molecule ordering and train/validation split
remain consistent across models.

Expected input layout, using the default templates:

    <data-root>/
      <molecule-id>/
        density_M<molecule-id>.npz
        tddft_spectrum_gamma_150meV_M<molecule-id>.dat

The density NPZ must contain a ``data`` array with shape ``(nx, ny, nz)`` or
``(nx, ny, nz, channels)``. The spectrum file may contain either one intensity
column or a grid column followed by an intensity column.

For a different layout, keep the code unchanged and pass templates. Available
fields are ``{root}``, ``{density_dir}``, and ``{mid}``. For example:

    --density-glob "{root}/molecules/*/density_M*.npz"
    --spectrum-template "{density_dir}/spectra/tddft_spectrum_gamma_150meV_M{mid}.dat"

Prepare only the reusable cache:

    python scripts/training/build_density_spectrum_cache.py \
      --root /path/to/data-root \
      --cache-dir /path/to/cache-dir \
      --prepare-only

Prepare the cache, then run the CNN sweep:

    python scripts/training/build_density_spectrum_cache.py \
      --root /path/to/data-root \
      --cache-dir /path/to/cache-dir \
      --outdir /path/to/cnn-runs \
      --experiments 20 \
      --epochs 200

The cache index written at ``<cache-dir>/index.json`` is the file to pass as
``--cache-index`` to the graph and MACE training scripts. If the data are moved,
rebuild the cache or make sure the paths recorded in the index still resolve.

Determinism
-----------
Shard membership is positional: pairs are discovered with a sorted glob, then
written in that order, so ``index.json`` fixes the molecule ordering for every
model family. The train/validation split is drawn from that ordering by
``build_fixed_split`` with ``numpy.random.default_rng(--seed)``, matching the
other training scripts and the published runs. ``--seed 0`` and
``--val-frac 0.15`` reproduce the paper split; see docs/TRAINING.md.
"""

import argparse
import os
import re
import json
import math
import random
import datetime
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scipy.ndimage import gaussian_filter1d

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
# -----------------------------
# Defaults (override via CLI)
# -----------------------------
DEFAULT_CACHE_DIRNAME = "_cache_density_spec_shards"
DEFAULT_DENSITY_GLOB = "{root}/*/density_M*.npz"
DEFAULT_SPECTRUM_TEMPLATE = "{density_dir}/tddft_spectrum_gamma_150meV_M{mid}.dat"

PROJECT_SAVE_ROOT = Path("density_spec_npz_runs")
NUM_EXPERIMENTS = 50
EPOCHS = 200
BATCH_SIZE = 8
LR = 5e-5
WEIGHT_DECAY = 1e-6
LOSS_TYPE = "kl"
RNG_SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_TARGET_POINTS = 900

# -----------------------------
# Pair discovery
# -----------------------------
_DENS_RE = re.compile(r"density_M(\d+)\.npz$")

def find_pairs(root: str, density_glob: str, spectrum_template: str):
    root_path = Path(root)
    dens_files = sorted(glob_glob(density_glob.format(root=root_path)))
    pairs = []
    missing = 0
    for dens in dens_files:
        m = _DENS_RE.search(dens)
        if not m:
            continue
        mid = m.group(1)
        d = os.path.dirname(dens)
        spec = spectrum_template.format(root=root_path, density_dir=d, mid=mid)
        if os.path.exists(spec):
            pairs.append((dens, spec))
        else:
            missing += 1
    if missing:
        print(f"[WARN] Missing spectra for {missing} density files.")
    return pairs

def glob_glob(pat):
    import glob
    return glob.glob(pat)


# -----------------------------
# Center/paste utilities (fast COM)
# -----------------------------
def center_of_mass_ijk(w: np.ndarray) -> np.ndarray:
    # w: (nx, ny, nz), non-negative
    wsum = float(w.sum())
    nx, ny, nz = w.shape
    if wsum <= 0.0:
        return np.array([(nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0], dtype=np.float64)

    wi = w.sum(axis=(1, 2))
    wj = w.sum(axis=(0, 2))
    wk = w.sum(axis=(0, 1))

    i = np.arange(nx, dtype=np.float64)
    j = np.arange(ny, dtype=np.float64)
    k = np.arange(nz, dtype=np.float64)

    ci = float((wi * i).sum() / wsum)
    cj = float((wj * j).sum() / wsum)
    ck = float((wk * k).sum() / wsum)
    return np.array([ci, cj, ck], dtype=np.float64)

def paste_centered_3d(dest: np.ndarray, src: np.ndarray, com_src: np.ndarray):
    """
    dest: (X,Y,Z)
    src : (x,y,z)
    com_src: (3,) float indices in src
    Places src into dest such that src COM aligns to dest geometric center.
    """
    X, Y, Z = dest.shape
    x0, y0, z0 = src.shape

    dest_center = np.array([(X - 1) / 2.0, (Y - 1) / 2.0, (Z - 1) / 2.0], dtype=np.float64)
    start = np.round(dest_center - com_src).astype(int)  # (sx,sy,sz) in dest

    # Destination overlap
    dst_start = np.maximum(start, 0)
    dst_end = np.minimum(start + np.array([x0, y0, z0]), np.array([X, Y, Z]))

    if np.any(dst_end - dst_start <= 0):
        return

    # Corresponding source overlap
    src_start = np.maximum(0, -start)
    src_end = src_start + (dst_end - dst_start)

    xs, ys, zs = src_start
    xe, ye, ze = src_end
    xd, yd, zd = dst_start
    xde, yde, zde = dst_end

    dest[xd:xde, yd:yde, zd:zde] = src[xs:xe, ys:ye, zs:ze]


# -----------------------------
# Cache builder (sharded .npy, memory-mappable)
# -----------------------------
def load_density_npz(path: str) -> np.ndarray:
    """
    Returns density as (nx,ny,nz) float32.
    Handles stored data as (nx,ny,nz) or (nx,ny,nz,1) or (nx,ny,nz,C) by taking first channel.
    """
    with np.load(path, allow_pickle=False) as z:
        if "data" not in z.files:
            raise KeyError(f"{path} has no 'data' key.")
        data = z["data"]
        if data.ndim == 3:
            vol = data
        elif data.ndim == 4:
            vol = data[..., 0]  # single channel expected; take first if multiple
        else:
            raise ValueError(f"Unexpected density shape {data.shape} in {path}")
        return vol.astype(np.float32, copy=False)

def load_spectrum_dat(path: str):
    """
    Returns (x, y) where x may be None if file is 1-col.
    """
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        return None, arr.astype(np.float32, copy=False)
    if arr.shape[1] < 2:
        return None, arr[:, 0].astype(np.float32, copy=False)
    return arr[:, 0].astype(np.float32, copy=False), arr[:, 1].astype(np.float32, copy=False)

def build_cache_shards(
    pairs,
    cache_dir: Path,
    shard_size: int = 128,
    cube_dtype: str = "float16",
    spec_sigma: float = 10.0,
    check_voxel_vectors: bool = False,
    target_points: int = DEFAULT_TARGET_POINTS,
):
    """
    Produces:
      cache_dir/index.json
      cache_dir/cubes_shard_XXXX.npy  shape (S, 1, X, Y, Z)
      cache_dir/specs_shard_XXXX.npy  shape (S, L)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.json"
    if index_path.exists():
        print(f"[INFO] Cache exists: {index_path} (delete it to rebuild)")
        return index_path

    # Pass 1: find max cube shape + reference spectrum grid/length
    max_nx = max_ny = max_nz = 0
    x_ref = None
    L_ref = None

    # optional voxel vector checks if present in npz
    ref_xvec = ref_yvec = ref_zvec = None

    for dens, spec in pairs:
        with np.load(dens, allow_pickle=False) as z:
            data = z["data"]
            nx, ny, nz = data.shape[:3]
            max_nx = max(max_nx, nx)
            max_ny = max(max_ny, ny)
            max_nz = max(max_nz, nz)

            if check_voxel_vectors and all(k in z.files for k in ("xvec", "yvec", "zvec")):
                xvec = np.array(z["xvec"], dtype=np.float64)
                yvec = np.array(z["yvec"], dtype=np.float64)
                zvec = np.array(z["zvec"], dtype=np.float64)
                if ref_xvec is None:
                    ref_xvec, ref_yvec, ref_zvec = xvec, yvec, zvec
                else:
                    if not (np.allclose(xvec, ref_xvec, atol=1e-6, rtol=0) and
                            np.allclose(yvec, ref_yvec, atol=1e-6, rtol=0) and
                            np.allclose(zvec, ref_zvec, atol=1e-6, rtol=0)):
                        raise ValueError(f"Voxel vectors differ: {dens}")

        x, y = load_spectrum_dat(spec)
        if L_ref is None:
            L_ref = int(y.shape[0])
            x_ref = x.copy() if x is not None else None
        else:
            # we will interpolate later if needed; just track reference length
            pass

    X, Y, Z = int(max_nx), int(max_ny), int(max_nz)
    print(f"[INFO] Target cube canvas: (1, {X}, {Y}, {Z})")

    # Pass 2: write shards
    samples = []
    n = len(pairs)
    n_shards = (n + shard_size - 1) // shard_size

    # allocate reusable buffers
    dest3 = np.zeros((X, Y, Z), dtype=np.float32)

    for shard_id in range(n_shards):
        s0 = shard_id * shard_size
        s1 = min(n, (shard_id + 1) * shard_size)
        S = s1 - s0

        cube_shard_path = cache_dir / f"cubes_shard_{shard_id:04d}.npy"
        spec_shard_path = cache_dir / f"specs_shard_{shard_id:04d}.npy"

        cubes_mm = open_memmap(
            cube_shard_path, mode="w+", dtype=cube_dtype, shape=(S, 1, X, Y, Z)
        )
        specs_mm = open_memmap(
            spec_shard_path, mode="w+", dtype="float32", shape=(S, int(L_ref))
        )

        for j, (dens, spec) in enumerate(pairs[s0:s1]):
            # density -> centered/padded
            vol = load_density_npz(dens)  # (nx,ny,nz)
            # per-sample normalisation (stable + no global scan)
            vmax = float(np.max(np.abs(vol)))
            if vmax > 0:
                vol = vol / vmax

            w = np.abs(vol)
            com = center_of_mass_ijk(w)

            dest3.fill(0.0)
            paste_centered_3d(dest3, vol, com_src=com)
            cubes_mm[j, 0, :, :, :] = dest3.astype(cube_dtype, copy=False)

            # spectrum -> common grid/length + smooth + normalise-to-sum1
            x, y = load_spectrum_dat(spec)
            y = np.clip(y, 0.0, None)

            if x_ref is not None and x is not None:
                if y.shape[0] != x_ref.shape[0] or not np.allclose(x, x_ref, atol=1e-6, rtol=0):
                    y = np.interp(x_ref, x, y).astype(np.float32, copy=False)
            else:
                if y.shape[0] != int(L_ref):
                    # interpolate by index
                    t_old = np.linspace(0.0, 1.0, y.shape[0], dtype=np.float32)
                    t_new = np.linspace(0.0, 1.0, int(L_ref), dtype=np.float32)
                    y = np.interp(t_new, t_old, y).astype(np.float32, copy=False)

            if spec_sigma and spec_sigma > 0:
                y = gaussian_filter1d(y, sigma=float(spec_sigma)).astype(np.float32, copy=False)

            s = float(y.sum())
            if s > 0:
                y = y / s

            specs_mm[j, :] = y

            samples.append({
                "dens_npz": dens,
                "spec_dat": spec,
                "cube_shard": str(cube_shard_path),
                "spec_shard": str(spec_shard_path),
                "offset": int(j),
            })

        # flush shard
        del cubes_mm
        del specs_mm
        print(f"[INFO] Wrote shard {shard_id+1}/{n_shards}: {cube_shard_path.name}, {spec_shard_path.name}")

    meta = {
        "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "root": str(cache_dir.resolve()),
        "num_samples": int(n),
        "shard_size": int(shard_size),
        "cube_shape": [1, X, Y, Z],
        "cube_dtype": str(cube_dtype),
        "spec_len": int(L_ref),
        "spec_sigma": float(spec_sigma),
        "target_points": int(target_points),
        "x_ref_present": bool(x_ref is not None),
        "samples": samples,
    }

    index_path.write_text(json.dumps(meta, indent=2))
    print(f"[OK] Wrote index: {index_path}")
    return index_path


# -----------------------------
# Dataset: opens shard mmaps lazily with a tiny per-worker cache
# -----------------------------
class ShardedMemmapDataset(Dataset):
    def __init__(self, index_json: str, target_points: int = DEFAULT_TARGET_POINTS):
        idx = json.loads(Path(index_json).read_text())
        self.samples = idx["samples"]
        self.cube_shape = tuple(idx["cube_shape"])
        self.spec_len = int(target_points)
        self._mm_cache = {}

    def __len__(self):
        return len(self.samples)

    def _load_shard(self, cube_path: str, spec_path: str):
        key = (cube_path, spec_path)
        mm = self._mm_cache.get(key)
        if mm is None:
            cubes = np.load(cube_path, mmap_mode="r")
            specs = np.load(spec_path, mmap_mode="r")   # do not slice here

            if len(self._mm_cache) >= 4:
                self._mm_cache.pop(next(iter(self._mm_cache)))

            self._mm_cache[key] = (cubes, specs)
            mm = (cubes, specs)

        return mm

    def __getitem__(self, i):
        s = self.samples[i]
        cubes, specs = self._load_shard(s["cube_shard"], s["spec_shard"])
        j = int(s["offset"])

        x = cubes[j]
        y = specs[j, :self.spec_len].astype(np.float32)

        # Important because the model softmax output sums to one over the target grid.
        ysum = float(y.sum())
        if ysum > 0:
            y = y / ysum

        return torch.from_numpy(np.asarray(x)), torch.from_numpy(y)

# -----------------------------
# Losses (same as your script)
# -----------------------------
class CorrelationMAELoss(nn.Module):
    def forward(self, y_pred, y_true):
        vx = y_pred - torch.mean(y_pred)
        vy = y_true - torch.mean(y_true)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-12)
        cost -= 5e1 * torch.mean(torch.abs(y_pred - y_true))
        return 1 - cost

class CorrelationLoss(nn.Module):
    def forward(self, y_pred, y_true):
        vx = y_pred - torch.mean(y_pred)
        vy = y_true - torch.mean(y_true)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-12)
        return 1 - cost

class KLLoss(nn.Module):
    def forward(self, y_pred, y_true, eps=1e-7):
        y_pred = torch.clamp(y_pred, eps, 1.0)
        y_true = torch.clamp(y_true, eps, 1.0)
        return torch.sum(y_true * torch.log(y_true / y_pred), dim=1).mean()

class JSLoss(nn.Module):
    def forward(self, y_pred, y_true, eps=1e-7):
        y_pred = torch.clamp(y_pred, eps, 1.0)
        y_true = torch.clamp(y_true, eps, 1.0)
        m = 0.5 * (y_pred + y_true)
        kl1 = torch.sum(y_true * torch.log(y_true / m), dim=1)
        kl2 = torch.sum(y_pred * torch.log(y_pred / m), dim=1)
        return 0.5 * (kl1 + kl2).mean()

class CosineLoss(nn.Module):
    def forward(self, y_pred, y_true):
        y_pred = F.normalize(y_pred, p=2, dim=1)
        y_true = F.normalize(y_true, p=2, dim=1)
        return 1 - torch.sum(y_pred * y_true, dim=1).mean()

LOSS_MAP = {
    "corrmae": CorrelationMAELoss(),
    "corr": CorrelationLoss(),
    "kl": KLLoss(),
    "js": JSLoss(),
    "cosine": CosineLoss(),
    "mse": nn.MSELoss(),
    "l1": nn.L1Loss(),
    "smoothl1": nn.SmoothL1Loss(),
    "corr+kl": lambda p, t: CorrelationLoss()(p, t) + 0.1 * KLLoss()(p, t),
    "corr+js": lambda p, t: CorrelationLoss()(p, t) + 0.1 * JSLoss()(p, t),
    "corr+cosine": lambda p, t: CorrelationLoss()(p, t) + 0.1 * CosineLoss()(p, t),
}


# -----------------------------
# Model (same style, but in_ch=1)
# -----------------------------
class CNNModel(nn.Module):
    def __init__(self, num_layers=10, kernel_size=7, base_filters=8, dropout_p=0.35,
                 input_shape=(1, 165, 169, 155), out_features=DEFAULT_TARGET_POINTS):
        super().__init__()
        assert 4 <= num_layers <= 10
        assert 3 <= kernel_size <= 15
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.base_filters = base_filters
        self.dropout_p = float(dropout_p)
        self.in_ch = int(input_shape[0])
        self.input_shape = input_shape
        self.out_features = out_features

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=1, padding=1)
        self.avgpool = nn.AvgPool3d(kernel_size=2, padding=1)

        blocks = []
        ks = self.kernel_size
        pad_same = int((ks - 1) / 2)

        def channel_mult(i):
            if i == 1:  return 1
            if i == 2:  return 2
            if i == 3:  return 4
            if i == 4:  return 8
            if i == 5:  return 16
            return 32

        in_c = self.in_ch
        for i in range(1, num_layers + 1):
            out_c = self.base_filters * channel_mult(i)
            out_c = min(out_c, self.base_filters * 32)
            conv = nn.Conv3d(in_c, out_c, kernel_size=ks, padding=pad_same, padding_mode="zeros")
            gn = nn.GroupNorm(out_c, out_c)
            blocks.append(nn.Sequential(conv, gn, nn.ReLU(inplace=True)))
            in_c = out_c

        self.blocks = nn.ModuleList(blocks)
        self.drop_mid = nn.Dropout(self.dropout_p, inplace=True) if self.dropout_p > 0 else nn.Identity()
        self.drop_end = nn.Dropout(self.dropout_p, inplace=True) if self.dropout_p > 0 else nn.Identity()
        self.drop_fc  = nn.Dropout(self.dropout_p, inplace=True) if self.dropout_p > 0 else nn.Identity()
        self.flatten = nn.Flatten()

        with torch.no_grad():
            x = torch.rand(1, *self.input_shape)
            x = self._forward_features(x)
            lin_in = self.flatten(x).shape[1]

        self.linear = nn.Linear(lin_in, self.out_features)
        self.softmax = nn.Softmax(dim=1)

    def _forward_features(self, x):
        mid_index = max(1, math.floor(self.num_layers * 0.6))
        for i, block in enumerate(self.blocks, start=1):
            x = block(x)
            x = self.avgpool(self.maxpool(x))
            if i == mid_index:
                x = self.drop_mid(x)
        x = self.drop_end(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.drop_fc(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x


# -----------------------------
# Checkpoint
# -----------------------------
class CustomCheckpoint:
    def __init__(self, model, save_dir):
        self.model = model
        self.save_dir = Path(save_dir)
        self.best_val_loss = float("inf")
        self.best_model_path = self.save_dir / "best_model.pth"

    def __call__(self, epoch, val_loss):
        """Save on improvement and report whether this epoch was the new best."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(self.model.state_dict(), self.best_model_path)
            print(f"✓ Saved best model at epoch {epoch+1} with val loss {val_loss:.6f}")
            return True
        return False


# -----------------------------
# Sweep utils
# -----------------------------
def ensure_root():
    PROJECT_SAVE_ROOT.mkdir(parents=True, exist_ok=True)

def odd_between(a, b):
    lo = a if a % 2 == 1 else a + 1
    hi = b if b % 2 == 1 else b - 1
    choices = list(range(lo, hi + 1, 2))
    return random.choice(choices)

def sample_config(idx):
    num_layers = random.randint(4, 10)
    kernel_size = odd_between(3, 15)
    base_filters = random.randint(2, 8)
    dropout_p = random.choice([0.0, 0.1, 0.2, 0.3, 0.4])
    return {
        "exp_id": f"exp_{idx:03d}",
        "num_layers": num_layers,
        "kernel_size": kernel_size,
        "base_filters": base_filters,
        "dropout_p": dropout_p,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "loss_type": LOSS_TYPE,
        "seed": RNG_SEED,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }

def append_summary_row(row_dict):
    summary_path = PROJECT_SAVE_ROOT / "summary.csv"
    header = [
        "exp_id","num_layers","kernel_size","base_filters","dropout_p",
        "best_val_loss","best_val_corr","best_mae",
        "epochs","batch_size","lr","weight_decay","loss_type","seed","timestamp","exp_dir"
    ]
    exists = summary_path.exists()
    with summary_path.open("a") as f:
        if not exists:
            f.write(",".join(header) + "\n")
        vals = [
            row_dict.get("exp_id"),
            str(row_dict.get("num_layers")),
            str(row_dict.get("kernel_size")),
            str(row_dict.get("base_filters")),
            str(row_dict.get("dropout_p")),
            f'{row_dict.get("best_val_loss", float("nan")):.6f}',
            f'{row_dict.get("best_val_corr", float("nan")):.6f}',
            f'{row_dict.get("best_mae", float("nan")):.6f}',
            str(row_dict.get("epochs")),
            str(row_dict.get("batch_size")),
            str(row_dict.get("lr")),
            str(row_dict.get("weight_decay")),
            row_dict.get("loss_type"),
            str(row_dict.get("seed")),
            row_dict.get("timestamp"),
            str(row_dict.get("exp_dir")),
        ]
        f.write(",".join(vals) + "\n")


# -----------------------------
# Fixed train/validation split
# -----------------------------
def build_fixed_split(n_samples: int, seed: int, val_frac: float, train_fraction: float = 1.0):
    """Return (train_idx, val_idx) over cache-index positions.

    This is the split shared by every training script in this repository and by
    the published runs. Do not change the generator, the slice order or the
    rounding rule: they define which molecules are held out. ``val_frac`` of the
    shuffled indices become validation; ``train_fraction`` then takes a leading
    prefix of the remaining training pool for size-scaling runs, leaving the
    validation set untouched.
    """
    sample_indices = np.arange(n_samples, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(sample_indices)

    n_val = int(round(val_frac * n_samples))
    n_val = max(1, min(n_val, n_samples - 1))
    val_idx = sample_indices[:n_val]
    train_pool = sample_indices[n_val:]

    n_train = int(round(train_fraction * train_pool.size))
    n_train = max(1, min(n_train, train_pool.size))
    return train_pool[:n_train], val_idx


# -----------------------------
# Training
# -----------------------------
def run_one_experiment(cfg, dataset, train_idx, val_idx, cube_shape, spec_len, num_workers=4):
    exp_dir = PROJECT_SAVE_ROOT / cfg["exp_id"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    (exp_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    # Written per experiment so that downstream analysis can map each saved
    # validation row back to a molecule through the cache index.
    (exp_dir / "indices.json").write_text(json.dumps({
        "fraction": cfg.get("train_fraction", 1.0),
        "n_train": len(train_idx),
        "train_idx": [int(i) for i in train_idx],
        "val_idx": [int(i) for i in val_idx],
    }))

    # subsets
    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )

    model = CNNModel(
        num_layers=cfg["num_layers"],
        kernel_size=cfg["kernel_size"],
        base_filters=cfg["base_filters"],
        dropout_p=cfg["dropout_p"],
        input_shape=tuple(cube_shape),
        out_features=int(spec_len),
    ).to(DEVICE)

    criterion = LOSS_MAP[cfg["loss_type"]]
    corr_loss = CorrelationLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    checkpoint = CustomCheckpoint(model, exp_dir)

    log_path = exp_dir / "train_log.csv"
    with log_path.open("w") as f:
        f.write("epoch,train_loss,val_loss,mae,train_corr,val_corr\n")

    best_val_corr = -1e9
    best_mae = 1e9

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss_sum = 0.0
        tr_corr_sum = 0.0
        tr_count = 0

        for cubes, spectra in train_loader:
            cubes = cubes.to(DEVICE, non_blocking=True).float()
            spectra = spectra.to(DEVICE, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            preds = model(cubes)
            loss = criterion(preds, spectra)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * cubes.size(0)
            tr_corr_sum += (1.0 - corr_loss(preds, spectra).item()) * spectra.size(0)
            tr_count += spectra.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)
        avg_tr_corr = tr_corr_sum / max(1, tr_count)

        model.eval()
        val_loss_sum = 0.0
        va_corr_sum = 0.0
        va_count = 0
        preds_list, targs_list = [], []
        with torch.no_grad():
            for cubes, spectra in val_loader:
                cubes = cubes.to(DEVICE, non_blocking=True).float()
                spectra = spectra.to(DEVICE, non_blocking=True).float()
                out = model(cubes)
                loss = criterion(out, spectra)
                val_loss_sum += loss.item() * cubes.size(0)
                va_corr_sum += (1.0 - corr_loss(out, spectra).item()) * spectra.size(0)
                va_count += spectra.size(0)
                preds_list.append(out.detach().cpu().numpy())
                targs_list.append(spectra.detach().cpu().numpy())

        val_loss = val_loss_sum / len(val_loader.dataset)
        preds_np = np.concatenate(preds_list, axis=0)
        targs_np = np.concatenate(targs_list, axis=0)
        mae = float(np.mean(np.abs(preds_np - targs_np)))
        avg_val_corr = va_corr_sum / max(1, va_count)

        if checkpoint(epoch, val_loss):
            # scripts/analysis/generate_figure_6_data.py consumes this file.
            # Rows follow val_idx, which the validation loader never shuffles.
            np.savez_compressed(
                exp_dir / "val_preds_true.npz",
                pred=preds_np, true=targs_np,
                val_idx=np.asarray(val_idx, dtype=np.int64),
            )
        if avg_val_corr > best_val_corr:
            best_val_corr = avg_val_corr
            best_mae = min(best_mae, mae)

        with log_path.open("a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{mae:.6f},{avg_tr_corr:.6f},{avg_val_corr:.6f}\n")

        print(f"[{cfg['exp_id']}] Epoch {epoch+1}/{cfg['epochs']}  "
              f"Train {train_loss:.6f} | Val {val_loss:.6f} | MAE {mae:.6f} | "
              f"TrCorr {avg_tr_corr:.4f} | VaCorr {avg_val_corr:.4f}")

    result = {
        "exp_id": cfg["exp_id"],
        "num_layers": cfg["num_layers"],
        "kernel_size": cfg["kernel_size"],
        "base_filters": cfg["base_filters"],
        "dropout_p": cfg["dropout_p"],
        "best_val_loss": float(checkpoint.best_val_loss),
        "best_val_corr": float(best_val_corr),
        "best_mae": float(best_mae),
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "loss_type": cfg["loss_type"],
        "seed": cfg["seed"],
        "timestamp": cfg["timestamp"],
        "exp_dir": str(exp_dir.resolve()),
    }
    return result


# -----------------------------
# Main
# -----------------------------
def main():
    global NUM_EXPERIMENTS, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, LOSS_TYPE, RNG_SEED, PROJECT_SAVE_ROOT

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Directory containing molecule subdirectories, or the root used by --density-glob.")
    ap.add_argument("--cache-dir", default=None, help="Default: <root>/_cache_density_spec_shards")
    ap.add_argument("--density-glob", default=DEFAULT_DENSITY_GLOB, help="Glob template used to find density files. Fields: {root}.")
    ap.add_argument("--spectrum-template", default=DEFAULT_SPECTRUM_TEMPLATE, help="Template used to find the spectrum paired with each density. Fields: {root}, {density_dir}, {mid}.")
    ap.add_argument("--shard-size", type=int, default=128)
    ap.add_argument("--cube-dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--spec-sigma", type=float, default=10.0)
    ap.add_argument("--target-points", type=int, default=DEFAULT_TARGET_POINTS, help="Number of spectrum bins used by the model head.")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--train-fraction", type=float, default=1.0,
                    help="Leading fraction of the training pool to use, for size-scaling runs. "
                         "The validation set is unaffected.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--experiments", type=int, default=NUM_EXPERIMENTS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    ap.add_argument("--loss-type", default=LOSS_TYPE, choices=list(LOSS_MAP.keys()))
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--outdir", default=str(PROJECT_SAVE_ROOT), help="Directory for CNN sweep outputs.")
    args = ap.parse_args()

    NUM_EXPERIMENTS = args.experiments
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    LOSS_TYPE = args.loss_type
    RNG_SEED = args.seed
    
    root = args.root
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(root) / DEFAULT_CACHE_DIRNAME
    PROJECT_SAVE_ROOT = Path(args.outdir)

    pairs = find_pairs(root, args.density_glob, args.spectrum_template)
    if not pairs:
        raise FileNotFoundError("No (density_M*.npz, tddft_spectrum_*.dat) pairs found.")

    # deterministic order + deterministic split
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)

    index_json = build_cache_shards(
        pairs=pairs,
        cache_dir=cache_dir,
        shard_size=args.shard_size,
        cube_dtype=args.cube_dtype,
        spec_sigma=args.spec_sigma,
        check_voxel_vectors=False,
        target_points=args.target_points,
    )

    if args.prepare_only:
        print("[DONE] Cache prepared only.")
        return

    ensure_root()

    dataset = ShardedMemmapDataset(str(index_json), target_points=args.target_points)
    n = len(dataset)
    train_arr, val_arr = build_fixed_split(n, RNG_SEED, args.val_frac, args.train_fraction)
    train_idx = train_arr.tolist()
    val_idx = val_arr.tolist()
    print(f"[INFO] fixed split: train={len(train_idx)} val={len(val_idx)} "
          f"(seed={RNG_SEED}, val_frac={args.val_frac}, train_fraction={args.train_fraction})")

    split_path = PROJECT_SAVE_ROOT / "split.json"
    split_path.write_text(json.dumps({
        "seed": RNG_SEED,
        "val_frac": args.val_frac,
        "train_fraction": args.train_fraction,
        "target_points": args.target_points,
        "cache_index": str(index_json),
        "N": int(n),
        "train_idx": train_idx,
        "val_idx": val_idx,
    }, indent=2))
    print(f"[INFO] split written to {split_path}")

    # performance tweaks
    torch.backends.cudnn.benchmark = True

    # run sweep
    all_results = []
    for i in range(NUM_EXPERIMENTS):
        cfg = sample_config(i + 1)
        cfg.update({
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "loss_type": LOSS_TYPE,
            "seed": RNG_SEED,
            "val_frac": args.val_frac,
            "train_fraction": args.train_fraction,
        })
        print(f"\n=== Starting {cfg['exp_id']} ===")
        print(cfg)
        res = run_one_experiment(
            cfg, dataset, train_idx, val_idx,
            cube_shape=dataset.cube_shape,
            spec_len=dataset.spec_len,
            num_workers=args.num_workers,
        )
        append_summary_row(res)
        all_results.append(res)

    (PROJECT_SAVE_ROOT / "summary.json").write_text(json.dumps(all_results, indent=2))
    print("\nAll experiments complete. Summary written to:")
    print(f"- {PROJECT_SAVE_ROOT / 'summary.csv'}")
    print(f"- {PROJECT_SAVE_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
