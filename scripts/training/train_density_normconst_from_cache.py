#!/usr/bin/env python3
"""
Train the 3D CNN that predicts a spectrum's normalization constant from a
density cube.

The spectrum models in this repository all end in a softmax, so they predict
the *shape* of an absorption spectrum and nothing about its overall scale.
This model supplies the missing scale: a single scalar per molecule, the sum
of the first 900 points of the TDDFT reference spectrum. Multiplying a
predicted unit-sum spectrum by this constant recovers an absolute spectrum.

It is the model behind Figure 5, whose parity plot compares the predicted and
true constants over the validation set.

Inputs are the shards and `index.json` written by
`build_density_spectrum_cache.py`; the targets are read once from the
`spec_dat` files that index records, and cached as `targets_sum<N>.npz` inside
the run directory so later runs start immediately.

Determinism
-----------
The split matches every other training script in this repository. Positions in
the cache index are shuffled with `np.random.default_rng(--seed)` and the
leading `--val-frac` fraction is held out. Here the shuffle is written as
`rng.permutation(n)` over positions in the list of samples that have a usable
spectrum; when every cached sample has one — which is the case for the
published 6874-molecule cache — that list is `arange(n)` and the result is
exactly the permutation the other scripts use. `--seed 0 --val-frac 0.15`
reproduces the published 5843/1031 split, so Figure 5's validation molecules
are the same 1031 molecules the Figure 6/7 analysis uses.

Outputs
-------
    <outdir>/
      config.json               hyperparameters, target transform, y_mean/y_std
      split.json                train_ds_idx / val_ds_idx
      targets_sum900.npz        ds_idx / y_sum, reused across runs
      train_log.csv             per-epoch losses and raw-unit metrics
      best_model.pth            state dict at the best validation loss
      normconst_model.pt        TorchScript export of that state dict
      val_preds_true_raw.npz    pred / true / val_ds_idx / epoch, at that epoch

`config.json` carries the standardization constants, so
`scripts/inference/predict_normconst_from_density.py` can undo them and report
a constant in raw units.

Example
-------
    python scripts/training/train_density_normconst_from_cache.py \
        --cache-index /your/scratch/qm7-spectra/cache/index.json \
        --outdir /your/scratch/qm7-spectra/runs/normconst
"""

import argparse
import datetime
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =============================
# Spectrum loader: read last column, take first n_points, sum
# =============================
def load_sum_from_dat(path: str, n_points: int = 900, clip_negative: bool = True) -> float:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        y = arr.astype(np.float64, copy=False)
    else:
        # last column is intensity
        y = arr[:, -1].astype(np.float64, copy=False)

    if y.size == 0:
        return float("nan")

    if clip_negative:
        y = np.clip(y, 0.0, None)

    y_window = y[:n_points] if y.size >= n_points else y
    return float(np.sum(y_window))


def transform_y(y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return y
    if transform == "log":
        eps = 1e-12
        return np.log(y + eps)
    raise ValueError(f"Unknown target transform: {transform}")


def inverse_transform_y(y_t: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return y_t
    if transform == "log":
        return np.exp(y_t)
    raise ValueError(f"Unknown target transform: {transform}")


# =============================
# Dataset: cube from shard mmaps + scalar target
# =============================
class CubeScalarDataset(Dataset):
    def __init__(self, cache_index: dict, ds_indices, y_transformed, y_mean=None, y_std=None):
        self.samples = cache_index["samples"]
        self.ds_indices = np.asarray(ds_indices, dtype=np.int64)

        # already transformed (optionally log), not yet standardized
        self.y = np.asarray(y_transformed, dtype=np.float32)

        self.y_mean = y_mean
        self.y_std = y_std

        self.cube_shape = tuple(cache_index["cube_shape"])  # (C,X,Y,Z)
        self._mm_cache = {}  # cube_shard -> memmap

    def __len__(self):
        return len(self.ds_indices)

    def _load_cube_shard(self, cube_path: str):
        mm = self._mm_cache.get(cube_path)
        if mm is None:
            mm = np.load(cube_path, mmap_mode="r")
            if len(self._mm_cache) >= 4:
                self._mm_cache.pop(next(iter(self._mm_cache)))
            self._mm_cache[cube_path] = mm
        return mm

    def __getitem__(self, i):
        ds_idx = int(self.ds_indices[i])
        rec = self.samples[ds_idx]

        cubes = self._load_cube_shard(rec["cube_shard"])
        off = int(rec["offset"])
        # copy out of the read-only memmap so torch gets a writable array
        x = np.array(cubes[off])  # (C,X,Y,Z)

        yy = float(self.y[i])
        if self.y_mean is not None and self.y_std is not None:
            yy = (yy - self.y_mean) / (self.y_std + 1e-12)

        return torch.from_numpy(x), torch.tensor([yy], dtype=torch.float32)


# =============================
# Model: the spectrum CNN backbone with a scalar regression head.
# out_features=1 and no softmax, unlike the spectrum models.
# =============================
class CNNRegressor(nn.Module):
    def __init__(self, num_layers=10, kernel_size=7, base_filters=8, dropout_p=0.35,
                 input_shape=(1, 165, 169, 155), out_features=1):
        super().__init__()
        assert 4 <= num_layers <= 10
        assert 3 <= kernel_size <= 15

        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.base_filters = int(base_filters)
        self.dropout_p = float(dropout_p)

        self.in_ch = int(input_shape[0])
        self.input_shape = tuple(input_shape)
        self.out_features = int(out_features)

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=1, padding=1)
        self.avgpool = nn.AvgPool3d(kernel_size=2, padding=1)

        ks = self.kernel_size
        pad_same = int((ks - 1) / 2)

        def channel_mult(i):
            if i == 1:  return 1
            if i == 2:  return 2
            if i == 3:  return 4
            if i == 4:  return 8
            if i == 5:  return 16
            return 32

        blocks = []
        in_c = self.in_ch
        for i in range(1, self.num_layers + 1):
            out_c = self.base_filters * channel_mult(i)
            out_c = min(out_c, self.base_filters * 32)
            conv = nn.Conv3d(in_c, out_c, kernel_size=ks, padding=pad_same, padding_mode="zeros")
            gn = nn.GroupNorm(out_c, out_c)
            blocks.append(nn.Sequential(conv, gn, nn.ReLU(inplace=True)))
            in_c = out_c

        self.blocks = nn.ModuleList(blocks)
        self.drop_mid = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        self.drop_end = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        self.drop_fc  = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        self.flatten = nn.Flatten()

        with torch.no_grad():
            tmp = torch.rand(1, *self.input_shape)
            tmp = self._forward_features(tmp)
            lin_in = self.flatten(tmp).shape[1]

        self.linear = nn.Linear(lin_in, self.out_features)

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
        return self.linear(x)  # [B,1]


# =============================
# Metrics
# =============================
def pearsonr(a, b, eps=1e-12):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    a0 = a - a.mean()
    b0 = b - b.mean()
    return float((a0 * b0).sum() / (np.sqrt((a0 * a0).sum() * (b0 * b0).sum()) + eps))


def build_targets(samples, targets_path: Path, spec_points: int, clip_negative: bool):
    """Sum each reference spectrum once and cache the result beside the run."""
    if targets_path.exists():
        zz = np.load(targets_path)
        ds_idx_list = zz["ds_idx"].astype(np.int64)
        y_sum = zz["y_sum"].astype(np.float64)
        print("[INFO] Loaded cached targets:", targets_path, "N=", ds_idx_list.size)
        return ds_idx_list, y_sum

    ds_idx_list = []
    y_sum_list = []
    missing = 0
    bad = 0

    for ds_idx, rec in enumerate(samples):
        spec = rec.get("spec_dat", None)
        if spec is None or not Path(spec).exists():
            missing += 1
            continue
        try:
            s = load_sum_from_dat(spec, n_points=spec_points, clip_negative=clip_negative)
            if not np.isfinite(s):
                bad += 1
                continue
            ds_idx_list.append(ds_idx)
            y_sum_list.append(s)
        except Exception:
            bad += 1

    ds_idx_list = np.asarray(ds_idx_list, dtype=np.int64)
    y_sum = np.asarray(y_sum_list, dtype=np.float64)

    np.savez(targets_path, ds_idx=ds_idx_list, y_sum=y_sum)
    print("[INFO] Saved targets:", targets_path)
    print(f"[INFO] valid={ds_idx_list.size}, missing_spec={missing}, bad_parse={bad}")

    if missing or bad:
        print("[WARN] Some samples were skipped, so the split is taken over the "
              "surviving subset and will not match a full-cache run.")

    return ds_idx_list, y_sum


def export_torchscript(model, cube_shape, out_path: Path, device):
    """Trace the trained model in eval mode for the inference script."""
    model.eval()
    try:
        with torch.no_grad():
            example = torch.zeros(1, *cube_shape, device=device)
            traced = torch.jit.trace(model, example)
        traced.save(str(out_path))
        print("Saved:", out_path)
    except Exception as exc:  # tracing is a convenience, never lose the run over it
        print(f"[WARN] TorchScript export failed ({exc}). "
              f"best_model.pth is still usable: the inference script accepts a "
              f"state dict together with config.json.")


def build_parser():
    p = argparse.ArgumentParser(
        description="Train a density -> spectrum-normalization-constant 3D CNN "
                    "from the shared density/spectrum cache.")

    p.add_argument("--cache-index", required=True, type=Path,
                   help="index.json written by build_density_spectrum_cache.py")
    p.add_argument("--outdir", type=Path, default=Path("runs/normconst"),
                   help="Run directory. Default: runs/normconst")

    g = p.add_argument_group("target")
    g.add_argument("--spec-points", type=int, default=900,
                   help="Sum the leading N points of the reference spectrum. Default: 900")
    g.add_argument("--no-clip-negative", dest="clip_negative", action="store_false",
                   help="Do not clip negative intensities before summing.")
    g.add_argument("--target-transform", choices=("none", "log"), default="none",
                   help="Fit log(sum) instead of sum. Default: none")
    g.add_argument("--no-standardize-target", dest="standardize_target", action="store_false",
                   help="Do not standardize the target with the training mean and std.")

    g = p.add_argument_group("architecture")
    g.add_argument("--num-layers", type=int, default=7)
    g.add_argument("--kernel-size", type=int, default=15)
    g.add_argument("--base-filters", type=int, default=8)
    g.add_argument("--dropout-p", type=float, default=0.2)

    g = p.add_argument_group("training")
    g.add_argument("--seed", type=int, default=0,
                   help="Split and initialisation seed. The published run used 0.")
    g.add_argument("--val-frac", type=float, default=0.15,
                   help="Validation fraction. The published run used 0.15.")
    g.add_argument("--epochs", type=int, default=200)
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--lr", type=float, default=5e-5)
    g.add_argument("--weight-decay", type=float, default=1e-6)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--loss", choices=("mse", "l1"), default="l1")
    g.add_argument("--device", default=None,
                   help="Device such as cuda, cuda:0 or cpu. Default: CUDA if available.")
    g.add_argument("--no-torchscript", dest="torchscript", action="store_false",
                   help="Skip the TorchScript export of the best model.")

    return p


def main():
    args = build_parser().parse_args()

    cache_index_path = args.cache_index.expanduser().resolve()
    if not cache_index_path.exists():
        raise SystemExit(f"Missing cache index: {cache_index_path}")

    run_dir = args.outdir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache_index = json.loads(cache_index_path.read_text())
    samples = cache_index["samples"]
    cube_shape = tuple(cache_index["cube_shape"])

    print("[INFO] Loaded cache index:", cache_index_path)
    print("[INFO] cube_shape:", cube_shape)
    print("[INFO] total cached samples:", len(samples))
    print(f"[INFO] target: sum of first {args.spec_points} spectrum points (last column). "
          f"clip_neg={args.clip_negative}, transform={args.target_transform}")

    targets_path = run_dir / f"targets_sum{args.spec_points}.npz"
    ds_idx_list, y_sum = build_targets(samples, targets_path,
                                       args.spec_points, args.clip_negative)

    if ds_idx_list.size == 0:
        raise SystemExit("No valid spectrum targets found. Check spec_dat paths in the cache index.")

    y_t = transform_y(y_sum.astype(np.float64), args.target_transform)

    # ---- split: identical to every other training script here.
    # permutation(n) over positions in ds_idx_list; when no sample was skipped
    # ds_idx_list is arange(n) and this is the shared split exactly.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(ds_idx_list.size)
    n_val = int(round(args.val_frac * ds_idx_list.size))
    n_val = max(1, min(n_val, ds_idx_list.size - 1))
    va_sel = perm[:n_val]
    tr_sel = perm[n_val:]

    tr_ds_idx = ds_idx_list[tr_sel]
    va_ds_idx = ds_idx_list[va_sel]
    tr_y_t = y_t[tr_sel]
    va_y_t = y_t[va_sel]

    # ---- standardize in transformed space, using training statistics only
    if args.standardize_target:
        y_mean = float(tr_y_t.mean())
        y_std = float(tr_y_t.std() + 1e-12)
    else:
        y_mean = None
        y_std = None

    cfg = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cache_index": str(cache_index_path),
        "run_dir": str(run_dir),
        "target": {
            "type": f"sum_first_{args.spec_points}_points",
            "clip_negative": bool(args.clip_negative),
            "transform": args.target_transform,
            "standardize": bool(args.standardize_target),
            "y_mean": y_mean,
            "y_std": y_std,
        },
        "cube_shape": list(cube_shape),
        "arch": dict(num_layers=args.num_layers, kernel_size=args.kernel_size,
                     base_filters=args.base_filters, dropout_p=args.dropout_p,
                     out_features=1),
        "train": dict(seed=args.seed, val_frac=args.val_frac, epochs=args.epochs,
                      batch_size=args.batch_size, lr=args.lr,
                      weight_decay=args.weight_decay, loss=args.loss),
        "counts": dict(valid=int(ds_idx_list.size), train=int(tr_ds_idx.size),
                       val=int(va_ds_idx.size)),
        "targets_file": str(targets_path),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (run_dir / "split.json").write_text(json.dumps({
        "train_ds_idx": tr_ds_idx.tolist(),
        "val_ds_idx": va_ds_idx.tolist(),
    }, indent=2))

    print("[INFO] split:", len(tr_ds_idx), "train,", len(va_ds_idx), "val")
    if args.standardize_target:
        print("[INFO] target stats (train, transformed): mean/std =", y_mean, y_std)

    # ---- datasets/loaders
    train_ds = CubeScalarDataset(cache_index, tr_ds_idx, tr_y_t, y_mean=y_mean, y_std=y_std)
    val_ds = CubeScalarDataset(cache_index, va_ds_idx, va_y_t, y_mean=y_mean, y_std=y_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=(args.num_workers > 0))

    # ---- model/loss/optim
    model = CNNRegressor(
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        base_filters=args.base_filters,
        dropout_p=args.dropout_p,
        input_shape=cube_shape,
        out_features=1,
    ).to(device)

    criterion = nn.MSELoss() if args.loss == "mse" else nn.L1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_path = run_dir / "best_model.pth"
    preds_path = run_dir / "val_preds_true_raw.npz"
    best_val = float("inf")

    log_path = run_dir / "train_log.csv"
    with log_path.open("w") as f:
        f.write("epoch,train_loss,val_loss,mae_raw,rmse_raw,corr_raw\n")

    torch.backends.cudnn.benchmark = True
    print("[INFO] Training on", device, "...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_sum = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).float()  # [B,1]
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            tr_sum += loss.item() * xb.size(0)

        tr_loss = tr_sum / len(train_loader.dataset)

        # ---- validation. The loader never shuffles, so rows follow va_ds_idx.
        model.eval()
        va_sum = 0.0
        preds_t = []
        trues_t = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True).float()
                yb = yb.to(device, non_blocking=True).float()
                pred = model(xb)
                loss = criterion(pred, yb)
                va_sum += loss.item() * xb.size(0)
                preds_t.append(pred.cpu().numpy())
                trues_t.append(yb.cpu().numpy())

        va_loss = va_sum / len(val_loader.dataset)

        p = np.concatenate(preds_t, axis=0).reshape(-1)
        t = np.concatenate(trues_t, axis=0).reshape(-1)

        # un-standardize, then invert the transform, to get raw sum units
        if args.standardize_target:
            p_t = p * y_std + y_mean
            t_t = t * y_std + y_mean
        else:
            p_t, t_t = p, t

        p_raw = inverse_transform_y(p_t, args.target_transform)
        t_raw = inverse_transform_y(t_t, args.target_transform)

        err = p_raw - t_raw
        mae_raw = float(np.mean(np.abs(err)))
        rmse_raw = float(np.sqrt(np.mean(err ** 2)))
        corr_raw = pearsonr(p_raw, t_raw)

        tag = ""
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)
            # Predictions from the same epoch as the checkpoint, so the saved
            # parity data always describes the saved model.
            np.savez(
                preds_path,
                pred=p_raw.astype(np.float32),
                true=t_raw.astype(np.float32),
                val_ds_idx=np.asarray(va_ds_idx, dtype=np.int64),
                epoch=np.int64(epoch),
            )
            tag = " [best]"

        with log_path.open("a") as f:
            f.write(f"{epoch},{tr_loss:.6f},{va_loss:.6f},{mae_raw:.6f},{rmse_raw:.6f},{corr_raw:.6f}\n")

        print(f"Epoch {epoch:03d}/{args.epochs} | train {tr_loss:.6f} | val {va_loss:.6f} | "
              f"MAE(raw) {mae_raw:.6f} | RMSE(raw) {rmse_raw:.6f} | Corr(raw) {corr_raw:.4f}{tag}")

    if args.torchscript and best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        export_torchscript(model, cube_shape, run_dir / "normconst_model.pt", device)

    print("\n[DONE] Best val loss:", best_val)
    print("Saved:", best_path)
    print("Saved:", preds_path)
    print("Saved:", log_path)


if __name__ == "__main__":
    main()
