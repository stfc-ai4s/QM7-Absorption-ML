#!/usr/bin/env python3
"""
Predict a spectrum's normalization constant from an electron density.

The spectrum models in this repository end in a softmax, so they predict only
the shape of a spectrum. This model supplies the scale: one scalar per
molecule, the sum of the first 900 points of the reference spectrum. Multiply
a predicted unit-sum spectrum by this constant to recover an absolute one.

Trained by `scripts/training/train_density_normconst_from_cache.py`; the model
behind Figure 5.

Two modes:

1. `validation` — recompute `val_preds_true_raw.npz` for a whole run, from the
   run's split, its cached targets and the density shards. This is the file
   Figure 5 plots.
2. `density` — predict the constant for a single density stored as .npz, .npy
   or a .pt/.pth tensor.

Both modes need the run's `config.json`, which carries the target transform
and the standardization constants needed to report a value in raw units.

The model may be either a TorchScript export (`normconst_model.pt`) or a plain
state dict (`best_model.pth`); the architecture for a state dict is rebuilt
from `config.json`, never guessed.

Examples
--------
Recompute validation predictions:

    python scripts/inference/predict_normconst_from_density.py validation \
        --model   /your/scratch/qm7-spectra/weights/normconst_model.pt \
        --config  /your/scratch/qm7-spectra/runs/normconst/config.json \
        --split   /your/scratch/qm7-spectra/runs/normconst/split.json \
        --targets /your/scratch/qm7-spectra/runs/normconst/targets_sum900.npz \
        --output  /tmp/val_preds_true_raw.npz

Predict one density:

    python scripts/inference/predict_normconst_from_density.py density \
        --model  /your/scratch/qm7-spectra/weights/normconst_model.pt \
        --config /your/scratch/qm7-spectra/runs/normconst/config.json \
        --density /your/scratch/qm7-spectra/dataset/1/density_M1.npz

For an .npz holding several arrays, name the one to use with `--key`.

Note on preprocessing: the `validation` mode reads densities straight out of
the cache shards, so they are already normalized and centred. A density passed
to `density` mode must be prepared the same way the cache builder prepares one
— divided by its maximum absolute amplitude and centre-of-mass aligned onto
the cube shape in `config.json`.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# The regressor class lives with the training script; import it rather than
# keeping a second copy that could drift out of step.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from train_density_normconst_from_cache import CNNRegressor  # noqa: E402


def inverse_target(values, config):
    """Convert model outputs back to raw normalization-constant units."""
    values = np.asarray(values, dtype=np.float64)
    target_cfg = config["target"]

    if target_cfg.get("standardize", False):
        y_mean = float(target_cfg["y_mean"])
        y_std = float(target_cfg["y_std"])
        values = values * y_std + y_mean

    transform = target_cfg.get("transform", "none")

    if transform == "none":
        return values
    if transform == "log":
        return np.exp(values)

    raise ValueError(f"Unsupported target transform: {transform}")


def load_model(path: Path, config: dict, device):
    """Load a TorchScript export, or a state dict rebuilt from config.json."""
    try:
        model = torch.jit.load(str(path), map_location=device)
        model.eval()
        return model
    except RuntimeError:
        pass  # not TorchScript; fall through to the state-dict path

    obj = torch.load(str(path), map_location=device)

    if isinstance(obj, torch.nn.Module):
        obj.to(device).eval()
        return obj

    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint object in {path}: {type(obj)}")

    state = obj.get("state_dict", obj)

    arch = config.get("arch")
    if arch is None:
        raise ValueError(
            f"{path} is a state dict, so the architecture must come from "
            f"config.json, but it has no 'arch' entry."
        )

    model = CNNRegressor(
        num_layers=int(arch["num_layers"]),
        kernel_size=int(arch["kernel_size"]),
        base_filters=int(arch["base_filters"]),
        dropout_p=float(arch["dropout_p"]),
        input_shape=tuple(config["cube_shape"]),
        out_features=int(arch.get("out_features", 1)),
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_density(path, key=None):
    """Load one density array from .npy, .npz, .pt, or .pth."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        x = np.load(path)

    elif suffix == ".npz":
        z = np.load(path)

        if key is not None:
            if key not in z.files:
                raise KeyError(
                    f"Key '{key}' not found in {path}. Available keys: {z.files}"
                )
            x = z[key]
        else:
            preferred = ("data", "density", "rho", "cube", "arr_0")
            selected = next((k for k in preferred if k in z.files), None)

            if selected is None:
                if len(z.files) == 1:
                    selected = z.files[0]
                else:
                    raise ValueError(
                        f"{path} contains multiple arrays {z.files}. "
                        "Specify the density array with --key."
                    )
            x = z[selected]

    elif suffix in (".pt", ".pth"):
        obj = torch.load(path, map_location="cpu")

        if torch.is_tensor(obj):
            x = obj.detach().cpu().numpy()
        elif isinstance(obj, np.ndarray):
            x = obj
        elif isinstance(obj, dict):
            if key is not None:
                if key not in obj:
                    raise KeyError(
                        f"Key '{key}' not found in {path}. Available keys: {list(obj)}"
                    )
                value = obj[key]
            else:
                preferred = ("data", "density", "rho", "cube")
                selected = next((k for k in preferred if k in obj), None)
                if selected is None:
                    raise ValueError(
                        "Tensor file is a dictionary. Specify the density entry with --key."
                    )
                value = obj[selected]

            x = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        else:
            raise TypeError(f"Unsupported object stored in {path}: {type(obj)}")

    else:
        raise ValueError(
            f"Unsupported density file type '{suffix}'. Use .npy, .npz, .pt, or .pth."
        )

    return np.asarray(x)


def prepare_single_density(x, cube_shape):
    """
    Convert one density to model input shape (1, C, X, Y, Z).

    Accepted input shapes:
      (X, Y, Z)
      (C, X, Y, Z)
      (1, C, X, Y, Z)
    """
    x = np.asarray(x)

    if x.ndim == 3 and cube_shape[0] == 1:
        x = x[None, ...]  # -> (C,X,Y,Z)

    if x.ndim == 4:
        if tuple(x.shape) != tuple(cube_shape):
            raise ValueError(
                f"Density shape {tuple(x.shape)} does not match expected "
                f"(C,X,Y,Z) = {tuple(cube_shape)}"
            )
        x = x[None, ...]  # -> (1,C,X,Y,Z)

    elif x.ndim == 5:
        if x.shape[0] != 1 or tuple(x.shape[1:]) != tuple(cube_shape):
            raise ValueError(
                f"Density shape {tuple(x.shape)} does not match expected "
                f"(1,C,X,Y,Z) = {(1, *cube_shape)}"
            )

    else:
        raise ValueError(
            f"Expected a 3D, 4D, or single-sample 5D density array; got {x.ndim}D."
        )

    return np.ascontiguousarray(x, dtype=np.float32)


class CachedDensityDataset(Dataset):
    """Read validation densities directly from the training cache shards."""

    def __init__(self, cache_index, ds_indices):
        self.samples = cache_index["samples"]
        self.ds_indices = np.asarray(ds_indices, dtype=np.int64)
        self._mm_cache = {}

    def __len__(self):
        return len(self.ds_indices)

    def _load_shard(self, path):
        mm = self._mm_cache.get(path)

        if mm is None:
            mm = np.load(path, mmap_mode="r")
            if len(self._mm_cache) >= 4:
                self._mm_cache.pop(next(iter(self._mm_cache)))
            self._mm_cache[path] = mm

        return mm

    def __getitem__(self, i):
        ds_idx = int(self.ds_indices[i])
        rec = self.samples[ds_idx]

        shard = self._load_shard(rec["cube_shard"])
        # copy out of the read-only memmap so torch gets a writable array
        density = np.array(shard[int(rec["offset"])], dtype=np.float32)

        return torch.from_numpy(density), ds_idx


def pearsonr(a, b, eps=1e-12):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)

    a0 = a - a.mean()
    b0 = b - b.mean()

    return float(
        (a0 * b0).sum()
        / (np.sqrt((a0 * a0).sum() * (b0 * b0).sum()) + eps)
    )


def resolve_device(requested):
    return torch.device(
        requested if requested else ("cuda" if torch.cuda.is_available() else "cpu")
    )


def run_validation(args):
    config = json.loads(args.config.read_text())
    split = json.loads(args.split.read_text())

    cache_index_path = Path(args.cache_index or config["cache_index"])
    if not cache_index_path.exists():
        raise SystemExit(
            f"Cache index not found: {cache_index_path}\n"
            f"config.json records the path used at training time; pass "
            f"--cache-index if the cache has moved."
        )
    cache_index = json.loads(cache_index_path.read_text())

    val_ds_idx = np.asarray(split["val_ds_idx"], dtype=np.int64)
    cube_shape = tuple(config["cube_shape"])

    targets = np.load(args.targets)
    target_ds_idx = targets["ds_idx"].astype(np.int64)
    target_raw = targets["y_sum"].astype(np.float64)

    # Map dataset index -> true raw target.
    target_lookup = {
        int(ds_idx): float(y)
        for ds_idx, y in zip(target_ds_idx, target_raw)
    }

    missing = [int(i) for i in val_ds_idx if int(i) not in target_lookup]
    if missing:
        raise RuntimeError(
            f"{len(missing)} validation indices are missing from the targets file. "
            f"First few: {missing[:10]}"
        )

    dataset = CachedDensityDataset(cache_index, val_ds_idx)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
    )

    device = resolve_device(args.device)
    model = load_model(args.model, config, device)

    preds_model_space = []
    seen_indices = []

    with torch.inference_mode():
        for xb, ds_idx in loader:
            xb = xb.to(device, non_blocking=True).float()

            if tuple(xb.shape[1:]) != cube_shape:
                raise ValueError(
                    f"Cached density shape {tuple(xb.shape[1:])} does not match "
                    f"config cube_shape {cube_shape}"
                )

            pred = model(xb).reshape(-1)
            preds_model_space.append(pred.cpu().numpy())
            seen_indices.append(ds_idx.numpy())

    pred_model = np.concatenate(preds_model_space)
    seen_indices = np.concatenate(seen_indices).astype(np.int64)

    pred_raw = inverse_target(pred_model, config)
    true_raw = np.asarray(
        [target_lookup[int(i)] for i in seen_indices],
        dtype=np.float64,
    )

    err = pred_raw - true_raw
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    corr = pearsonr(pred_raw, true_raw)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Same pred/true keys as the training output, so the result can be handed
    # straight to make_figures_3_4_5.py --normalization-npz.
    np.savez(
        args.output,
        pred=pred_raw.astype(np.float32),
        true=true_raw.astype(np.float32),
        val_ds_idx=seen_indices,
    )

    print(f"[DONE] Saved: {args.output}")
    print(f"[INFO] N validation samples: {len(pred_raw)}")
    print(f"[INFO] MAE(raw):  {mae:.6f}")
    print(f"[INFO] RMSE(raw): {rmse:.6f}")
    print(f"[INFO] Corr(raw): {corr:.6f}")


def run_density(args):
    config = json.loads(args.config.read_text())
    cube_shape = tuple(config["cube_shape"])

    x = load_density(args.density, key=args.key)
    x = prepare_single_density(x, cube_shape)

    device = resolve_device(args.device)
    model = load_model(args.model, config, device)

    xb = torch.from_numpy(x).to(device)

    with torch.inference_mode():
        pred_model = model(xb).reshape(-1).cpu().numpy()

    pred_raw = inverse_target(pred_model, config)
    value = float(pred_raw[0])

    print(f"Density file: {args.density}")
    print(f"Normalization constant: {value:.10g}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output,
            normalization_constant=np.float64(value),
            density_file=str(args.density),
        )
        print(f"[DONE] Saved: {args.output}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Predict a spectrum normalization constant from an electron density."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_val = subparsers.add_parser(
        "validation",
        help="Recompute validation predictions and save val_preds_true_raw.npz.",
    )
    p_val.add_argument("--model", required=True, type=Path,
                       help="normconst_model.pt (TorchScript) or best_model.pth (state dict)")
    p_val.add_argument("--config", required=True, type=Path, help="Training config.json")
    p_val.add_argument("--split", required=True, type=Path, help="Training split.json")
    p_val.add_argument("--targets", required=True, type=Path, help="targets_sum*.npz")
    p_val.add_argument(
        "--cache-index",
        type=Path,
        default=None,
        help="Cache index JSON. Default: use config.json entry.",
    )
    p_val.add_argument(
        "--output",
        type=Path,
        default=Path("val_preds_true_raw.npz"),
        help="Output NPZ file.",
    )
    p_val.add_argument("--batch-size", type=int, default=16)
    p_val.add_argument("--num-workers", type=int, default=4)
    p_val.add_argument(
        "--device",
        default=None,
        help="Device such as cuda, cuda:0, or cpu. Default: CUDA if available.",
    )
    p_val.set_defaults(func=run_validation)

    p_den = subparsers.add_parser(
        "density",
        help="Predict the normalization constant for one density file.",
    )
    p_den.add_argument("--model", required=True, type=Path,
                       help="normconst_model.pt (TorchScript) or best_model.pth (state dict)")
    p_den.add_argument("--config", required=True, type=Path, help="Training config.json")
    p_den.add_argument("--density", required=True, type=Path, help="Input density file")
    p_den.add_argument(
        "--key",
        default=None,
        help="Array/dictionary key for density if the input contains multiple entries.",
    )
    p_den.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional .npz file for saving the predicted normalization constant.",
    )
    p_den.add_argument(
        "--device",
        default=None,
        help="Device such as cuda, cuda:0, or cpu. Default: CUDA if available.",
    )
    p_den.set_defaults(func=run_density)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
