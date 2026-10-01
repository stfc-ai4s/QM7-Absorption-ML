#!/usr/bin/env python3
"""Run SchNet and DimeNet++ sweeps for molecular spectrum prediction.

This script trains graph neural networks from atomic numbers and XYZ positions
to normalized absorption spectra. It uses the cache index produced by
``scripts/training/build_density_spectrum_cache.py`` as the source of truth for molecule
ordering, target spectra, and deterministic train/validation splits.

Default geometry layout:

    <xyz-root>/
      <molecule-id>/
        GEO_M<molecule-id>.xyz

Default spectrum source:

    The path stored in ``samples[i]["spec_dat"]`` inside the cache index.

For a different layout, pass templates instead of editing the code. Available
fields are ``{xyz_root}``, ``{mid}``, ``{dens_npz}``, and ``{spec_dat}``.

Run example:

    python scripts/training/sweep_schnet_dimenetpp_spectrum_from_cache.py \
      --cache-index /path/to/cache-dir/index.json \
      --xyz-root /path/to/xyz-root \
      --outdir /path/to/graph-runs \
      --model both \
      --runs 10 \
      --epochs 100

If paths inside the cache index are no longer valid, rebuild the cache or pass
``--spectrum-template`` to construct spectrum paths from molecule ids.
"""

import os
import re
import json
import math
import time
import random
import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Sequence

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import SchNet
from torch_geometric.nn.models import DimeNetPlusPlus


DEFAULT_TARGET_POINTS = 900
DEFAULT_XYZ_TEMPLATE = "{xyz_root}/{mid}/GEO_M{mid}.xyz"


# -----------------------------
# Periodic table mapping (enough for QM7/QM7b)
# -----------------------------
_SYMBOLS = [
    "X",
    "H","He",
    "Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "In","Sn","Sb","Te","I","Xe",
]
_SYM2Z = {s: i for i, s in enumerate(_SYMBOLS) if i > 0}
_DENS_RE = re.compile(r"density_M(\d+)\.npz$")


def parse_mid_from_dens(dens_npz: str) -> int:
    m = _DENS_RE.search(str(dens_npz))
    if not m:
        raise ValueError(f"Cannot parse molecule id from density path: {dens_npz}")
    return int(m.group(1))


def format_record_path(template: str, *, mid: int, xyz_root: Path | None = None, rec: dict | None = None) -> Path:
    rec = rec or {}
    return Path(
        template.format(
            mid=mid,
            xyz_root=xyz_root,
            dens_npz=rec.get("dens_npz", ""),
            spec_dat=rec.get("spec_dat", ""),
        )
    )


def resolve_spectrum_path(rec: dict, mid: int, spectrum_template: str | None = None) -> Path:
    if spectrum_template:
        return format_record_path(spectrum_template, mid=mid, rec=rec)
    return Path(rec["spec_dat"])


def read_xyz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Reads XYZ file with format:
    N
    comment
    Sym x y z
    ...
    Returns:
      z: (N,) int64
      pos: (N,3) float32
    """
    lines = path.read_text().strip().splitlines()
    if len(lines) < 3:
        raise ValueError(f"Bad xyz (too short): {path}")
    try:
        n = int(lines[0].strip())
    except Exception as e:
        raise ValueError(f"Bad xyz header: {path} :: {e}")
    atom_lines = lines[2:2+n]
    if len(atom_lines) != n:
        raise ValueError(f"Bad xyz atom count: {path} expected {n} got {len(atom_lines)}")

    z = np.zeros((n,), dtype=np.int64)
    pos = np.zeros((n, 3), dtype=np.float32)
    for i, ln in enumerate(atom_lines):
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"Bad xyz line: {path} :: {ln}")
        sym = parts[0]
        if sym not in _SYM2Z:
            raise ValueError(f"Unknown element '{sym}' in {path}")
        z[i] = _SYM2Z[sym]
        pos[i, 0] = float(parts[1])
        pos[i, 1] = float(parts[2])
        pos[i, 2] = float(parts[3])
    return z, pos


def read_spectrum(path: Path, n_points: int = DEFAULT_TARGET_POINTS) -> np.ndarray:
    """Load a spectrum, select the target grid, and normalize by its sum."""
    arr = np.loadtxt(path, skiprows=1)
    if arr.ndim == 1:
        y = arr.astype(np.float32, copy=False)
    else:
        y = arr[:, -1].astype(np.float32, copy=False)

    if y.size == 0:
        raise ValueError(f"Empty spectrum: {path}")

    if y.size < n_points:
        y2 = np.zeros((n_points,), dtype=np.float32)
        y2[:y.size] = y
        y = y2
    else:
        y = y[:n_points]

    s = float(y.sum())
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"Non-positive spectrum sum over selected target grid: {path}")
    y = y / s
    return y


# -----------------------------
# Models (exactly as you specified)
# -----------------------------
class CustomIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class SchNetSpectrum(nn.Module):
    def __init__(self, out_dim, hidden=8, interactions=6,
                 gaussians=50, cutoff=6.0):
        super().__init__()
        self.core = SchNet(hidden_channels=hidden,
                           num_filters=hidden,
                           num_interactions=interactions,
                           num_gaussians=gaussians,
                           cutoff=cutoff,
                           readout='add')
        self.core.readout = CustomIdentity()
        self.core.lin1    = nn.Identity()
        self.core.lin2    = nn.Identity()

        self.head = nn.Sequential(nn.Linear(hidden, out_dim), nn.Softmax(dim=1))

    def forward(self, batch):
        h_nodes = self.core(batch.z, batch.pos, batch.batch)   # [N, hidden]
        g_emb   = global_mean_pool(h_nodes, batch.batch)       # [B, hidden]
        return self.head(g_emb)                                # [B, out_dim]


class DimeNetPPSpectrum(nn.Module):
    def __init__(
        self,
        out_dim: int,
        hidden: int = 128,
        interactions: int = 4,
        gaussians: int = 6,
        cutoff: float = 6.0,
        num_spherical: int = 7,
        envelope_exponent: int = 5,
        max_num_neighbors: int = 32,
    ):
        super().__init__()
        self.core = DimeNetPlusPlus(
            hidden_channels=hidden,
            out_channels=hidden,
            num_blocks=interactions,
            int_emb_size=64,
            basis_emb_size=8,
            out_emb_channels=256,
            num_spherical=num_spherical,
            num_radial=gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
            act='swish',
            output_initializer='zeros',
        )
        self.head = nn.Sequential(nn.Linear(hidden, out_dim), nn.Softmax(dim=1))

    def forward(self, batch):
        g_emb = self.core(batch.z, batch.pos, batch.batch)
        return self.head(g_emb)


# -----------------------------
# Losses
# -----------------------------
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


class Wasserstein1DLoss(nn.Module):
    """W1 between 1D distributions via CDF distance: sum |CDF_p - CDF_q|."""
    def forward(self, y_pred, y_true):
        # both (B,L), non-negative and sum to 1
        cdf_p = torch.cumsum(y_pred, dim=1)
        cdf_q = torch.cumsum(y_true, dim=1)
        return torch.mean(torch.sum(torch.abs(cdf_p - cdf_q), dim=1))


LOSS_MAP = {
    "js": JSLoss(),
    "kl": KLLoss(),
    "cosine": CosineLoss(),
    "wasserstein": Wasserstein1DLoss(),
    "l1": nn.L1Loss(),
    "mse": nn.MSELoss(),
}


def corr_np(a: np.ndarray, b: np.ndarray, eps=1e-12) -> float:
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    # correlation per-sample then mean (cheap + stable)
    a0 = a - a.mean(axis=1, keepdims=True)
    b0 = b - b.mean(axis=1, keepdims=True)
    num = np.sum(a0 * b0, axis=1)
    den = np.sqrt(np.sum(a0*a0, axis=1) * np.sum(b0*b0, axis=1)) + eps
    return float(np.mean(num / den))


# -----------------------------
# Dataset
# -----------------------------
class MoleculeSpectrumDataset(torch.utils.data.Dataset):
    """Graph dataset built in the exact order defined by the cache index."""

    def __init__(
        self,
        records: Sequence[dict],
        ds_indices: Sequence[int],
        xyz_root: Path,
        target_points: int = DEFAULT_TARGET_POINTS,
        xyz_template: str = DEFAULT_XYZ_TEMPLATE,
        spectrum_template: str | None = None,
        preload: bool = True,
    ):
        self.records = list(records)
        self.ds_indices = [int(i) for i in ds_indices]
        self.xyz_root = Path(xyz_root)
        self.target_points = int(target_points)
        self.xyz_template = xyz_template
        self.spectrum_template = spectrum_template
        self.preload = bool(preload)

        self._data_list: Optional[List[Data]] = None
        if self.preload:
            self._data_list = self._build_all()

    def _paths(self, ds_idx: int) -> Tuple[int, Path, Path]:
        rec = self.records[int(ds_idx)]
        mid = parse_mid_from_dens(rec["dens_npz"])
        xyz = format_record_path(self.xyz_template, mid=mid, xyz_root=self.xyz_root, rec=rec)
        spec = resolve_spectrum_path(rec, mid, spectrum_template=self.spectrum_template)
        return mid, xyz, spec

    def _build_one(self, ds_idx: int) -> Data:
        mid, xyz, spec = self._paths(ds_idx)
        z, pos = read_xyz(xyz)
        y = read_spectrum(spec, n_points=self.target_points)

        # Keep y as (1, L), so PyG batches it into (B, L).
        data = Data(
            z=torch.from_numpy(z.astype(np.int64)),
            pos=torch.from_numpy(pos.astype(np.float32)),
            y=torch.from_numpy(y.astype(np.float32)).view(1, -1),
            ds_idx=torch.tensor([int(ds_idx)], dtype=torch.long),
            mid=torch.tensor([mid], dtype=torch.long),
        )
        return data

    def _build_all(self) -> List[Data]:
        out = []
        bad = 0
        t0 = time.time()
        for ds_idx in self.ds_indices:
            try:
                out.append(self._build_one(ds_idx))
            except Exception as exc:
                bad += 1
                print(f"[WARN] failed to load ds_idx={ds_idx}: {exc}")
        dt = time.time() - t0
        if len(out) == 0:
            raise RuntimeError("No valid samples. Check cache, XYZ paths, and spectrum paths.")
        print(f"[INFO] preload: kept={len(out)} bad={bad} in {dt:.1f}s")
        return out

    def __len__(self):
        return len(self._data_list) if self._data_list is not None else len(self.ds_indices)

    def __getitem__(self, idx: int) -> Data:
        if self._data_list is not None:
            return self._data_list[idx]
        return self._build_one(self.ds_indices[idx])


# -----------------------------
# Training
# -----------------------------
def train_one(
    model_name: str,
    exp_dir: Path,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_name: str,
    lr: float,
    weight_decay: float,
    epochs: int,
    device: torch.device,
):
    exp_dir.mkdir(parents=True, exist_ok=True)

    criterion = LOSS_MAP[loss_name]
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_path = exp_dir / "best_model.pth"

    log_path = exp_dir / "train_log.csv"
    with log_path.open("w") as f:
        f.write("epoch,train_loss,val_loss,val_corr,val_mae\n")

    for epoch in range(1, epochs + 1):
        model.train()
        tr_sum = 0.0
        tr_n = 0

        for batch in train_loader:
            batch = batch.to(device)
            y = batch.y
            if y.dim() == 1:
                y = y.view(int(batch.num_graphs), -1)

            optim.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = criterion(pred, y)
            loss.backward()
            optim.step()

            tr_sum += float(loss.item()) * int(batch.num_graphs)
            tr_n += int(batch.num_graphs)

        train_loss = tr_sum / max(1, tr_n)

        # val (keep num_workers=0 to avoid silent worker failures)
        model.eval()
        va_sum = 0.0
        va_n = 0
        preds = []
        trues = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                y = batch.y
                if y.dim() == 1:
                    y = y.view(int(batch.num_graphs), -1)

                pred = model(batch)
                loss = criterion(pred, y)
                va_sum += float(loss.item()) * int(batch.num_graphs)
                va_n += int(batch.num_graphs)

                preds.append(pred.detach().cpu().numpy())
                trues.append(y.detach().cpu().numpy())

        val_loss = va_sum / max(1, va_n)

        P = np.concatenate(preds, axis=0)
        T = np.concatenate(trues, axis=0)

        val_corr = corr_np(P, T)
        val_mae = float(np.mean(np.abs(P - T)))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            tag = " [best]"
        else:
            tag = ""

        with log_path.open("a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_corr:.6f},{val_mae:.6f}\n")

        print(f"[{model_name}] epoch {epoch:03d}/{epochs}  train {train_loss:.6f}  val {val_loss:.6f}  corr {val_corr:.4f}  mae {val_mae:.6f}{tag}")

    # final eval dump on val
    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    model.to(device)
    model.eval()
    preds = []
    trues = []
    mids = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            y = batch.y
            if y.dim() == 1:
                y = y.view(int(batch.num_graphs), -1)
            pred = model(batch)
            preds.append(pred.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())
            mids.append(batch.mid.detach().cpu().numpy())

    P = np.concatenate(preds, axis=0)
    T = np.concatenate(trues, axis=0)
    M = np.concatenate(mids, axis=0).reshape(-1)

    np.savez(exp_dir / "val_preds_true.npz", pred=P.astype(np.float32), true=T.astype(np.float32), mid=M.astype(np.int64))

    return {
        "best_val_loss": float(best_val),
        "best_model": str(best_path),
        "val_corr": float(corr_np(P, T)),
        "val_mae": float(np.mean(np.abs(P - T))),
        "log": str(log_path),
    }


def sample_schnet_cfg(rng: random.Random) -> Dict:
    return {
        "hidden": rng.choice([32, 64, 128]),
        "interactions": rng.choice([3, 4, 6, 8]),
        "gaussians": rng.choice([25, 50, 75]),
        "cutoff": rng.choice([3.0, 5.0, 6.0]),
        "lr": 10 ** rng.uniform(-5, -3),
        "loss": rng.choice(["js", "kl", "cosine", "wasserstein", "l1"]),
    }


def sample_dimenet_cfg(rng: random.Random) -> Dict:
    return {
        "hidden": rng.choice([64, 128, 256]),
        "interactions": rng.choice([2, 3, 4, 6]),
        "gaussians": rng.choice([6, 12, 24]),
        "cutoff": rng.choice([3.0, 5.0, 6.0]),
        "max_num_neighbors": rng.choice([16, 32, 64]),
        "lr": 10 ** rng.uniform(-5, -3),
        "loss": rng.choice(["js", "kl", "cosine", "wasserstein", "l1"]),
    }


def write_summary_row(csv_path: Path, row: Dict, header: List[str]):
    exists = csv_path.exists()
    with csv_path.open("a") as f:
        if not exists:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(row.get(k, "")) for k in header) + "\n")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-index", required=True, help="Path to index.json produced by the cache-building script.")
    ap.add_argument("--xyz-root", required=True, help="Root directory used by --xyz-template.")
    ap.add_argument("--xyz-template", default=DEFAULT_XYZ_TEMPLATE, help="Template for XYZ files. Fields: {xyz_root}, {mid}.")
    ap.add_argument("--spectrum-template", default=None, help="Optional spectrum template. Fields: {mid}, {dens_npz}, {spec_dat}.")
    ap.add_argument("--outdir", default="graph_spectrum_sweeps")
    ap.add_argument("--target-points", type=int, default=DEFAULT_TARGET_POINTS)
    ap.add_argument("--model", choices=["schnet", "dimenetpp", "both"], default="both")
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=0, help="Keep 0 to avoid worker-related empty validation.")
    ap.add_argument("--max-samples", type=int, default=0, help="For quick tests using the first N cache records after ordering (0=all).")
    ap.add_argument("--preload", action="store_true", help="Preload all graphs and targets into RAM.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cache_index = json.loads(Path(args.cache_index).read_text())
    records = cache_index["samples"]
    n_total = int(cache_index["num_samples"])
    if n_total != len(records):
        raise ValueError("cache index num_samples does not match samples length")

    n_used = min(n_total, args.max_samples) if args.max_samples else n_total
    all_idx = np.arange(n_used, dtype=np.int64)
    rng_np = np.random.default_rng(args.seed)
    perm = rng_np.permutation(all_idx)
    n_val = int(round(args.val_frac * n_used))
    n_val = max(1, min(n_val, n_used - 1))

    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()

    split_payload = {
        "seed": args.seed,
        "val_frac": args.val_frac,
        "target_points": args.target_points,
        "cache_index": args.cache_index,
        "xyz_root": args.xyz_root,
        "xyz_template": args.xyz_template,
        "spectrum_template": args.spectrum_template,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_mids": [parse_mid_from_dens(records[i]["dens_npz"]) for i in train_idx],
        "val_mids": [parse_mid_from_dens(records[i]["dens_npz"]) for i in val_idx],
    }
    (outdir / "split.json").write_text(json.dumps(split_payload, indent=2))

    print(f"[INFO] cache records used: {n_used}")
    print(f"[INFO] split: train={len(train_idx)} val={len(val_idx)}")

    train_ds = MoleculeSpectrumDataset(
        records, train_idx, Path(args.xyz_root),
        target_points=args.target_points,
        xyz_template=args.xyz_template,
        spectrum_template=args.spectrum_template,
        preload=args.preload,
    )
    val_ds = MoleculeSpectrumDataset(
        records, val_idx, Path(args.xyz_root),
        target_points=args.target_points,
        xyz_template=args.xyz_template,
        spectrum_template=args.spectrum_template,
        preload=args.preload,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    summary_csv = outdir / "summary.csv"
    header = [
        "model", "exp_id", "hidden", "interactions", "gaussians", "cutoff", "max_num_neighbors",
        "lr", "loss", "epochs", "batch_size", "seed",
        "best_val_loss", "val_corr", "val_mae", "exp_dir"
    ]

    rng = random.Random(args.seed)

    def run_sweep(model_name: str, sampler_fn, runs: int):
        for r in range(1, runs + 1):
            exp_id = f"exp_{r:03d}"
            cfg = sampler_fn(rng)
            cfg["seed"] = args.seed
            cfg["epochs"] = args.epochs
            cfg["batch_size"] = args.batch_size

            exp_dir = outdir / model_name / exp_id
            exp_dir.mkdir(parents=True, exist_ok=True)
            (exp_dir / "config.json").write_text(json.dumps({
                "model": model_name,
                "exp_id": exp_id,
                "cfg": cfg,
                "target_points": args.target_points,
                "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
            }, indent=2))

            print(f"\n=== {model_name} {exp_id} ===")
            print(cfg)

            if model_name == "schnet":
                model = SchNetSpectrum(
                    out_dim=args.target_points,
                    hidden=int(cfg["hidden"]),
                    interactions=int(cfg["interactions"]),
                    gaussians=int(cfg["gaussians"]),
                    cutoff=float(cfg["cutoff"]),
                )
            else:
                model = DimeNetPPSpectrum(
                    out_dim=args.target_points,
                    hidden=int(cfg["hidden"]),
                    interactions=int(cfg["interactions"]),
                    gaussians=int(cfg["gaussians"]),
                    cutoff=float(cfg["cutoff"]),
                    max_num_neighbors=int(cfg["max_num_neighbors"]),
                )

            model.to(device)

            row_metrics = train_one(
                model_name=model_name,
                exp_dir=exp_dir,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_name=str(cfg["loss"]),
                lr=float(cfg["lr"]),
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                device=device,
            )

            row = {
                "model": model_name,
                "exp_id": exp_id,
                "hidden": cfg.get("hidden"),
                "interactions": cfg.get("interactions"),
                "gaussians": cfg.get("gaussians"),
                "cutoff": cfg.get("cutoff"),
                "max_num_neighbors": cfg.get("max_num_neighbors", ""),
                "lr": cfg.get("lr"),
                "loss": cfg.get("loss"),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "best_val_loss": row_metrics["best_val_loss"],
                "val_corr": row_metrics["val_corr"],
                "val_mae": row_metrics["val_mae"],
                "exp_dir": str(exp_dir),
            }

            write_summary_row(summary_csv, row, header)

    if args.model in ("schnet", "both"):
        run_sweep("schnet", sample_schnet_cfg, args.runs)

    if args.model in ("dimenetpp", "both"):
        run_sweep("dimenetpp", sample_dimenet_cfg, args.runs)

    print("\n[OK] Done. Summary:", summary_csv)


if __name__ == "__main__":
    main()
