#!/usr/bin/env python3
"""Train a MACE-based graph model for normalized absorption spectra.

The model reads molecular geometries from XYZ files, uses a MACE backbone to
produce node features, pools the invariant features to a molecular embedding,
and predicts a normalized spectrum on a fixed frequency grid. The target vector
is treated as a probability distribution, so the final layer is a softmax and
the default loss is KL divergence.

This script expects the cache index produced by ``scripts/training/build_density_spectrum_cache.py``.
The index defines the molecule ordering used for the fixed split and contains
``samples`` entries with at least ``dens_npz`` and ``spec_dat`` fields.

Default geometry layout:

    <xyz-root>/
      <molecule-id>/
        GEO_M<molecule-id>.xyz

Default spectrum source:

    The spectrum path stored in the cache index, ``samples[i]["spec_dat"]``.

For a different layout, pass templates instead of changing the code. Available
fields are ``{xyz_root}``, ``{mid}``, ``{dens_npz}``, and ``{spec_dat}``.

Run example:

    python scripts/training/train_mace_spectrum_from_cache.py \
      --cache-index /path/to/cache-dir/index.json \
      --xyz-root /path/to/xyz-root \
      --outdir /path/to/mace-runs \
      --epochs 200

If the cache index points to spectra in an old location, either rebuild the
cache or pass ``--spectrum-template`` to reconstruct the spectrum path from the
molecule id.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from torch.serialization import add_safe_globals

add_safe_globals([slice])

from e3nn import o3
from torch.utils.data import Dataset

from mace import modules
from mace.data.atomic_data import AtomicData
from mace.data.utils import Configuration
from mace.modules.utils import compute_avg_num_neighbors, extract_invariant
from mace.tools import AtomicNumberTable, torch_geometric
from mace.tools.scatter import scatter_mean

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_TARGET_POINTS = 900
DEFAULT_XYZ_TEMPLATE = "{xyz_root}/{mid}/GEO_M{mid}.xyz"



# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
class KLLossProb(nn.Module):
    """KL(q || p), with both p and q interpreted as probability distributions."""

    def forward(self, p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        p = torch.clamp(p, eps, 1.0)
        q = torch.clamp(q, eps, 1.0)
        return torch.sum(q * (torch.log(q) - torch.log(p)), dim=1).mean()


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpectrumMetrics:
    @staticmethod
    def pearson_per_sample(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        pred0 = pred - pred.mean(axis=1, keepdims=True)
        true0 = true - true.mean(axis=1, keepdims=True)
        num = np.sum(pred0 * true0, axis=1)
        den = np.sqrt(np.sum(pred0 * pred0, axis=1) * np.sum(true0 * true0, axis=1)) + eps
        return num / den


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


_DENS_RE = re.compile(r"density_M(\d+)\.npz$")


def parse_mid_from_dens(dens_npz: str) -> int:
    m = _DENS_RE.search(dens_npz)
    if not m:
        raise ValueError(f"Cannot parse molecule id from {dens_npz}")
    return int(m.group(1))


ATOM_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Br": 35,
    "I": 53,
}


def format_record_path(template: str, *, mid: int, xyz_root: Path | None = None, rec: dict | None = None) -> Path:
    """Format a user-provided path template for molecule-specific files."""
    rec = rec or {}
    return Path(
        template.format(
            mid=mid,
            xyz_root=xyz_root,
            dens_npz=rec.get("dens_npz", ""),
            spec_dat=rec.get("spec_dat", ""),
        )
    )


def read_xyz_geo(xyz_root: Path, mid: int, xyz_template: str = DEFAULT_XYZ_TEMPLATE) -> tuple[np.ndarray, np.ndarray]:
    path = format_record_path(xyz_template, mid=mid, xyz_root=xyz_root)
    with path.open("r", encoding="utf-8") as f:
        n = int(f.readline().strip())
        _ = f.readline()
        z = np.zeros((n,), dtype=np.int64)
        pos = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            parts = f.readline().split()
            sym = parts[0]
            if sym not in ATOM_Z:
                raise ValueError(f"Unknown element '{sym}' in {path}")
            z[i] = ATOM_Z[sym]
            pos[i] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return z, pos


def resolve_spectrum_path(rec: dict, mid: int, spectrum_template: str | None = None, spec_tag: str | None = None) -> Path:
    if spectrum_template:
        return format_record_path(spectrum_template, mid=mid, rec=rec)

    path = str(rec["spec_dat"])
    if spec_tag:
        path = path.replace("gamma_150meV_", f"gamma_{spec_tag}_")
    return Path(path)


def load_spectrum_from_dat(
    spec_path: str | Path,
    target_points: int = DEFAULT_TARGET_POINTS,
    skiprows: int = 1,
    clip_neg: bool = True,
) -> np.ndarray:
    arr = np.loadtxt(spec_path, skiprows=skiprows)
    y = arr if arr.ndim == 1 else arr[:, -1]
    y = y.astype(np.float32, copy=False)

    if y.shape[0] < target_points:
        y2 = np.zeros((target_points,), dtype=np.float32)
        y2[: y.shape[0]] = y
        y = y2
    else:
        y = y[:target_points]

    if clip_neg:
        y = np.clip(y, 0.0, None)

    s = float(y.sum())
    if s <= 0:
        y[:] = 1.0 / float(target_points)
    else:
        y /= s
    return y


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class MACEAtomicDataset(Dataset):
    """Preload non-periodic molecular graphs and their spectrum targets."""

    def __init__(
        self,
        cache_index: dict,
        targets_all: np.ndarray,
        ds_indices: Sequence[int],
        xyz_root: Path,
        xyz_template: str,
        z_table: AtomicNumberTable,
        r_max: float,
    ) -> None:
        self.samples = cache_index["samples"]
        self.ds_indices = np.asarray(ds_indices, dtype=np.int64)
        self.data_list: List[AtomicData] = []

        for ds_idx in self.ds_indices:
            rec = self.samples[int(ds_idx)]
            mid = parse_mid_from_dens(rec["dens_npz"])
            z, pos = read_xyz_geo(xyz_root, mid, xyz_template)
            y = targets_all[int(ds_idx)]

            cfg = Configuration(
                atomic_numbers=z,
                positions=pos,
                properties={},
                property_weights={},
                cell=np.zeros((3, 3), dtype=np.float64),
                pbc=(False, False, False),
                head="Default",
                config_type="Default",
                weight=1.0,
            )
            data = AtomicData.from_config(
                config=cfg,
                z_table=z_table,
                cutoff=r_max,
                heads=["Default"],
            )
            data.y = torch.from_numpy(y.copy()).float().view(1, -1)
            data.ds_idx = torch.tensor([int(ds_idx)], dtype=torch.long)
            data.mid = torch.tensor([mid], dtype=torch.long)
            self.data_list.append(data)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> AtomicData:
        return self.data_list[idx]


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class MACEBackboneSpectrum(nn.Module):
    """
    Official MACE backbone + custom graph readout for spectra.

    The backbone is the official ScaleShiftMACE module. We ignore its energy
    output and instead use out["node_feats"], then keep only invariant channels
    with extract_invariant(...), pool them graph-wise, and predict the target
    spectrum.
    """

    def __init__(
        self,
        z_table: AtomicNumberTable,
        out_dim: int = DEFAULT_TARGET_POINTS,
        r_max: float = 5.0,
        num_radial_basis: int = 8,
        num_cutoff_basis: int = 5,
        max_ell: int = 3,
        correlation: int = 3,
        num_interactions: int = 2,
        hidden_irreps: str = "128x0e + 128x1o",
        mlp_irreps: str = "16x0e",
        radial_mlp: str = "[64, 64, 64]",
        radial_type: str = "bessel",
        distance_transform: str = "None",
        gate: str = "silu",
        head_hidden: int = 512,
        head_dropout: float = 0.1,
        avg_num_neighbors: float = 12.0,
        first_interaction: str = "RealAgnosticInteractionBlock",
        interaction: str = "RealAgnosticResidualInteractionBlock",
        use_last_readout_only: bool = False,
        use_embedding_readout: bool = False,
        use_agnostic_product: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.num_interactions = int(num_interactions)
        self.max_L = self.hidden_irreps.lmax
        self.num_scalar_channels = self.hidden_irreps.count(o3.Irrep(0, 1))

        # One head only. We set atomic energies to zero because they are irrelevant
        # for the custom spectrum objective.
        atomic_energies = np.zeros((len(z_table),), dtype=np.float64)

        self.backbone = modules.ScaleShiftMACE(
            r_max=r_max,
            num_bessel=num_radial_basis,
            num_polynomial_cutoff=num_cutoff_basis,
            max_ell=max_ell,
            interaction_cls=modules.interaction_classes[interaction],
            interaction_cls_first=modules.interaction_classes[first_interaction],
            num_interactions=num_interactions,
            num_elements=len(z_table),
            hidden_irreps=self.hidden_irreps,
            MLP_irreps=o3.Irreps(mlp_irreps),
            atomic_energies=atomic_energies,
            avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=z_table.zs,
            correlation=correlation,
            gate=modules.gate_dict[gate],
            pair_repulsion=False,
            apply_cutoff=True,
            distance_transform=distance_transform,
            radial_MLP=json.loads(radial_mlp),
            radial_type=radial_type,
            heads=["Default"],
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
            use_last_readout_only=use_last_readout_only,
            use_embedding_readout=use_embedding_readout,
            use_agnostic_product=use_agnostic_product,
        )

        inv_dim = self.num_scalar_channels * self.num_interactions
        self.head = MLPHead(
            in_dim=inv_dim,
            hidden_dim=head_hidden,
            out_dim=out_dim,
            dropout=head_dropout,
        )

    def forward(self, batch: AtomicData) -> torch.Tensor:
        out = self.backbone(
            batch,
            training=self.training,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            compute_hessian=False,
            compute_edge_forces=False,
        )
        node_feats = out["node_feats"]
        node_invariants = extract_invariant(
            node_feats,
            num_layers=self.num_interactions,
            num_features=self.num_scalar_channels,
            l_max=self.max_L,
        )
        graph_emb = scatter_mean(
            src=node_invariants,
            index=batch.batch,
            dim=0,
            dim_size=batch.num_graphs,
        )
        return self.head(graph_emb)


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------
def attach_optim_meta(loader, lr: float, weight_decay: float):
    loader.lr = lr
    loader.weight_decay = weight_decay
    return loader


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n = 0
    preds = []
    trues = []

    for batch in loader:
        batch = batch.to(device)
        y = batch.y.view(batch.num_graphs, -1)
        pred = model(batch)
        loss = loss_fn(pred, y)
        bs = batch.num_graphs
        total_loss += float(loss.item()) * bs
        n += bs
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())

    P = np.concatenate(preds, axis=0)
    T = np.concatenate(trues, axis=0)
    mae = float(np.mean(np.abs(P - T)))
    corr = float(np.mean(SpectrumMetrics.pearson_per_sample(P, T)))
    return total_loss / max(1, n), mae, corr, P, T


def train_eval_loop(model, train_loader, val_loader, loss_fn, device, epochs: int, out_dir: Path) -> float:
    ensure_dir(out_dir)
    best_val = float("inf")
    best_path = out_dir / "best_model.pth"
    log_path = out_dir / "train_log.csv"

    with log_path.open("w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,val_mae,val_corr\n")

    opt = torch.optim.AdamW(model.parameters(), lr=train_loader.lr, weight_decay=train_loader.weight_decay)

    for ep in range(1, epochs + 1):
        model.train()
        tr_sum = 0.0
        tr_n = 0

        for batch in train_loader:
            batch = batch.to(device)
            y = batch.y.view(batch.num_graphs, -1)
            opt.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            bs = batch.num_graphs
            tr_sum += float(loss.item()) * bs
            tr_n += bs

        tr_loss = tr_sum / max(1, tr_n)
        va_loss, mae, corr, P, T = evaluate(model, val_loader, loss_fn, device)

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)
            np.savez(out_dir / "val_preds_true.npz", pred=P.astype(np.float32), true=T.astype(np.float32))

        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{ep},{tr_loss:.6f},{va_loss:.6f},{mae:.6f},{corr:.6f}\n")

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"  epoch {ep:03d} | tr {tr_loss:.6f} | va {va_loss:.6f} | mae {mae:.6f} | corr {corr:.4f}")

    return best_val


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def build_targets(
    cache_index: dict,
    out_root: Path,
    target_points: int,
    spectrum_template: str | None = None,
    spec_tag: str | None = None,
) -> np.ndarray:
    """Build or load normalized spectrum targets in cache-index order."""
    n_samples = int(cache_index["num_samples"])
    samples = cache_index["samples"]
    tag = spec_tag or ("template" if spectrum_template else "cache")
    targets_npz = out_root / f"targets_len{target_points}_{tag}_skip1_clip1.npz"

    if targets_npz.exists():
        z = np.load(targets_npz)
        targets_all = z["targets"].astype(np.float32)
        print("[INFO] loaded cached targets:", targets_npz, targets_all.shape)
        return targets_all

    targets_all = np.zeros((n_samples, target_points), dtype=np.float32)
    bad = 0
    for i, rec in enumerate(samples):
        mid = parse_mid_from_dens(rec["dens_npz"])
        try:
            spec_path = resolve_spectrum_path(rec, mid, spectrum_template=spectrum_template, spec_tag=spec_tag)
            targets_all[i] = load_spectrum_from_dat(spec_path, target_points=target_points, skiprows=1, clip_neg=True)
        except Exception:
            bad += 1
            targets_all[i] = 1.0 / float(target_points)

    np.savez(targets_npz, targets=targets_all)
    print("[INFO] wrote targets:", targets_npz, "bad:", bad)
    return targets_all


def gather_atomic_numbers(cache_index: dict, ds_indices: Sequence[int], xyz_root: Path, xyz_template: str) -> List[int]:
    zs = set()
    for ds_idx in ds_indices:
        rec = cache_index["samples"][int(ds_idx)]
        mid = parse_mid_from_dens(rec["dens_npz"])
        z, _ = read_xyz_geo(xyz_root, mid, xyz_template)
        zs.update(int(v) for v in z.tolist())
    return sorted(zs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-index", type=str, required=True, help="Path to index.json produced by the cache-building script.")
    ap.add_argument("--xyz-root", type=str, required=True, help="Root directory used by --xyz-template.")
    ap.add_argument("--xyz-template", type=str, default=DEFAULT_XYZ_TEMPLATE, help="Template for XYZ files. Fields: {xyz_root}, {mid}.")
    ap.add_argument("--spectrum-template", type=str, default=None, help="Optional template for spectrum files. Fields: {mid}, {dens_npz}, {spec_dat}.")
    ap.add_argument("--spec-tag", type=str, default=None, help="Optional replacement for gamma_150meV_ in cached spectrum paths, e.g. 50meV.")
    ap.add_argument("--target-points", type=int, default=DEFAULT_TARGET_POINTS, help="Number of bins predicted by the spectrum head.")
    ap.add_argument("--outdir", type=str, default="mace_spectra_runs")

    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fractions", type=float, nargs="+", default=[1.0])

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=8.0e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--default-dtype", choices=["float32", "float64"], default="float32")

    ap.add_argument("--r-max", type=float, default=5.0)
    ap.add_argument("--num-radial-basis", type=int, default=8)
    ap.add_argument("--num-cutoff-basis", type=int, default=5)
    ap.add_argument("--max-ell", type=int, default=3)
    ap.add_argument("--correlation", type=int, default=3)
    ap.add_argument("--num-interactions", type=int, default=2)
    ap.add_argument("--hidden-irreps", type=str, default="128x0e + 128x1o")
    ap.add_argument("--mlp-irreps", type=str, default="16x0e")
    ap.add_argument("--radial-mlp", type=str, default="[64, 64, 64]")
    ap.add_argument("--radial-type", type=str, default="bessel")
    ap.add_argument("--distance-transform", type=str, default="None")
    ap.add_argument("--gate", type=str, default="silu")
    ap.add_argument("--head-hidden", type=int, default=512)
    ap.add_argument("--head-dropout", type=float, default=0.1)
    ap.add_argument("--first-interaction", type=str, default="RealAgnosticInteractionBlock")
    ap.add_argument("--interaction", type=str, default="RealAgnosticResidualInteractionBlock")
    ap.add_argument("--use-last-readout-only", action="store_true")
    ap.add_argument("--use-embedding-readout", action="store_true")
    ap.add_argument("--use-agnostic-product", action="store_true")
    ap.add_argument("--ckpt", type=str, default=None)
    args = ap.parse_args()

    if args.default_dtype == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_root = Path(args.outdir)
    ensure_dir(out_root)
    cache_path = Path(args.cache_index)
    xyz_root = Path(args.xyz_root)

    cache_index = json.loads(cache_path.read_text())
    N = int(cache_index["num_samples"])
    all_idx = np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_idx)
    n_val = int(round(args.val_frac * N))
    val_idx = all_idx[:n_val]
    train_pool = all_idx[n_val:]

    print(f"[INFO] fixed split: train_pool={train_pool.size} val={val_idx.size} (seed={args.seed}, val_frac={args.val_frac})")

    split_path = out_root / f"split_seed{args.seed}_val{args.val_frac}.json"
    split_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "val_frac": args.val_frac,
                "N": N,
                "val_idx": val_idx.tolist(),
                "train_pool": train_pool.tolist(),
            },
            indent=2,
        )
    )

    targets_all = build_targets(
        cache_index=cache_index,
        out_root=out_root,
        target_points=args.target_points,
        spectrum_template=args.spectrum_template,
        spec_tag=args.spec_tag,
    )
    loss_fn = KLLossProb()

    fractions = [float(f) for f in args.fractions]
    summary_rows = []

    for frac in fractions:
        n_tr = int(round(frac * train_pool.size))
        n_tr = max(1, min(n_tr, train_pool.size))
        tr_idx = train_pool[:n_tr]

        exp_tag = f"mace_frac{int(frac * 100):02d}"
        exp_dir = out_root / exp_tag
        ensure_dir(exp_dir)

        (exp_dir / "indices.json").write_text(
            json.dumps(
                {
                    "fraction": frac,
                    "n_train": int(n_tr),
                    "train_idx": tr_idx.tolist(),
                    "val_idx": val_idx.tolist(),
                },
                indent=2,
            )
        )

        print(f"\n=== {exp_tag} ===")
        print(f"[INFO] train={n_tr} val={val_idx.size}")

        # z-table must cover every element appearing in train/val for this run.
        zs = gather_atomic_numbers(cache_index, np.concatenate([tr_idx, val_idx]), xyz_root, args.xyz_template)
        z_table = AtomicNumberTable(zs)
        print(f"[INFO] z_table = {z_table.zs}")

        train_ds = MACEAtomicDataset(cache_index, targets_all, tr_idx, xyz_root, args.xyz_template, z_table, args.r_max)
        val_ds = MACEAtomicDataset(cache_index, targets_all, val_idx, xyz_root, args.xyz_template, z_table, args.r_max)

        MaceLoader = torch_geometric.dataloader.DataLoader
        avg_loader = MaceLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        avg_num_neighbors = compute_avg_num_neighbors(avg_loader)
        print(f"[INFO] avg_num_neighbors = {avg_num_neighbors:.4f}")

        model = MACEBackboneSpectrum(
            z_table=z_table,
            out_dim=args.target_points,
            r_max=args.r_max,
            num_radial_basis=args.num_radial_basis,
            num_cutoff_basis=args.num_cutoff_basis,
            max_ell=args.max_ell,
            correlation=args.correlation,
            num_interactions=args.num_interactions,
            hidden_irreps=args.hidden_irreps,
            mlp_irreps=args.mlp_irreps,
            radial_mlp=args.radial_mlp,
            radial_type=args.radial_type,
            distance_transform=args.distance_transform,
            gate=args.gate,
            head_hidden=args.head_hidden,
            head_dropout=args.head_dropout,
            avg_num_neighbors=avg_num_neighbors,
            first_interaction=args.first_interaction,
            interaction=args.interaction,
            use_last_readout_only=args.use_last_readout_only,
            use_embedding_readout=args.use_embedding_readout,
            use_agnostic_product=args.use_agnostic_product,
        ).to(DEVICE)

        if args.ckpt:
            model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=True)

        train_loader = attach_optim_meta(
            MaceLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        val_loader = MaceLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        best_val = train_eval_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=DEVICE,
            epochs=args.epochs,
            out_dir=exp_dir,
        )

        summary_rows.append(
            {
                "model": "mace",
                "fraction": frac,
                "n_train": int(n_tr),
                "n_val": int(val_idx.size),
                "best_val_loss": float(best_val),
                "exp_dir": str(exp_dir),
            }
        )

    summary_path = out_root / "summary_mace.csv"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("model,fraction,n_train,n_val,best_val_loss,exp_dir\n")
        for r in summary_rows:
            f.write(
                f"{r['model']},{r['fraction']},{r['n_train']},{r['n_val']},{r['best_val_loss']},{r['exp_dir']}\n"
            )

    print("\n[OK] Wrote summary:", summary_path)
    print("[OK] Split file:", split_path)


if __name__ == "__main__":
    main()
