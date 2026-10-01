#!/usr/bin/env python3
"""Run a four-model graph sweep for molecular spectrum prediction.

This script compares SchNet, SchNet with attention-style pooling, DimeNet++,
and a DimeNet++ variant with a richer graph head. All models read molecule
geometries from XYZ files and predict a normalized absorption spectrum.

Use this after building the shared cache with
``scripts/training/build_density_spectrum_cache.py``. The cache index controls molecule
ordering and the deterministic train/validation split, so CNN, SchNet,
DimeNet++, and MACE runs can be compared on the same samples.

Default geometry layout:

    <xyz-root>/
      <molecule-id>/
        GEO_M<molecule-id>.xyz

Default spectrum source:

    The path stored in ``samples[i]["spec_dat"]`` inside the cache index.

For another layout, pass templates rather than editing the code. Available
fields are ``{xyz_root}``, ``{mid}``, ``{dens_npz}``, and ``{spec_dat}``.

Run example:

    python scripts/training/sweep_graph_spectrum_models_from_cache.py \
      --cache-index /path/to/cache-dir/index.json \
      --xyz-root /path/to/xyz-root \
      --outdir /path/to/four-model-runs \
      --runs-per-model 5 \
      --epochs 200

If the cache index contains old spectrum paths, rebuild the cache or pass
``--spectrum-template`` to construct spectrum paths from molecule ids.
"""

import argparse
import datetime
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    global_mean_pool, global_max_pool, global_add_pool, GlobalAttention
)
from torch_geometric.nn.models import SchNet, DimeNetPlusPlus


DEFAULT_TARGET_POINTS = 900
DEFAULT_XYZ_TEMPLATE = "{xyz_root}/{mid}/GEO_M{mid}.xyz"

VAL_FRAC = 0.15
SEED = 0
OUT_DIM = DEFAULT_TARGET_POINTS
SKIP_SPEC_ROWS = 1
CLIP_NEG = True
RUNS_PER_MODEL = 5
EPOCHS = 200
BATCH_SIZE = 8
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_DECAY = 1e-6

# =========================
# Losses for distributions
# =========================
class KLLoss(nn.Module):
    def forward(self, p, q, eps=1e-7):
        p = torch.clamp(p, eps, 1.0)
        q = torch.clamp(q, eps, 1.0)
        return torch.sum(q * torch.log(q / p), dim=1).mean()

class JSLoss(nn.Module):
    def forward(self, p, q, eps=1e-7):
        p = torch.clamp(p, eps, 1.0)
        q = torch.clamp(q, eps, 1.0)
        m = 0.5 * (p + q)
        kl_qm = torch.sum(q * torch.log(q / m), dim=1)
        kl_pm = torch.sum(p * torch.log(p / m), dim=1)
        return 0.5 * (kl_qm + kl_pm).mean()

class Wasserstein1D(nn.Module):
    """
    1D EMD/Wasserstein-1 on discrete bins: mean |CDF(p)-CDF(q)|.
    Assumes p,q are distributions over bins.
    """
    def forward(self, p, q):
        cdf_p = torch.cumsum(p, dim=1)
        cdf_q = torch.cumsum(q, dim=1)
        return torch.mean(torch.abs(cdf_p - cdf_q))

LOSS_MAP = {
    "js": JSLoss(),
    "kl": KLLoss(),
    "wasserstein": Wasserstein1D(),
    "l1": nn.L1Loss(),
    "mse": nn.MSELoss(),
}

# =========================
# Models (as you requested)
# =========================
class CustomIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

class SchNetSpectrum(nn.Module):
    def __init__(self, out_dim, hidden=128, interactions=6, gaussians=50, cutoff=6.0):
        super().__init__()
        self.core = SchNet(
            hidden_channels=hidden,
            num_filters=hidden,
            num_interactions=interactions,
            num_gaussians=gaussians,
            cutoff=cutoff,
            readout="add",
        )
        # bypass graph-level parts
        self.core.readout = CustomIdentity()
        self.core.lin1 = nn.Identity()
        self.core.lin2 = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, batch):
        h_nodes = self.core(batch.z, batch.pos, batch.batch)   # [N, hidden]
        g_emb = global_mean_pool(h_nodes, batch.batch)         # [B, hidden]
        return self.head(g_emb)                                 # [B, out_dim]

class SchNetSpectrum_attpoolv0(nn.Module):
    def __init__(self, out_dim, hidden=128, interactions=6, gaussians=50, cutoff=6.0):
        super().__init__()
        self.core = SchNet(
            hidden_channels=hidden,
            num_filters=hidden,
            num_interactions=interactions,
            num_gaussians=gaussians,
            cutoff=cutoff,
            readout="add",
        )

        # bypass graph-level parts
        self.core.readout = CustomIdentity()
        self.core.lin1 = nn.Identity()
        self.core.lin2 = nn.Identity()

        self.att_gate = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.att_pool = GlobalAttention(self.att_gate)

        self.head = nn.Sequential(
            nn.Linear(4 * hidden, out_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, batch):
        h_nodes = self.core(batch.z, batch.pos, batch.batch)   # [N, hidden]
        g_mean = global_mean_pool(h_nodes, batch.batch)
        g_max  = global_max_pool(h_nodes, batch.batch)
        g_add  = global_add_pool(h_nodes, batch.batch)
        g_att  = self.att_pool(h_nodes, batch.batch)
        combined = torch.cat([g_mean, g_max, g_add, g_att], dim=1)
        return self.head(combined)

class DimeNetPPSpectrum(nn.Module):
    def __init__(self, out_dim, hidden=128, interactions=4, gaussians=6, cutoff=6.0,
                 num_spherical=7, envelope_exponent=5, max_num_neighbors=32):
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
            act="swish",
            output_initializer="zeros",
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, batch):
        g_emb = self.core(batch.z, batch.pos, batch.batch)      # [B, hidden]
        return self.head(g_emb)

class DimeNetPPSpectrum_attpoolv0(nn.Module):
    """
    DimeNet++ does not expose node embeddings in its standard forward.
    So we build an "attpool-like" richer graph head on the graph embedding:
      g_att = g * sigmoid(W g)
      combined = [g, |g|, g^2, g_att]
    """
    def __init__(self, out_dim, hidden=128, interactions=4, gaussians=6, cutoff=6.0,
                 num_spherical=7, envelope_exponent=5, max_num_neighbors=32):
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
            act="swish",
            output_initializer="zeros",
        )
        self.gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.Linear(4 * hidden, out_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, batch):
        g = self.core(batch.z, batch.pos, batch.batch)          # [B, hidden]
        g_att = g * self.gate(g)
        combined = torch.cat([g, torch.abs(g), g * g, g_att], dim=1)
        return self.head(combined)

# =========================
# IO helpers
# =========================
DENS_RE = re.compile(r"density_M(\d+)\.npz$")

ATOM_Z = {
    "H": 1,  "He": 2,
    "Li": 3, "Be": 4, "B": 5,  "C": 6,  "N": 7,  "O": 8,  "F": 9,  "Ne": 10,
    "Na": 11,"Mg": 12,"Al": 13,"Si": 14,"P": 15, "S": 16, "Cl": 17,"Ar": 18,
    "K": 19, "Ca": 20, "Br": 35, "I": 53,
}


def parse_mid_from_dens(dens_npz: str) -> int:
    m = DENS_RE.search(str(dens_npz))
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


def read_xyz(path: Path):
    """Read an XYZ file and return atomic numbers and positions."""
    with path.open("r", encoding="utf-8") as f:
        n = int(f.readline().strip())
        _ = f.readline()
        z = np.zeros((n,), dtype=np.int64)
        pos = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            parts = f.readline().split()
            sym = parts[0]
            if sym not in ATOM_Z:
                raise ValueError(f"Unknown element '{sym}' in {path}")
            z[i] = ATOM_Z[sym]
            pos[i, 0] = float(parts[1])
            pos[i, 1] = float(parts[2])
            pos[i, 2] = float(parts[3])
    return z, pos


def read_spectrum(path: Path, target_points: int = DEFAULT_TARGET_POINTS):
    """Load the selected spectrum grid and normalize it to unit area."""
    arr = np.loadtxt(path, skiprows=SKIP_SPEC_ROWS)
    if arr.ndim == 1:
        y = arr.astype(np.float32, copy=False)
    else:
        y = arr[:, -1].astype(np.float32, copy=False)

    if y.shape[0] < target_points:
        y2 = np.zeros((target_points,), dtype=np.float32)
        y2[:y.shape[0]] = y
        y = y2
    else:
        y = y[:target_points]

    if CLIP_NEG:
        y = np.clip(y, 0.0, None)

    s = float(y.sum())
    if s <= 0:
        y[:] = 1.0 / float(target_points)
    else:
        y /= s
    return y

# =========================
# Dataset (preloaded, stable)
# =========================
class PreloadedGraphDataset(torch.utils.data.Dataset):
    """Preload graphs and targets in the exact cache-index order."""

    def __init__(
        self,
        records,
        xyz_root: Path,
        target_points: int = DEFAULT_TARGET_POINTS,
        xyz_template: str = DEFAULT_XYZ_TEMPLATE,
        spectrum_template: str | None = None,
    ):
        self.records = list(records)
        self.xyz_root = Path(xyz_root)
        self.target_points = int(target_points)
        self.xyz_template = xyz_template
        self.spectrum_template = spectrum_template

        self.data_list = []
        bad = 0
        for ds_idx, rec in enumerate(self.records):
            try:
                mid = parse_mid_from_dens(rec["dens_npz"])
                xyz = format_record_path(self.xyz_template, mid=mid, xyz_root=self.xyz_root, rec=rec)
                spec = resolve_spectrum_path(rec, mid, spectrum_template=self.spectrum_template)
                z, pos = read_xyz(xyz)
                y = read_spectrum(spec, target_points=self.target_points)

                d = Data(
                    z=torch.from_numpy(z),
                    pos=torch.from_numpy(pos),
                    y=torch.from_numpy(y.copy()).view(1, -1),
                    ds_idx=torch.tensor([ds_idx], dtype=torch.long),
                    mid=torch.tensor([mid], dtype=torch.long),
                )
                self.data_list.append(d)
            except Exception as exc:
                bad += 1
                self.data_list.append(None)
                print(f"[WARN] failed to load ds_idx={ds_idx}: {exc}")

        if bad != 0:
            missing = sum(1 for x in self.data_list if x is None)
            raise RuntimeError(f"Preload failed for {missing} samples. Fix missing XYZ or spectrum paths first.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        d = self.data_list[i]
        return d

# =========================
# Training helpers
# =========================
def per_sample_corr(p, t, eps=1e-12):
    # p,t are numpy arrays with shape [batch, target_points]
    p0 = p - p.mean(axis=1, keepdims=True)
    t0 = t - t.mean(axis=1, keepdims=True)
    num = np.sum(p0 * t0, axis=1)
    den = np.sqrt(np.sum(p0*p0, axis=1) * np.sum(t0*t0, axis=1)) + eps
    return num / den

def train_one(model, loss_name, lr, train_loader, val_loader, epochs, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    model = model.to(device)

    criterion = LOSS_MAP[loss_name]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    best_path = outdir / "best_model.pth"

    log_path = outdir / "train_log.csv"
    with log_path.open("w") as f:
        f.write("epoch,train_loss,val_loss,val_mae,val_corr\n")

    for ep in range(1, epochs + 1):
        model.train()
        tr_sum = 0.0
        n_tr = 0
        for batch in train_loader:
            batch = batch.to(device)
            y = batch.y.view(batch.num_graphs, -1)  # [B, L]
            opt.zero_grad(set_to_none=True)
            pred = model(batch)                     # [B, L]
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
            tr_sum += float(loss.item()) * batch.num_graphs
            n_tr += batch.num_graphs
        tr_loss = tr_sum / max(1, n_tr)

        model.eval()
        va_sum = 0.0
        n_va = 0
        preds = []
        trues = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                y = batch.y.view(batch.num_graphs, -1)
                pred = model(batch)
                loss = criterion(pred, y)
                va_sum += float(loss.item()) * batch.num_graphs
                n_va += batch.num_graphs
                preds.append(pred.detach().cpu().numpy())
                trues.append(y.detach().cpu().numpy())

        va_loss = va_sum / max(1, n_va)
        P = np.concatenate(preds, axis=0)
        T = np.concatenate(trues, axis=0)
        mae = float(np.mean(np.abs(P - T)))
        corr = float(np.mean(per_sample_corr(P, T)))

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)

            # Save preds for best epoch
            np.savez(outdir / "val_preds_true.npz", pred=P.astype(np.float32), true=T.astype(np.float32))

        with log_path.open("a") as f:
            f.write(f"{ep},{tr_loss:.6f},{va_loss:.6f},{mae:.6f},{corr:.6f}\n")

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"  epoch {ep:03d} | tr {tr_loss:.6f} | va {va_loss:.6f} | mae {mae:.6f} | corr {corr:.4f}")

    return dict(best_val_loss=best_val)

# =========================
# Random config samplers
# =========================
def log_uniform(lo, hi):
    x = random.random()
    return float(10 ** (math.log10(lo) + x * (math.log10(hi) - math.log10(lo))))

def sample_schnet_cfg():
    return dict(
        hidden=random.choice([64, 128]),
        interactions=random.choice([4, 6, 8]),
        gaussians=random.choice([25, 50]),
        cutoff=random.choice([3.0, 5.0, 6.0]),
        lr=log_uniform(1e-5, 1e-3),
        loss=random.choice(["js", "kl", "wasserstein"]),
        seed=SEED,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

def sample_dimenet_cfg():
    return dict(
        hidden=random.choice([64, 128]),
        interactions=random.choice([2, 3, 4, 5]),
        gaussians=random.choice([4, 6, 8, 10]),
        cutoff=random.choice([3.0, 5.0, 6.0]),
        lr=log_uniform(1e-5, 5e-4),
        loss=random.choice(["js", "kl", "wasserstein"]),
        seed=SEED,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

# =========================
# Main
# =========================
def main():
    global VAL_FRAC, SEED, OUT_DIM, RUNS_PER_MODEL, EPOCHS, BATCH_SIZE, NUM_WORKERS, DEVICE, WEIGHT_DECAY

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-index", required=True, help="Path to index.json produced by the cache-building script.")
    ap.add_argument("--xyz-root", required=True, help="Root directory used by --xyz-template.")
    ap.add_argument("--xyz-template", default=DEFAULT_XYZ_TEMPLATE, help="Template for XYZ files. Fields: {xyz_root}, {mid}.")
    ap.add_argument("--spectrum-template", default=None, help="Optional spectrum template. Fields: {mid}, {dens_npz}, {spec_dat}.")
    ap.add_argument("--outdir", default="graph_spectrum_sweeps_4models")
    ap.add_argument("--target-points", type=int, default=DEFAULT_TARGET_POINTS)
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--runs-per-model", type=int, default=RUNS_PER_MODEL)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    VAL_FRAC = args.val_frac
    SEED = args.seed
    OUT_DIM = args.target_points
    RUNS_PER_MODEL = args.runs_per_model
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    DEVICE = torch.device(args.device)
    WEIGHT_DECAY = args.weight_decay

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] cache:", args.cache_index)
    idx = json.loads(Path(args.cache_index).read_text())
    samples = idx["samples"]
    n_samples = int(idx["num_samples"])
    if n_samples != len(samples):
        raise ValueError("index.json num_samples mismatch")

    all_idx = np.arange(n_samples, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    rng.shuffle(all_idx)
    n_val = int(round(VAL_FRAC * n_samples))
    n_val = max(1, min(n_val, n_samples - 1))
    val_idx = all_idx[:n_val]
    train_idx = all_idx[n_val:]

    split_path = out_dir / "split.json"
    split_path.write_text(json.dumps({
        "seed": SEED,
        "val_frac": VAL_FRAC,
        "target_points": OUT_DIM,
        "cache_index": args.cache_index,
        "xyz_root": args.xyz_root,
        "xyz_template": args.xyz_template,
        "spectrum_template": args.spectrum_template,
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "train_mids": [parse_mid_from_dens(samples[i]["dens_npz"]) for i in train_idx.tolist()],
        "val_mids": [parse_mid_from_dens(samples[i]["dens_npz"]) for i in val_idx.tolist()],
    }, indent=2))

    print(f"[INFO] split fixed: train={train_idx.size} val={val_idx.size} (val_frac={VAL_FRAC}, seed={SEED})")

    print("[INFO] preloading graphs + spectra...")
    full_ds = PreloadedGraphDataset(
        samples,
        Path(args.xyz_root),
        target_points=OUT_DIM,
        xyz_template=args.xyz_template,
        spectrum_template=args.spectrum_template,
    )
    print("[OK] preload complete:", len(full_ds))

    train_ds = torch.utils.data.Subset(full_ds, train_idx.tolist())
    val_ds = torch.utils.data.Subset(full_ds, val_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    summary_path = out_dir / "summary.csv"
    if not summary_path.exists():
        with summary_path.open("w") as f:
            f.write("model,exp_id,hidden,interactions,gaussians,cutoff,lr,loss,best_val_loss,exp_dir\n")

    def run_block(model_name, build_fn, cfg_sampler):
        for r in range(1, RUNS_PER_MODEL + 1):
            cfg = cfg_sampler()
            exp_id = f"exp_{r:03d}"
            exp_dir = out_dir / model_name / exp_id
            exp_dir.mkdir(parents=True, exist_ok=True)

            (exp_dir / "config.json").write_text(json.dumps({
                **cfg,
                "target_points": OUT_DIM,
                "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
            }, indent=2))
            print(f"\n=== {model_name} {exp_id} ===")
            print(cfg)

            model = build_fn(cfg)
            res = train_one(
                model=model,
                loss_name=cfg["loss"],
                lr=cfg["lr"],
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=cfg["epochs"],
                outdir=exp_dir,
            )

            with summary_path.open("a") as f:
                f.write(",".join([
                    model_name, exp_id,
                    str(cfg["hidden"]), str(cfg["interactions"]), str(cfg["gaussians"]),
                    str(cfg["cutoff"]), str(cfg["lr"]), cfg["loss"],
                    str(res["best_val_loss"]), str(exp_dir)
                ]) + "\n")

    def build_schnet(cfg):
        return SchNetSpectrum(out_dim=OUT_DIM, hidden=cfg["hidden"], interactions=cfg["interactions"],
                              gaussians=cfg["gaussians"], cutoff=cfg["cutoff"])

    def build_schnet_att(cfg):
        return SchNetSpectrum_attpoolv0(out_dim=OUT_DIM, hidden=cfg["hidden"], interactions=cfg["interactions"],
                                        gaussians=cfg["gaussians"], cutoff=cfg["cutoff"])

    def build_dimenet(cfg):
        return DimeNetPPSpectrum(out_dim=OUT_DIM, hidden=cfg["hidden"], interactions=cfg["interactions"],
                                 gaussians=cfg["gaussians"], cutoff=cfg["cutoff"])

    def build_dimenet_att(cfg):
        return DimeNetPPSpectrum_attpoolv0(out_dim=OUT_DIM, hidden=cfg["hidden"], interactions=cfg["interactions"],
                                           gaussians=cfg["gaussians"], cutoff=cfg["cutoff"])

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    run_block("schnet", build_schnet, sample_schnet_cfg)
    run_block("schnet_attpoolv0", build_schnet_att, sample_schnet_cfg)
    run_block("dimenetpp", build_dimenet, sample_dimenet_cfg)
    run_block("dimenetpp_attpoolv0", build_dimenet_att, sample_dimenet_cfg)

    print("\n[OK] Done. Summary:", summary_path)
    print("[OK] Split:", split_path)


if __name__ == "__main__":
    main()
