#!/usr/bin/env python3
"""Run a small MACE hyperparameter sweep for molecular absorption spectra.

The model uses molecular geometries as graphs and predicts a normalized spectrum
on a fixed frequency grid. A MACE backbone produces equivariant node features;
those features are converted to rotation-invariant graph features and passed to a
softmax head. The default objective is KL divergence between the reference and
predicted normalized spectra.

This script is a downstream consumer of the cache produced by the density-to-
spectrum cache builder, for example ``scripts/training/build_density_spectrum_cache.py`` or the
original ``sweep_runner_npz_shard_900_github.py``. The cache index controls the
sample order used for the deterministic train/validation split. The density
cubes themselves are not used by this graph model, but the cache index provides
the common molecule ordering and, when available, cached spectrum shards.

Expected cache layout:

    <cache-dir>/
      index.json
      cubes_shard_0000.npy        # produced by the cache builder; not used here
      specs_shard_0000.npy        # used when --target-source cache or auto
      ...

Expected default XYZ layout:

    <xyz-root>/
      <molecule-id>/
        GEO_M<molecule-id>.xyz

Run example using the cached spectrum shards:

    python scripts/training/train_mace_spectrum_sweep_from_cache.py \
      --cache-index /path/to/cache-dir/index.json \
      --xyz-root /path/to/xyz-root \
      --outdir /path/to/mace-spectrum-sweep \
      --target-source cache

Run example using raw spectrum files recorded in the cache index:

    python scripts/training/train_mace_spectrum_sweep_from_cache.py \
      --cache-index /path/to/cache-dir/index.json \
      --xyz-root /path/to/xyz-root \
      --outdir /path/to/mace-spectrum-sweep \
      --target-source raw

For a different directory structure, pass templates instead of editing the code:

    --xyz-template "{xyz_root}/molecules/{mid}/geometry.xyz"
    --spectrum-template "/path/to/spectra/{mid}/spectrum.dat"

Template fields available to both templates are ``{mid}``, ``{molecule_id}``,
``{xyz_root}``, ``{cache_dir}``, ``{dens_npz}``, ``{spec_dat}``, and
``{spectrum_tag}``. If the cache was moved after creation, cached shard paths are
resolved relative to the directory containing ``index.json`` when possible.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from ase import Atoms
from e3nn import o3
from torch.utils.data import Dataset

# Compatibility patch for PyTorch 2.6+ when loading e3nn/MACE constants.
try:
    from torch.serialization import add_safe_globals

    add_safe_globals([slice])
except Exception:
    pass

import mace
from mace import modules

try:
    from mace.data import AtomicData
except Exception:
    from mace.data.atomic_data import AtomicData

try:
    from mace.data.utils import KeySpecification, config_from_atoms
except Exception:
    from mace.data.utils import KeySpecification, config_from_atoms

try:
    from mace.tools import AtomicNumberTable
except Exception:
    from mace.tools.utils import AtomicNumberTable

try:
    from mace.tools import torch_geometric as mace_tg
except Exception:
    import torch_geometric as mace_tg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_TARGET_POINTS = 900
DEFAULT_XYZ_TEMPLATE = "{xyz_root}/{mid}/GEO_M{mid}.xyz"
DENSITY_NAME_RE = re.compile(r"density_M(\d+)\.npz$")

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


@dataclass
class RunConfig:
    cache_index: str
    xyz_root: str
    outdir: str = "mace_spectrum_sweep"

    xyz_template: str = DEFAULT_XYZ_TEMPLATE
    spectrum_template: Optional[str] = None
    spectrum_tag: str = ""
    target_source: str = "auto"
    target_points: int = DEFAULT_TARGET_POINTS
    clip_neg: bool = True
    skiprows: int = 1

    seed: int = 0
    val_frac: float = 0.15
    train_fraction: float = 1.0

    epochs: int = 400
    batch_size: int = 8
    lr: float = 8e-5
    weight_decay: float = 1e-6
    num_workers: int = 0

    r_max: float = 5.0
    num_bessel: int = 8
    num_polynomial_cutoff: int = 5
    max_ell: int = 3
    num_interactions: int = 2
    hidden_irreps: str = "128x0e"
    mlp_irreps: str = "16x0e"
    correlation: int = 3
    radial_mlp: Tuple[int, ...] = (64, 64, 64)
    dropout: float = 0.10


@dataclass
class ExperimentConfig:
    name: str
    hidden_irreps: str
    num_interactions: int = 2
    max_ell: int = 3
    lr: Optional[float] = None
    dropout: Optional[float] = None
    mlp_irreps: Optional[str] = None
    note: str = ""


def default_experiments() -> List[ExperimentConfig]:
    """A compact sweep around scalar, vector, and tensor hidden features."""
    return [
        ExperimentConfig(
            name="exp01_mix_128x0e_128x1o",
            hidden_irreps="128x0e + 128x1o",
            note="scalar and vector channels",
        ),
        ExperimentConfig(
            name="exp02_mix_128x0e_128x1o_int3",
            hidden_irreps="128x0e + 128x1o",
            num_interactions=3,
            note="deeper message passing",
        ),
        ExperimentConfig(
            name="exp03_mix_128x0e_128x1o_int4",
            hidden_irreps="128x0e + 128x1o",
            num_interactions=4,
            note="deeper message passing with four interaction blocks",
        ),
        ExperimentConfig(
            name="exp04_mix_128x0e_64x1o_64x2e",
            hidden_irreps="128x0e + 64x1o + 64x2e",
            note="adds tensor channels while keeping scalar-dominant capacity",
        ),
        ExperimentConfig(
            name="exp05_mix_128x0e_128x1o_64x2e",
            hidden_irreps="128x0e + 128x1o + 64x2e",
            note="strong vector channels with some tensor capacity",
        ),
        ExperimentConfig(
            name="exp06_mix_192x0e_192x1o",
            hidden_irreps="192x0e + 192x1o",
            note="wider scalar and vector model",
        ),
        ExperimentConfig(
            name="exp07_mix_96x0e_96x1o_96x2e",
            hidden_irreps="96x0e + 96x1o + 96x2e",
            note="balanced scalar, vector, and tensor channels",
        ),
        ExperimentConfig(
            name="exp08_scalar_128x0e_control",
            hidden_irreps="128x0e",
            note="scalar-only control",
        ),
        ExperimentConfig(
            name="exp09_mix_128x0e_128x1o_lr5e5",
            hidden_irreps="128x0e + 128x1o",
            lr=5e-5,
            note="lower learning rate",
        ),
        ExperimentConfig(
            name="exp10_mix_128x0e_128x1o_drop20",
            hidden_irreps="128x0e + 128x1o",
            dropout=0.20,
            note="stronger dropout",
        ),
    ]


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Run a MACE spectrum sweep using the sample order from a cache index."
    )
    parser.add_argument("--cache-index", required=True, help="Path to index.json produced by the cache builder.")
    parser.add_argument("--xyz-root", required=True, help="Root directory used by the XYZ path template.")
    parser.add_argument("--outdir", default=RunConfig.outdir, help="Directory for sweep outputs.")
    parser.add_argument("--xyz-template", default=RunConfig.xyz_template)
    parser.add_argument("--spectrum-template", default=None)
    parser.add_argument(
        "--spectrum-tag",
        default="",
        help="Optional tag used to rewrite gamma_<tag> filenames when reading raw spectrum files.",
    )
    parser.add_argument(
        "--target-source",
        choices=["auto", "cache", "raw"],
        default=RunConfig.target_source,
        help="auto uses cached spectrum shards when available; raw reads .dat files.",
    )
    parser.add_argument("--target-points", type=int, default=RunConfig.target_points)
    parser.add_argument("--clip-neg", action="store_true", default=RunConfig.clip_neg)
    parser.add_argument("--no-clip-neg", action="store_false", dest="clip_neg")
    parser.add_argument("--skiprows", type=int, default=RunConfig.skiprows)
    parser.add_argument("--seed", type=int, default=RunConfig.seed)
    parser.add_argument("--val-frac", type=float, default=RunConfig.val_frac)
    parser.add_argument("--train-fraction", type=float, default=RunConfig.train_fraction)
    parser.add_argument("--epochs", type=int, default=RunConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=RunConfig.batch_size)
    parser.add_argument("--lr", type=float, default=RunConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=RunConfig.weight_decay)
    parser.add_argument("--num-workers", type=int, default=RunConfig.num_workers)
    parser.add_argument("--r-max", type=float, default=RunConfig.r_max)
    parser.add_argument("--num-bessel", type=int, default=RunConfig.num_bessel)
    parser.add_argument("--num-polynomial-cutoff", type=int, default=RunConfig.num_polynomial_cutoff)
    parser.add_argument("--max-ell", type=int, default=RunConfig.max_ell)
    parser.add_argument("--num-interactions", type=int, default=RunConfig.num_interactions)
    parser.add_argument("--hidden-irreps", type=str, default=RunConfig.hidden_irreps)
    parser.add_argument("--mlp-irreps", type=str, default=RunConfig.mlp_irreps)
    parser.add_argument("--correlation", type=int, default=RunConfig.correlation)
    parser.add_argument("--radial-mlp", type=int, nargs="+", default=list(RunConfig.radial_mlp))
    parser.add_argument("--dropout", type=float, default=RunConfig.dropout)
    args = parser.parse_args()

    return RunConfig(
        cache_index=args.cache_index,
        xyz_root=args.xyz_root,
        outdir=args.outdir,
        xyz_template=args.xyz_template,
        spectrum_template=args.spectrum_template,
        spectrum_tag=args.spectrum_tag,
        target_source=args.target_source,
        target_points=args.target_points,
        clip_neg=args.clip_neg,
        skiprows=args.skiprows,
        seed=args.seed,
        val_frac=args.val_frac,
        train_fraction=args.train_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        r_max=args.r_max,
        num_bessel=args.num_bessel,
        num_polynomial_cutoff=args.num_polynomial_cutoff,
        max_ell=args.max_ell,
        num_interactions=args.num_interactions,
        hidden_irreps=args.hidden_irreps,
        mlp_irreps=args.mlp_irreps,
        correlation=args.correlation,
        radial_mlp=tuple(args.radial_mlp),
        dropout=args.dropout,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_molecule_id_from_density_path(density_path: str) -> int:
    match = DENSITY_NAME_RE.search(density_path)
    if not match:
        raise ValueError(f"Cannot parse molecule id from density path: {density_path}")
    return int(match.group(1))


def format_path_template(
    template: str,
    *,
    mid: int,
    config: RunConfig,
    record: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Format a molecule-specific file path from a user-provided template."""
    record = record or {}
    return Path(
        template.format(
            mid=mid,
            molecule_id=mid,
            xyz_root=config.xyz_root,
            cache_dir=str(cache_dir or Path(config.cache_index).parent),
            dens_npz=record.get("dens_npz", ""),
            spec_dat=record.get("spec_dat", ""),
            spectrum_tag=config.spectrum_tag,
        )
    )


def resolve_cache_file(path_text: str, cache_index_path: Path) -> Path:
    """Resolve a path stored in index.json, including moved-cache cases."""
    original = Path(path_text)
    if original.exists():
        return original

    candidates = []
    if not original.is_absolute():
        candidates.append(cache_index_path.parent / original)
    candidates.append(cache_index_path.parent / original.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve cache file '{path_text}'. Tried the stored path and paths relative to {cache_index_path.parent}."
    )


def read_xyz_geometry(config: RunConfig, mid: int, record: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
    path = format_path_template(
        config.xyz_template,
        mid=mid,
        config=config,
        record=record,
        cache_dir=Path(config.cache_index).parent,
    )
    with path.open("r", encoding="utf-8") as handle:
        n_atoms = int(handle.readline().strip())
        _ = handle.readline()
        z = np.zeros((n_atoms,), dtype=np.int64)
        pos = np.zeros((n_atoms, 3), dtype=np.float32)
        for atom_index in range(n_atoms):
            parts = handle.readline().split()
            symbol = parts[0]
            if symbol not in ATOM_Z:
                raise ValueError(f"Unknown element '{symbol}' in {path}")
            z[atom_index] = ATOM_Z[symbol]
            pos[atom_index] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return z, pos


def resolve_raw_spectrum_path(config: RunConfig, record: Dict[str, Any], mid: int) -> Path:
    if config.spectrum_template:
        return format_path_template(
            config.spectrum_template,
            mid=mid,
            config=config,
            record=record,
            cache_dir=Path(config.cache_index).parent,
        )

    path_text = str(record["spec_dat"])
    if config.spectrum_tag:
        path_text = re.sub(r"gamma_[^/\\]+_", f"gamma_{config.spectrum_tag}_", path_text)
    return Path(path_text)


def load_raw_spectrum(path: Path, target_points: int, skiprows: int, clip_neg: bool) -> np.ndarray:
    arr = np.loadtxt(path, skiprows=skiprows)
    y = arr if arr.ndim == 1 else arr[:, -1]
    y = y.astype(np.float32, copy=False)
    return normalize_and_trim_spectrum(y, target_points=target_points, clip_neg=clip_neg)


def normalize_and_trim_spectrum(y: np.ndarray, target_points: int, clip_neg: bool) -> np.ndarray:
    if y.shape[0] < target_points:
        padded = np.zeros((target_points,), dtype=np.float32)
        padded[: y.shape[0]] = y
        y = padded
    else:
        y = y[:target_points]

    if clip_neg:
        y = np.clip(y, 0.0, None)

    total = float(y.sum())
    if total <= 0.0 or not np.isfinite(total):
        y[:] = 1.0 / float(target_points)
    else:
        y = y / total
    return y.astype(np.float32, copy=False)


def load_cached_spectrum(
    record: Dict[str, Any],
    cache_index_path: Path,
    target_points: int,
    clip_neg: bool,
    shard_cache: Dict[Path, np.ndarray],
) -> np.ndarray:
    if "spec_shard" not in record or "offset" not in record:
        raise KeyError("Cache index sample does not contain spec_shard and offset fields.")

    shard_path = resolve_cache_file(str(record["spec_shard"]), cache_index_path)
    shard = shard_cache.get(shard_path)
    if shard is None:
        shard = np.load(shard_path, mmap_mode="r")
        shard_cache[shard_path] = shard

    offset = int(record["offset"])
    y = np.asarray(shard[offset], dtype=np.float32)
    return normalize_and_trim_spectrum(y, target_points=target_points, clip_neg=clip_neg)


def select_target_source(config: RunConfig, samples: Sequence[Dict[str, Any]], cache_index_path: Path) -> str:
    if config.target_source != "auto":
        return config.target_source

    first = samples[0]
    if "spec_shard" not in first or "offset" not in first:
        return "raw"

    try:
        resolve_cache_file(str(first["spec_shard"]), cache_index_path)
        return "cache"
    except FileNotFoundError:
        return "raw"


class KLLossProb(nn.Module):
    """KL(q || p), where p is the prediction and q is the reference spectrum."""

    def forward(self, p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        p = torch.clamp(p, eps, 1.0)
        q = torch.clamp(q, eps, 1.0)
        return torch.sum(q * (torch.log(q) - torch.log(p)), dim=1).mean()


def pearson_per_sample(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    pred0 = pred - pred.mean(axis=1, keepdims=True)
    true0 = true - true.mean(axis=1, keepdims=True)
    numerator = np.sum(pred0 * true0, axis=1)
    denominator = np.sqrt(np.sum(pred0 * pred0, axis=1) * np.sum(true0 * true0, axis=1)) + eps
    return numerator / denominator


def mean_pool_nodes(x: torch.Tensor, batch: torch.Tensor, num_graphs: Optional[int] = None) -> torch.Tensor:
    if num_graphs is None:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0

    pooled = x.new_zeros((num_graphs, x.size(-1)))
    pooled.index_add_(0, batch, x)

    counts = x.new_zeros((num_graphs,))
    counts.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
    return pooled / counts.clamp_min(1).unsqueeze(-1)


class IrrepInvariantProjector(nn.Module):
    """Convert e3nn irrep features into invariant per-multiplicity features."""

    def __init__(self, irreps: o3.Irreps):
        super().__init__()
        self.irreps = o3.Irreps(irreps).remove_zero_multiplicities().simplify()
        self.slices = self.irreps.slices()
        self.out_dim = self.irreps.num_irreps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for (mul, irrep), feature_slice in zip(self.irreps, self.slices):
            block = x[:, feature_slice].reshape(x.shape[0], mul, irrep.dim)
            if irrep.l == 0:
                parts.append(block.reshape(x.shape[0], mul))
            else:
                parts.append(torch.linalg.norm(block, dim=-1))

        if not parts:
            return x.new_zeros((x.shape[0], 0))
        return torch.cat(parts, dim=-1)


class MACEAtomicDataset(Dataset):
    """Preloaded molecular graphs and normalized spectrum targets."""

    def __init__(
        self,
        cache_index: Dict[str, Any],
        targets: np.ndarray,
        sample_indices: Sequence[int],
        config: RunConfig,
        z_table: Any,
        keyspec: KeySpecification,
    ) -> None:
        self.samples = cache_index["samples"]
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.data_list: List[Any] = []

        for sample_index in self.sample_indices:
            record = self.samples[int(sample_index)]
            mid = parse_molecule_id_from_density_path(record["dens_npz"])
            z, pos = read_xyz_geometry(config, mid, record)
            y = targets[int(sample_index)]

            atoms = Atoms(
                numbers=z.tolist(),
                positions=np.asarray(pos, dtype=np.float32),
                pbc=[False, False, False],
                cell=np.zeros((3, 3), dtype=np.float32),
            )
            mace_config = config_from_atoms(
                atoms,
                key_specification=keyspec,
                head_name="Default",
            )
            data = AtomicData.from_config(
                config=mace_config,
                z_table=z_table,
                cutoff=config.r_max,
                heads=["Default"],
            )
            data.y = torch.from_numpy(y.copy()).float().view(1, -1)
            data.ds_idx = torch.tensor([int(sample_index)], dtype=torch.long)
            data.mid = torch.tensor([int(mid)], dtype=torch.long)
            self.data_list.append(data)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int):
        return self.data_list[idx]


class MACEForSpectrum(nn.Module):
    """MACE backbone followed by an invariant graph-level spectrum head."""

    def __init__(self, config: RunConfig, z_table: Any, avg_num_neighbors: float, out_dim: int) -> None:
        super().__init__()
        self.hidden_irreps = o3.Irreps(config.hidden_irreps).remove_zero_multiplicities().simplify()
        self.mlp_irreps = o3.Irreps(config.mlp_irreps).remove_zero_multiplicities().simplify()

        self.core = modules.ScaleShiftMACE(
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
            r_max=config.r_max,
            num_bessel=config.num_bessel,
            num_polynomial_cutoff=config.num_polynomial_cutoff,
            max_ell=config.max_ell,
            interaction_cls=modules.RealAgnosticResidualInteractionBlock,
            interaction_cls_first=modules.RealAgnosticResidualInteractionBlock,
            num_interactions=config.num_interactions,
            num_elements=len(z_table),
            hidden_irreps=self.hidden_irreps,
            MLP_irreps=self.mlp_irreps,
            atomic_energies=np.zeros(len(z_table), dtype=np.float32),
            avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=list(z_table.zs) if hasattr(z_table, "zs") else list(z_table),
            correlation=config.correlation,
            gate=torch.nn.functional.silu,
            radial_MLP=list(config.radial_mlp),
            radial_type="bessel",
            distance_transform="None",
            heads=["Default"],
        )

        self.to_invariants = IrrepInvariantProjector(self.hidden_irreps)
        self.head = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(512, out_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, batch) -> torch.Tensor:
        out = self.core(
            batch.to_dict(),
            training=self.training,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
        )
        if "node_feats" in out:
            node_features = out["node_feats"]
        elif "node_features" in out:
            node_features = out["node_features"]
        else:
            raise KeyError(f"MACE output does not contain node features. Available keys: {sorted(out.keys())}")

        invariant_nodes = self.to_invariants(node_features)
        graph_features = mean_pool_nodes(invariant_nodes, batch.batch, num_graphs=batch.num_graphs)
        return self.head(graph_features)


def estimate_avg_num_neighbors(dataset: Dataset) -> float:
    values = []
    for graph in dataset:
        num_nodes = int(graph.node_attrs.shape[0])
        num_edges = int(graph.edge_index.shape[1])
        values.append(num_edges / max(1, num_nodes))
    return float(np.mean(values))


def build_targets(config: RunConfig, cache_index: Dict[str, Any], out_root: Path) -> Tuple[np.ndarray, str]:
    samples = cache_index["samples"]
    n_samples = int(cache_index["num_samples"])
    cache_index_path = Path(config.cache_index)
    source = select_target_source(config, samples, cache_index_path)
    target_cache = out_root / f"targets_{source}_{config.target_points}pts.npz"

    if target_cache.exists():
        data = np.load(target_cache)
        targets = data["targets"].astype(np.float32)
        print(f"[INFO] loaded targets: {target_cache} {targets.shape}")
        return targets, source

    targets = np.zeros((n_samples, config.target_points), dtype=np.float32)
    shard_cache: Dict[Path, np.ndarray] = {}
    bad_records = []

    for row, record in enumerate(samples):
        try:
            mid = parse_molecule_id_from_density_path(record["dens_npz"])
            if source == "cache":
                targets[row] = load_cached_spectrum(
                    record,
                    cache_index_path=cache_index_path,
                    target_points=config.target_points,
                    clip_neg=config.clip_neg,
                    shard_cache=shard_cache,
                )
            else:
                spec_path = resolve_raw_spectrum_path(config, record, mid)
                targets[row] = load_raw_spectrum(
                    spec_path,
                    target_points=config.target_points,
                    skiprows=config.skiprows,
                    clip_neg=config.clip_neg,
                )
        except Exception as exc:
            bad_records.append((row, str(exc)))
            targets[row] = 1.0 / float(config.target_points)

    if len(bad_records) == n_samples:
        preview = "\n".join(f"  sample {i}: {msg}" for i, msg in bad_records[:5])
        raise RuntimeError(f"No targets could be loaded. First failures:\n{preview}")

    np.savez(target_cache, targets=targets)
    print(f"[INFO] wrote targets: {target_cache} bad={len(bad_records)} source={source}")

    if bad_records:
        failure_path = out_root / f"target_load_failures_{source}.txt"
        failure_path.write_text("\n".join(f"{i}: {msg}" for i, msg in bad_records), encoding="utf-8")
        print(f"[WARN] target load failures written to: {failure_path}")

    return targets, source


def build_fixed_split(config: RunConfig, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    sample_indices = np.arange(n_samples, dtype=np.int64)
    rng = np.random.default_rng(config.seed)
    rng.shuffle(sample_indices)

    n_val = int(round(config.val_frac * n_samples))
    n_val = max(1, min(n_val, n_samples - 1))
    val_idx = sample_indices[:n_val]
    train_pool = sample_indices[n_val:]

    n_train = int(round(config.train_fraction * train_pool.size))
    n_train = max(1, min(n_train, train_pool.size))
    train_idx = train_pool[:n_train]
    return train_idx, val_idx


def build_z_table(
    samples: Sequence[Dict[str, Any]],
    config: RunConfig,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Any:
    atomic_numbers: List[int] = []
    for sample_index in np.concatenate([train_idx, val_idx]):
        record = samples[int(sample_index)]
        mid = parse_molecule_id_from_density_path(record["dens_npz"])
        z, _ = read_xyz_geometry(config, mid, record)
        atomic_numbers.extend(z.tolist())
    return AtomicNumberTable(sorted(set(atomic_numbers)))


def make_loaders(train_ds: Dataset, val_ds: Dataset, batch_size: int, num_workers: int):
    train_loader = mace_tg.dataloader.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    val_loader = mace_tg.dataloader.DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    train: bool,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    model.train(train)

    total_loss = 0.0
    total_n = 0
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []

    for batch in loader:
        batch = batch.to(DEVICE)
        y = batch.y.view(batch.num_graphs, -1)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            pred = model(batch)
            loss = loss_fn(pred, y)
            if train:
                loss.backward()
                optimizer.step()

        batch_size = batch.num_graphs
        total_loss += float(loss.item()) * batch_size
        total_n += batch_size
        preds.append(pred.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())

    pred_np = np.concatenate(preds, axis=0)
    true_np = np.concatenate(trues, axis=0)
    mean_loss = total_loss / max(1, total_n)
    mae = float(np.mean(np.abs(pred_np - true_np)))
    corr = float(np.mean(pearson_per_sample(pred_np, true_np)))
    return mean_loss, mae, corr, pred_np, true_np


def apply_experiment(base: RunConfig, experiment: ExperimentConfig) -> RunConfig:
    config = replace(base)
    config.hidden_irreps = experiment.hidden_irreps
    config.num_interactions = experiment.num_interactions
    config.max_ell = experiment.max_ell
    if experiment.lr is not None:
        config.lr = experiment.lr
    if experiment.dropout is not None:
        config.dropout = experiment.dropout
    if experiment.mlp_irreps is not None:
        config.mlp_irreps = experiment.mlp_irreps
    return config


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    def convert(obj: Any):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, default=convert), encoding="utf-8")


def train_one_experiment(
    run_dir: Path,
    base_config: RunConfig,
    experiment: ExperimentConfig,
    z_table: Any,
    avg_num_neighbors: float,
    train_loader,
    val_loader,
) -> Dict[str, Any]:
    config = apply_experiment(base_config, experiment)
    set_seed(base_config.seed)

    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", asdict(config))
    save_json(run_dir / "experiment.json", asdict(experiment))

    model = MACEForSpectrum(
        config=config,
        z_table=z_table,
        avg_num_neighbors=avg_num_neighbors,
        out_dim=config.target_points,
    ).to(DEVICE)

    loss_fn = KLLossProb()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    log_path = run_dir / "train_log.csv"
    best_path = run_dir / "best_model.pth"
    preds_path = run_dir / "val_preds_true.npz"
    metrics_path = run_dir / "best_metrics.json"

    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_mae", "train_corr", "val_loss", "val_mae", "val_corr"])

    best_val = float("inf")
    best_metrics: Dict[str, Any] = {}

    for epoch in range(1, config.epochs + 1):
        tr_loss, tr_mae, tr_corr, _, _ = run_epoch(model, train_loader, loss_fn, optimizer, train=True)
        va_loss, va_mae, va_corr, pred_np, true_np = run_epoch(model, val_loader, loss_fn, optimizer, train=False)

        with log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    epoch,
                    f"{tr_loss:.8f}",
                    f"{tr_mae:.8f}",
                    f"{tr_corr:.8f}",
                    f"{va_loss:.8f}",
                    f"{va_mae:.8f}",
                    f"{va_corr:.8f}",
                ]
            )

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)
            np.savez(preds_path, pred=pred_np.astype(np.float32), true=true_np.astype(np.float32))
            best_metrics = {
                "experiment": experiment.name,
                "best_epoch": epoch,
                "best_val_loss": float(va_loss),
                "best_val_mae": float(va_mae),
                "best_val_corr": float(va_corr),
                "hidden_irreps": config.hidden_irreps,
                "num_interactions": config.num_interactions,
                "max_ell": config.max_ell,
                "lr": config.lr,
                "dropout": config.dropout,
            }
            save_json(metrics_path, best_metrics)

        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(
                f"[{experiment.name}] epoch {epoch:03d} | "
                f"train {tr_loss:.6f} | val {va_loss:.6f} | "
                f"val_mae {va_mae:.6f} | val_corr {va_corr:.4f}"
            )

    if not best_metrics:
        raise RuntimeError(f"Experiment {experiment.name} finished without recording metrics.")

    return {
        "status": "ok",
        **best_metrics,
        "run_dir": str(run_dir),
        "best_model": str(best_path),
        "train_log": str(log_path),
        "predictions": str(preds_path),
    }


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    out_root = Path(config.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    cache_path = Path(config.cache_index)
    cache_index = json.loads(cache_path.read_text(encoding="utf-8"))
    samples = cache_index["samples"]
    n_samples = int(cache_index["num_samples"])
    if n_samples != len(samples):
        raise ValueError("index.json num_samples does not match the number of samples entries.")

    print("device:", DEVICE)
    print("torch:", torch.__version__)
    print("mace:", getattr(mace, "__version__", "unknown"))
    print("output:", out_root)
    print("samples:", n_samples)

    save_json(out_root / "base_config.json", asdict(config))

    experiments = default_experiments()
    save_json(out_root / "experiments.json", {"experiments": [asdict(item) for item in experiments]})

    targets, target_source = build_targets(config, cache_index, out_root)
    train_idx, val_idx = build_fixed_split(config, n_samples)
    print(f"target_source: {target_source}")
    print(f"split: train={train_idx.size} val={val_idx.size} seed={config.seed} val_frac={config.val_frac}")

    save_json(
        out_root / "split.json",
        {
            "seed": config.seed,
            "val_frac": config.val_frac,
            "train_fraction": config.train_fraction,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "cache_index": config.cache_index,
            "target_source": target_source,
        },
    )

    z_table = build_z_table(samples, config, train_idx, val_idx)
    print("z_table:", list(z_table.zs) if hasattr(z_table, "zs") else z_table)

    keyspec = KeySpecification.from_defaults()
    train_ds = MACEAtomicDataset(cache_index, targets, train_idx, config, z_table, keyspec)
    val_ds = MACEAtomicDataset(cache_index, targets, val_idx, config, z_table, keyspec)
    print("graphs:", "train", len(train_ds), "val", len(val_ds))

    avg_num_neighbors = estimate_avg_num_neighbors(train_ds)
    print("avg_num_neighbors:", avg_num_neighbors)

    train_loader, val_loader = make_loaders(
        train_ds,
        val_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    summary_path = out_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "experiment",
                "status",
                "best_epoch",
                "best_val_loss",
                "best_val_mae",
                "best_val_corr",
                "hidden_irreps",
                "num_interactions",
                "max_ell",
                "lr",
                "dropout",
                "run_dir",
                "best_model",
                "error",
            ]
        )

    for index, experiment in enumerate(experiments, start=1):
        run_dir = out_root / f"{index:02d}_{experiment.name}"
        print("=" * 88)
        print(f"Starting {index:02d}/{len(experiments)}: {experiment.name}")
        print(
            "hidden_irreps=", experiment.hidden_irreps,
            "num_interactions=", experiment.num_interactions,
            "max_ell=", experiment.max_ell,
        )
        if experiment.note:
            print("note:", experiment.note)

        try:
            result = train_one_experiment(
                run_dir=run_dir,
                base_config=config,
                experiment=experiment,
                z_table=z_table,
                avg_num_neighbors=avg_num_neighbors,
                train_loader=train_loader,
                val_loader=val_loader,
            )
        except Exception as exc:
            failure_text = traceback.format_exc()
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "failure.txt").write_text(failure_text, encoding="utf-8")
            result = {
                "experiment": experiment.name,
                "status": "failed",
                "best_epoch": "",
                "best_val_loss": "",
                "best_val_mae": "",
                "best_val_corr": "",
                "hidden_irreps": experiment.hidden_irreps,
                "num_interactions": experiment.num_interactions,
                "max_ell": experiment.max_ell,
                "lr": experiment.lr if experiment.lr is not None else config.lr,
                "dropout": experiment.dropout if experiment.dropout is not None else config.dropout,
                "run_dir": str(run_dir),
                "best_model": "",
                "error": str(exc),
            }
            print(f"Experiment failed: {experiment.name}")
            print(exc)

        with summary_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    result.get("experiment", experiment.name),
                    result.get("status", "unknown"),
                    result.get("best_epoch", ""),
                    result.get("best_val_loss", ""),
                    result.get("best_val_mae", ""),
                    result.get("best_val_corr", ""),
                    result.get("hidden_irreps", ""),
                    result.get("num_interactions", ""),
                    result.get("max_ell", ""),
                    result.get("lr", ""),
                    result.get("dropout", ""),
                    result.get("run_dir", ""),
                    result.get("best_model", ""),
                    result.get("error", ""),
                ]
            )

    print("=" * 88)
    print("Sweep finished.")
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
