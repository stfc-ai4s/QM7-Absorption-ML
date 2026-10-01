#!/usr/bin/env python3
"""
Predict an absorption spectrum from a molecular XYZ geometry using a trained graph model.

The model takes atomic numbers and Cartesian coordinates as input. It returns a
normalized absorption spectrum on a fixed energy grid. The script writes a
plain-text table containing the energy grid in atomic units and the predicted
intensity. If a reference TDDFT spectrum is provided, it is normalized in the
same way and written as an additional column.

Example
-------
python predict_spectrum_from_xyz_schnet.py \
  --model /path/to/model_full_schnet.pt \
  --xyz /path/to/mol.xyz \
  --reference /path/to/tddft_spec.dat \
  --outdir predictions

The reference file is optional. Without it, provide --energy-grid or an explicit
--energy-step-au matching training. With a reference, columns are energy_au,
reference, prediction. --length selects leading bins (default 900); longer
outputs are cropped and normalised, never stretched onto a different grid.
Place this file beside spectrum_io.py in scripts/inference/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Keep spectrum_io importable when this file is imported as a module rather than
# run as a script, so every predictor writes the same table format.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectrum_io import prepare_spectra, write_spectrum_table

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import (
        GlobalAttention,
        global_add_pool,
        global_max_pool,
        global_mean_pool,
    )
    from torch_geometric.nn.models import DimeNetPlusPlus, SchNet
except ImportError as exc:
    raise SystemExit(
        "This script needs PyTorch Geometric and its model dependencies. "
        "Install the same environment used for training before running inference."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
EPS = 1.0e-12


_SYMBOLS = [
    "X",
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
]
SYM_TO_Z = {symbol: number for number, symbol in enumerate(_SYMBOLS) if number > 0}


class CustomIdentity(nn.Module):
    """Identity layer that also accepts the extra arguments used inside SchNet."""

    def forward(self, x, *args, **kwargs):
        return x


class SchNetSpectrum(nn.Module):
    """SchNet encoder followed by a softmax spectral head."""

    def __init__(
        self,
        out_dim: int,
        hidden: int = 128,
        interactions: int = 6,
        gaussians: int = 50,
        cutoff: float = 6.0,
    ):
        super().__init__()
        self.core = SchNet(
            hidden_channels=hidden,
            num_filters=hidden,
            num_interactions=interactions,
            num_gaussians=gaussians,
            cutoff=cutoff,
            readout="add",
        )
        self.core.readout = CustomIdentity()
        self.core.lin1 = nn.Identity()
        self.core.lin2 = nn.Identity()
        self.head = nn.Sequential(nn.Linear(hidden, out_dim), nn.Softmax(dim=1))

    def forward(self, batch: Data) -> torch.Tensor:
        node_features = self.core(batch.z, batch.pos, batch.batch)
        graph_features = global_mean_pool(node_features, batch.batch)
        return self.head(graph_features)


class SchNetSpectrum_attpoolv0(nn.Module):
    """SchNet encoder with mean, max, sum, and attention pooling."""

    def __init__(
        self,
        out_dim: int,
        hidden: int = 128,
        interactions: int = 6,
        gaussians: int = 50,
        cutoff: float = 6.0,
    ):
        super().__init__()
        self.core = SchNet(
            hidden_channels=hidden,
            num_filters=hidden,
            num_interactions=interactions,
            num_gaussians=gaussians,
            cutoff=cutoff,
            readout="add",
        )
        self.core.readout = CustomIdentity()
        self.core.lin1 = nn.Identity()
        self.core.lin2 = nn.Identity()

        self.att_gate = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.att_pool = GlobalAttention(self.att_gate)
        self.head = nn.Sequential(nn.Linear(4 * hidden, out_dim), nn.Softmax(dim=1))

    def forward(self, batch: Data) -> torch.Tensor:
        node_features = self.core(batch.z, batch.pos, batch.batch)
        pooled = torch.cat(
            [
                global_mean_pool(node_features, batch.batch),
                global_max_pool(node_features, batch.batch),
                global_add_pool(node_features, batch.batch),
                self.att_pool(node_features, batch.batch),
            ],
            dim=1,
        )
        return self.head(pooled)


class DimeNetPPSpectrum(nn.Module):
    """DimeNet++ encoder followed by a softmax spectral head."""

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
            act="swish",
            output_initializer="zeros",
        )
        self.head = nn.Sequential(nn.Linear(hidden, out_dim), nn.Softmax(dim=1))

    def forward(self, batch: Data) -> torch.Tensor:
        graph_features = self.core(batch.z, batch.pos, batch.batch)
        return self.head(graph_features)


class DimeNetPPSpectrum_attpoolv0(nn.Module):
    """DimeNet++ graph embedding with an expanded gated spectral head."""

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
            act="swish",
            output_initializer="zeros",
        )
        self.gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(4 * hidden, out_dim), nn.Softmax(dim=1))

    def forward(self, batch: Data) -> torch.Tensor:
        graph_features = self.core(batch.z, batch.pos, batch.batch)
        gated = graph_features * self.gate(graph_features)
        expanded = torch.cat(
            [graph_features, torch.abs(graph_features), graph_features * graph_features, gated],
            dim=1,
        )
        return self.head(expanded)


MODEL_REGISTRY = {
    "schnet": SchNetSpectrum,
    "schnet_attpoolv0": SchNetSpectrum_attpoolv0,
    "dimenetpp": DimeNetPPSpectrum,
    "dimenetpp_attpoolv0": DimeNetPPSpectrum_attpoolv0,
}


# A full model pickled elsewhere records the module its classes came from.
# Alias those names here so torch.load can resolve them. sys.modules has no
# entry for __name__ when this file is loaded through an explicit loader
# rather than imported, so fall back to __main__ before giving up.
_this_module = sys.modules.get(__name__) or sys.modules.get("__main__")
if _this_module is not None:
    sys.modules.setdefault("convert_graph_pth_to_pt", _this_module)


def read_xyz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read atomic numbers and positions from an XYZ file."""
    lines = path.read_text().strip().splitlines()
    if len(lines) < 3:
        raise ValueError(f"Bad XYZ file: {path}")

    n_atoms = int(lines[0].strip())
    atom_lines = lines[2 : 2 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise ValueError(f"Expected {n_atoms} atoms but found {len(atom_lines)} in {path}")

    z = np.zeros(n_atoms, dtype=np.int64)
    pos = np.zeros((n_atoms, 3), dtype=np.float32)

    for i, line in enumerate(atom_lines):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Bad atom line in {path}: {line}")
        symbol = parts[0]
        if symbol not in SYM_TO_Z:
            raise ValueError(f"Unknown element symbol '{symbol}' in {path}")
        z[i] = SYM_TO_Z[symbol]
        pos[i] = [float(parts[1]), float(parts[2]), float(parts[3])]

    return z, pos


def make_graph_from_xyz(path: Path, device: torch.device) -> Data:
    """Convert one XYZ geometry into a one-molecule PyG batch."""
    z, pos = read_xyz(path)
    graph = Data(
        z=torch.from_numpy(z),
        pos=torch.from_numpy(pos),
        batch=torch.zeros(len(z), dtype=torch.long),
    )
    return graph.to(device)


def torch_load(path: Path, device: torch.device) -> Any:
    """Load a PyTorch object while allowing full-model files from trusted local runs."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def flatten_config(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Accept both plain configs and configs wrapped under a 'cfg' key."""
    cfg = dict(obj.get("cfg", obj))
    if "model" in obj and "model" not in cfg:
        cfg["model"] = obj["model"]
    if "out_dim" in obj and "out_dim" not in cfg:
        cfg["out_dim"] = obj["out_dim"]
    return cfg


def clean_model_kwargs(model_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only constructor arguments relevant to the selected model."""
    allowed = {
        "out_dim",
        "hidden",
        "interactions",
        "gaussians",
        "cutoff",
        "num_spherical",
        "envelope_exponent",
        "max_num_neighbors",
    }
    cleaned = {key: value for key, value in kwargs.items() if key in allowed and value not in (None, "")}

    if model_name.startswith("schnet"):
        cleaned.pop("num_spherical", None)
        cleaned.pop("envelope_exponent", None)
        cleaned.pop("max_num_neighbors", None)

    return cleaned


def build_model_from_bundle(bundle: Dict[str, Any], device: torch.device) -> nn.Module:
    """Build a model from a portable checkpoint bundle."""
    model_name = str(bundle.get("model_name") or bundle.get("model") or "").lower()
    model_kwargs = dict(bundle.get("model_kwargs") or {})

    if not model_name:
        cfg = flatten_config(bundle.get("config", {}))
        model_name = str(cfg.get("model", "")).lower()
        model_kwargs.update(cfg)

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            "The checkpoint bundle does not specify a supported model name. "
            f"Supported models: {', '.join(MODEL_REGISTRY)}"
        )

    model = MODEL_REGISTRY[model_name](**clean_model_kwargs(model_name, model_kwargs))
    state_dict = bundle.get("state_dict") or bundle.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint bundle does not contain a state_dict.")
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)


def load_model(path: Path, device: torch.device) -> nn.Module:
    """Load either a full model object or a portable checkpoint bundle."""
    obj = torch_load(path, device)

    if isinstance(obj, nn.Module):
        return obj.to(device)

    if isinstance(obj, dict):
        if "state_dict" in obj or "model_state_dict" in obj:
            return build_model_from_bundle(obj, device)
        raise ValueError(
            "This looks like a plain dictionary, not a full model or bundle. "
            "Save a portable bundle with model_name, model_kwargs, and state_dict."
        )

    raise TypeError(f"Unsupported model file type: {type(obj)}")


def save_plot(
    out_path: Path,
    energy_au: np.ndarray,
    prediction: np.ndarray,
    reference: Optional[np.ndarray],
) -> None:
    """Save a simple comparison plot when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is not installed; skipping plot.")
        return

    plt.figure(figsize=(7, 4))
    plt.plot(energy_au, prediction, label="prediction")
    if reference is not None:
        plt.plot(energy_au, reference, label="reference", alpha=0.8)
    plt.xlabel("Energy / atomic units")
    plt.ylabel("Normalized intensity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict a normalized absorption spectrum from one XYZ geometry."
    )
    parser.add_argument("--model", required=True, help="Path to the saved .pt model or portable bundle.")
    parser.add_argument(
        "--xyz",
        required=True,
        help="Input XYZ geometry.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional reference spectrum. If omitted, only the prediction is written.",
    )
    parser.add_argument(
        "--outdir",
        default=str(REPO_ROOT / "predictions" / "schnet"),
        help="Directory where the prediction table and plot will be written.",
    )
    parser.add_argument(
        "--outfile",
        default=None,
        help="Optional output filename for the prediction table.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only write the table; do not save a PNG plot.",
    )
    parser.add_argument("--length", type=int, default=900, help="Leading output bins to retain and normalise.")
    parser.add_argument("--energy-grid", type=Path, help="Grid file; energy in the first column.")
    parser.add_argument("--energy-step-au", type=float, help="Explicit spacing from zero if no reference/grid is available.")
    parser.add_argument("--reference-column", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model).expanduser().resolve()
    xyz_path = Path(args.xyz).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    device = torch.device(args.device)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not xyz_path.exists():
        raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

    print(f"[INFO] model: {model_path}")
    print(f"[INFO] xyz:   {xyz_path}")
    print(f"[INFO] device: {device}")

    model = load_model(model_path, device)
    model.eval()

    graph = make_graph_from_xyz(xyz_path, device)
    with torch.no_grad():
        prediction = model(graph).detach().cpu().numpy().reshape(-1).astype(np.float64)

    if prediction.size == 0:
        raise RuntimeError("Model returned an empty prediction.")

    reference_path = Path(args.reference).expanduser().resolve() if args.reference else None
    full_output_bins = int(prediction.size)
    energy_au, prediction, reference = prepare_spectra(
        prediction, length=args.length, reference_path=reference_path,
        reference_column=args.reference_column,
        energy_grid_path=args.energy_grid.expanduser().resolve() if args.energy_grid else None,
        energy_step_au=args.energy_step_au,
    )

    out_name = args.outfile or f"{xyz_path.stem}_predicted_spectrum.dat"
    table_path = outdir / out_name
    write_spectrum_table(table_path, energy_au, prediction, reference)

    metadata = {
        "model": str(model_path),
        "xyz": str(xyz_path),
        "reference": str(reference_path) if reference is not None else None,
        "output_table": str(table_path),
        "energy_min_au": float(energy_au[0]),
        "energy_max_au": float(energy_au[-1]),
        "num_points": int(prediction.size),
        "model_output_bins": full_output_bins,
        "normalization": "unit sum within selected leading bins",
    }
    metadata_path = outdir / f"{Path(out_name).stem}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    if not args.no_plot:
        plot_path = outdir / f"{Path(out_name).stem}.png"
        save_plot(plot_path, energy_au, prediction, reference)
        print(f"[OK] plot:  {plot_path}")

    print(f"[OK] table: {table_path}")
    print(f"[OK] meta:  {metadata_path}")


if __name__ == "__main__":
    main()
