#!/usr/bin/env python3
"""
Predict a normalized molecular absorption spectrum with a MACE model.

Physics in brief
----------------
The input is an XYZ geometry: atomic numbers and Cartesian coordinates define the
molecular structure. The model is an equivariant MACE message-passing network
that converts this structure into learned atomic features. Only the rotationally
invariant scalar channels are pooled to a molecular representation, then a small
neural-network head predicts a normalized absorption spectrum on a fixed grid.

The output spectrum is a probability-like discretized spectrum: the predicted
intensities are non-negative and sum to one because the final layer is Softmax.
If a TDDFT reference spectrum is provided, it is read in the same way, clipped at
zero, truncated to --length leading bins, and normalized before comparison.
Its energy column supplies the grid. Without a reference, provide --energy-grid
or an explicit --energy-step-au matching training. --out-dim specifies the full
trained head size; --length specifies the evaluation window (default 900).
Place this file beside spectrum_io.py in scripts/inference/.

Typical use
-----------
python predict_mace_spectrum.py \
  --weights /path/to/best_model.pth \
  --xyz /path/to/mol.xyz \
  --reference /path/to/refernce.dat \
  --outdir mace_spectrum_prediction_M1

Outputs
-------
The output directory will contain:
  prediction_spectrum.dat    columns: energy_au, reference, prediction (or energy_au, prediction)
  prediction_spectrum.npz    NumPy arrays with the same information
  prediction_spectrum.png    quick plot, if matplotlib is installed
  prediction_metadata.json   paths, model settings, and simple comparison metrics
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

warnings.filterwarnings("ignore", message="The TorchScript type system.*")

import numpy as np
import torch
import torch.nn as nn

# Keep spectrum_io importable when this file is imported as a module rather than
# run as a script, so every predictor writes the same table format.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectrum_io import prepare_spectra, write_spectrum_table

# PyTorch 2.6 changed torch.load defaults. Some e3nn versions load a constants.pt
# file during import that contains a Python slice object. Allow-listing slice keeps
# the import path compatible without changing user checkpoint loading globally.
try:
    torch.serialization.add_safe_globals([slice])
except Exception:
    pass

from e3nn import o3
from mace import modules
from mace.data.atomic_data import AtomicData
from mace.data.utils import Configuration
from mace.modules.utils import extract_invariant
from mace.tools import AtomicNumberTable, torch_geometric
from mace.tools.scatter import scatter_mean


# -----------------------------
# Model defaults for exp01
# -----------------------------
DEFAULT_NUM_POINTS = 900
REPO_ROOT = Path(__file__).resolve().parents[2]

EXP01_HIDDEN_IRREPS = "128x0e + 128x1o"
EXP01_NUM_INTERACTIONS = 2
EXP01_MAX_ELL = 3
EXP01_DROPOUT = 0.10

# These are the training-script defaults used to build the MACE backbone.
DEFAULT_R_MAX = 5.0
DEFAULT_NUM_RADIAL_BASIS = 8
DEFAULT_NUM_CUTOFF_BASIS = 5
DEFAULT_CORRELATION = 3
DEFAULT_MLP_IRREPS = "16x0e"
DEFAULT_RADIAL_MLP = "[64, 64, 64]"
DEFAULT_RADIAL_TYPE = "bessel"
DEFAULT_DISTANCE_TRANSFORM = "None"
DEFAULT_GATE = "silu"
DEFAULT_HEAD_HIDDEN = 512
DEFAULT_AVG_NUM_NEIGHBORS = 12.0
DEFAULT_FIRST_INTERACTION = "RealAgnosticResidualInteractionBlock"
DEFAULT_INTERACTION = "RealAgnosticResidualInteractionBlock"


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


# -----------------------------
# Small utilities
# -----------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_xyz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read an XYZ file and return atomic numbers and positions."""
    with path.open("r", encoding="utf-8") as f:
        try:
            n_atoms = int(f.readline().strip())
        except Exception as exc:
            raise ValueError(f"Could not read atom count from XYZ file: {path}") from exc

        _comment = f.readline()
        atomic_numbers = np.zeros((n_atoms,), dtype=np.int64)
        positions = np.zeros((n_atoms, 3), dtype=np.float64)

        for i in range(n_atoms):
            line = f.readline()
            if not line:
                raise ValueError(f"XYZ file ended early: {path}")
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Bad XYZ atom line in {path}: {line!r}")

            symbol = parts[0]
            if symbol not in ATOM_Z:
                raise ValueError(f"Unknown element symbol {symbol!r} in {path}")

            atomic_numbers[i] = ATOM_Z[symbol]
            positions[i] = [float(parts[1]), float(parts[2]), float(parts[3])]

    return atomic_numbers, positions


def pearson_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a0 = a - float(np.mean(a))
    b0 = b - float(np.mean(b))
    denom = math.sqrt(float(np.sum(a0 * a0)) * float(np.sum(b0 * b0))) + eps
    return float(np.sum(a0 * b0) / denom)


def load_state_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Load a plain PyTorch state_dict.

    The default path is safe for a state_dict saved with torch.save(model.state_dict()).
    If the checkpoint was saved as a larger training dictionary, common wrappers are
    handled too.
    """
    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Older PyTorch versions do not have the weights_only argument.
        obj = torch.load(path, map_location=device)

    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    if not isinstance(obj, dict):
        raise TypeError(f"Expected a state_dict-like checkpoint, got {type(obj)!r}")

    state = {}
    for key, value in obj.items():
        clean_key = key[7:] if key.startswith("module.") else key
        state[clean_key] = value
    return state


def infer_atomic_numbers_from_checkpoint(state: Dict[str, torch.Tensor]) -> Optional[List[int]]:
    """Use the checkpoint's own z table when it is present."""
    z = state.get("core.atomic_numbers")
    if z is None:
        return None
    if torch.is_tensor(z):
        return [int(v) for v in z.detach().cpu().view(-1).tolist()]
    return None


def parse_atomic_numbers(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


# -----------------------------
# Data preparation
# -----------------------------
def build_atomic_data(
    xyz_path: Path,
    z_table: AtomicNumberTable,
    cutoff: float,
) -> AtomicData:
    """Convert one XYZ geometry into a MACE AtomicData object."""
    atomic_numbers, positions = read_xyz(xyz_path)

    cfg = Configuration(
        atomic_numbers=atomic_numbers,
        positions=positions,
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
        cutoff=cutoff,
        heads=["Default"],
    )
    return data


# -----------------------------
# Model
# -----------------------------
class MACEExp01Spectrum(nn.Module):
    """
    MACE spectrum architecture with module names matching the saved checkpoint.

    Important naming choice:
      - The MACE backbone is called ``core``.
      - The prediction head is a direct ``nn.Sequential`` with layers 0 and 3.

    This matches checkpoints with keys such as ``core.atomic_numbers``,
    ``core.interactions...``, ``head.0.weight`` and ``head.3.weight``.
    """

    def __init__(
        self,
        z_table: AtomicNumberTable,
        out_dim: int = DEFAULT_NUM_POINTS,
        r_max: float = DEFAULT_R_MAX,
        num_radial_basis: int = DEFAULT_NUM_RADIAL_BASIS,
        num_cutoff_basis: int = DEFAULT_NUM_CUTOFF_BASIS,
        max_ell: int = EXP01_MAX_ELL,
        correlation: int = DEFAULT_CORRELATION,
        num_interactions: int = EXP01_NUM_INTERACTIONS,
        hidden_irreps: str = EXP01_HIDDEN_IRREPS,
        mlp_irreps: str = DEFAULT_MLP_IRREPS,
        radial_mlp: str = DEFAULT_RADIAL_MLP,
        radial_type: str = DEFAULT_RADIAL_TYPE,
        distance_transform: str = DEFAULT_DISTANCE_TRANSFORM,
        gate: str = DEFAULT_GATE,
        head_hidden: int = DEFAULT_HEAD_HIDDEN,
        head_dropout: float = EXP01_DROPOUT,
        avg_num_neighbors: float = DEFAULT_AVG_NUM_NEIGHBORS,
        first_interaction: str = DEFAULT_FIRST_INTERACTION,
        interaction: str = DEFAULT_INTERACTION,
        use_agnostic_product: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.num_interactions = int(num_interactions)
        self.max_L = self.hidden_irreps.lmax
        self.num_scalar_channels = self.hidden_irreps.count(o3.Irrep(0, 1))

        atomic_energies = np.zeros((len(z_table),), dtype=np.float64)

        self.core = modules.ScaleShiftMACE(
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
            use_last_readout_only=False,
            use_embedding_readout=False,
            use_agnostic_product=use_agnostic_product,
        )

        invariant_dim = self.num_scalar_channels * self.num_interactions
        self.head = nn.Sequential(
            nn.Linear(invariant_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, out_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, batch: AtomicData) -> torch.Tensor:
        out = self.core(
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

        graph_embedding = scatter_mean(
            src=node_invariants,
            index=batch.batch,
            dim=0,
            dim_size=batch.num_graphs,
        )
        return self.head(graph_embedding)



def load_matching_model(
    state: Dict[str, torch.Tensor],
    z_table: AtomicNumberTable,
    out_dim: int,
    avg_num_neighbors: float,
    requested_first_interaction: str = "auto",
    requested_interaction: str = "auto",
) -> Tuple[MACEExp01Spectrum, Dict[str, object]]:
    """
    Build the MACE architecture that matches the saved state_dict.

    The important checkpoint clue is the first interaction block. Some MACE
    training paths use a residual first interaction for ScaleShiftMACE, while
    others explicitly use a non-residual first interaction. The tensor names are
    identical, but ``skip_tp`` tensor sizes differ. Trying both is safer than
    silently loading an incompatible model.
    """
    head3 = state.get("head.3.weight")
    head0 = state.get("head.0.weight")
    if torch.is_tensor(head3):
        out_dim = int(head3.shape[0])
    head_hidden = int(head0.shape[0]) if torch.is_tensor(head0) else DEFAULT_HEAD_HIDDEN

    first_options = (
        [requested_first_interaction]
        if requested_first_interaction != "auto"
        else ["RealAgnosticResidualInteractionBlock", "RealAgnosticInteractionBlock"]
    )
    later_options = (
        [requested_interaction]
        if requested_interaction != "auto"
        else ["RealAgnosticResidualInteractionBlock"]
    )

    tried: List[str] = []
    last_error: Optional[BaseException] = None

    max_ell_options = [EXP01_MAX_ELL, 1]
    correlation_options = [DEFAULT_CORRELATION]
    agnostic_product_options = [False, True]

    for first_interaction in first_options:
        for interaction in later_options:
            for max_ell in max_ell_options:
                for correlation in correlation_options:
                    for use_agnostic_product in agnostic_product_options:
                        settings = {
                            "first_interaction": first_interaction,
                            "interaction": interaction,
                            "hidden_irreps": EXP01_HIDDEN_IRREPS,
                            "num_interactions": EXP01_NUM_INTERACTIONS,
                            "max_ell": max_ell,
                            "correlation": correlation,
                            "use_agnostic_product": use_agnostic_product,
                            "head_hidden": head_hidden,
                        }
                        tried.append(json.dumps(settings, sort_keys=True))
                        try:
                            model = MACEExp01Spectrum(
                                z_table=z_table,
                                out_dim=out_dim,
                                avg_num_neighbors=avg_num_neighbors,
                                first_interaction=first_interaction,
                                interaction=interaction,
                                max_ell=max_ell,
                                correlation=correlation,
                                head_hidden=head_hidden,
                                use_agnostic_product=use_agnostic_product,
                            )
                            model.load_state_dict(state, strict=True)
                            print("[INFO] Loaded checkpoint with:", settings)
                            return model, settings
                        except RuntimeError as exc:
                            last_error = exc
                            continue

    print("\n[ERROR] Could not construct a model matching this checkpoint.")
    print("Tried these settings:")
    for item in tried:
        print("  -", item)
    print("\nLast PyTorch error:")
    if last_error is not None:
        print(str(last_error))
    raise RuntimeError("No matching MACE architecture found for this checkpoint.") from last_error

# -----------------------------
# Output
# -----------------------------
def write_outputs(
    outdir: Path,
    energy_grid: np.ndarray,
    prediction: np.ndarray,
    reference: Optional[np.ndarray],
    metadata: Dict,
) -> None:
    ensure_dir(outdir)

    write_spectrum_table(outdir / "prediction_spectrum.dat", energy_grid, prediction, reference, metadata)
    arrays = {"energy_au": energy_grid, "prediction": prediction}
    if reference is not None:
        arrays["reference"] = reference
    np.savez(outdir / "prediction_spectrum.npz", **arrays)

    (outdir / "prediction_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7.0, 4.0))
        plt.plot(energy_grid, prediction, label="MACE prediction")
        if reference is not None:
            plt.plot(energy_grid, reference, label="TDDFT reference", alpha=0.8)
        plt.xlabel("Frequency / energy (a.u.)")
        plt.ylabel("Normalized intensity")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(outdir / "prediction_spectrum.png", dpi=200)
        plt.close()
    except Exception as exc:
        print(f"[WARN] Could not write plot: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Predict a normalized absorption spectrum from one XYZ file using a MACE spectrum model."
    )
    ap.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the external best_model.pth checkpoint.",
    )
    ap.add_argument(
        "--xyz",
        type=str,
        required=True,
        help="Input molecular geometry in XYZ format.",
    )
    ap.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Optional TDDFT reference spectrum. Use an empty string to skip.",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(REPO_ROOT / "predictions" / "mace"),
        help="Directory where prediction files will be written.",
    )
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dim", type=int, default=DEFAULT_NUM_POINTS)
    ap.add_argument("--length", type=int, default=DEFAULT_NUM_POINTS, help="Leading output bins to retain and normalise.")
    ap.add_argument("--energy-grid", type=Path, help="Grid file; energy in its first column.")
    ap.add_argument("--energy-step-au", type=float, help="Explicit spacing from zero if no reference/grid is available.")
    ap.add_argument("--reference-column", type=int, default=-1)
    ap.add_argument("--avg-num-neighbors", type=float, default=DEFAULT_AVG_NUM_NEIGHBORS)
    ap.add_argument("--first-interaction", type=str, default="auto",
                    help="First MACE interaction block. Use auto unless reproducing a known training setting.")
    ap.add_argument("--interaction", type=str, default="auto",
                    help="Later MACE interaction block. Use auto unless reproducing a known training setting.")
    ap.add_argument(
        "--atomic-numbers",
        type=str,
        default="",
        help="Comma-separated z-table fallback, e.g. '1,6,7,8,16,17'. If omitted, the script reads core.atomic_numbers from the checkpoint.",
    )
    args = ap.parse_args()

    weights_path = Path(args.weights).expanduser().resolve()
    xyz_path = Path(args.xyz).expanduser().resolve()
    reference_path = Path(args.reference).expanduser().resolve() if args.reference else None
    outdir = Path(args.outdir).expanduser().resolve()
    device = torch.device(args.device)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not xyz_path.exists():
        raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

    state = load_state_dict(weights_path, device=torch.device("cpu"))

    atomic_numbers = infer_atomic_numbers_from_checkpoint(state)
    if atomic_numbers is None:
        if not args.atomic_numbers:
            raise ValueError(
                "Could not find core.atomic_numbers in the checkpoint. "
                "Pass --atomic-numbers, for example: --atomic-numbers 1,6,7,8,16,17"
            )
        atomic_numbers = parse_atomic_numbers(args.atomic_numbers)

    z_table = AtomicNumberTable(atomic_numbers)
    print(f"[INFO] z_table: {z_table.zs}")

    model, matched_settings = load_matching_model(
        state=state,
        z_table=z_table,
        out_dim=args.out_dim,
        avg_num_neighbors=args.avg_num_neighbors,
        requested_first_interaction=args.first_interaction,
        requested_interaction=args.interaction,
    )

    model.to(device)
    model.eval()

    data = build_atomic_data(xyz_path, z_table=z_table, cutoff=DEFAULT_R_MAX)
    loader = torch_geometric.dataloader.DataLoader([data], batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(device)

    with torch.no_grad():
        prediction = model(batch).detach().cpu().numpy().reshape(-1)

    energy_grid, prediction, reference = prepare_spectra(
        prediction, length=args.length, reference_path=reference_path,
        reference_column=args.reference_column,
        energy_grid_path=args.energy_grid.expanduser().resolve() if args.energy_grid else None,
        energy_step_au=args.energy_step_au,
    )

    metadata = {
        "model": "MACEExp01Spectrum",
        "weights": str(weights_path),
        "xyz": str(xyz_path),
        "reference": str(reference_path) if reference_path is not None else None,
        "out_dim": int(args.out_dim),
        "grid_min_au": float(energy_grid[0]),
        "grid_max_au": float(energy_grid[-1]),
        "selected_bins": int(args.length),
        "normalization": "unit sum within selected leading bins",
        "hidden_irreps": EXP01_HIDDEN_IRREPS,
        "num_interactions": EXP01_NUM_INTERACTIONS,
        "max_ell": EXP01_MAX_ELL,
        "dropout": EXP01_DROPOUT,
        "avg_num_neighbors": float(args.avg_num_neighbors),
        "matched_model_settings": matched_settings,
        "atomic_numbers": atomic_numbers,
        "prediction_sum": float(prediction.sum()),
    }

    if reference is not None:
        metadata.update(
            {
                "reference_sum": float(reference.sum()),
                "mae": float(np.mean(np.abs(prediction - reference))),
                "pearson_corr": pearson_corr(prediction, reference),
            }
        )

    write_outputs(outdir, energy_grid, prediction, reference, metadata)

    print("[OK] Wrote prediction files to:", outdir)
    print("     ", outdir / "prediction_spectrum.dat")
    print("     ", outdir / "prediction_spectrum.npz")
    print("     ", outdir / "prediction_metadata.json")
    if (outdir / "prediction_spectrum.png").exists():
        print("     ", outdir / "prediction_spectrum.png")

    if reference is not None:
        print(f"[OK] MAE={metadata['mae']:.6e}, corr={metadata['pearson_corr']:.6f}")


if __name__ == "__main__":
    main()
