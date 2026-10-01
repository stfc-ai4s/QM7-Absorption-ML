#!/usr/bin/env python3
"""
Predict an absorption spectrum from an electron-density cube.

Use any density NPZ with a single-channel `data` array, sampled with the same
voxel spacing, units and orientation convention as the training densities.
The filename does not have to contain a QM7 molecule number. The original
maximum-amplitude scaling and integer centre-of-mass alignment are retained.

The checkpoint stays outside the repository and is supplied with --model.
The first --length bins (default 900) are selected and normalised to unit sum.
Energy values come from --ref, --energy-grid, or an explicit --energy-step-au;
a longer model output is never stretched to fit the plotting window.

With a reference, output columns are energy_au, reference, prediction.
Without a reference, output columns are energy_au, prediction. The default
reference smoothing width is 10 original bins, retained from the uploaded
predictor; use --reference-sigma 0 for an already processed reference.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Keep spectrum_io importable when this file is imported as a module rather than
# run as a script, so every predictor writes the same table format.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectrum_io import prepare_spectra, write_spectrum_table


# The density grid used during training: channel, x, y, z.
DENSITY_SHAPE = (1, 155, 147, 143)

REPO_ROOT = Path(__file__).resolve().parents[2]


class CNNModel(nn.Module):
    """3D CNN definition needed when loading a model saved with torch.save(model)."""

    def __init__(
        self,
        num_layers: int = 8,
        kernel_size: int = 5,
        base_filters: int = 4,
        dropout_p: float = 0.1,
        input_shape: tuple[int, int, int, int] = DENSITY_SHAPE,
        out_features: int = 1,
    ) -> None:
        super().__init__()

        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.base_filters = int(base_filters)
        self.dropout_p = float(dropout_p)
        self.input_shape = tuple(input_shape)
        self.in_ch = int(input_shape[0])
        self.out_features = int(out_features)

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=1, padding=1)
        self.avgpool = nn.AvgPool3d(kernel_size=2, padding=1)

        padding = (self.kernel_size - 1) // 2

        def channel_multiplier(layer_index: int) -> int:
            if layer_index == 1:
                return 1
            if layer_index == 2:
                return 2
            if layer_index == 3:
                return 4
            if layer_index == 4:
                return 8
            if layer_index == 5:
                return 16
            return 32

        blocks = []
        in_channels = self.in_ch

        for layer_index in range(1, self.num_layers + 1):
            out_channels = self.base_filters * channel_multiplier(layer_index)
            out_channels = min(out_channels, self.base_filters * 32)

            blocks.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels,
                        out_channels,
                        kernel_size=self.kernel_size,
                        padding=padding,
                        padding_mode="zeros",
                    ),
                    nn.GroupNorm(out_channels, out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = out_channels

        self.blocks = nn.ModuleList(blocks)
        self.drop_mid = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        self.drop_end = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        self.flatten = nn.Flatten()
        self.linear = nn.LazyLinear(self.out_features)
        self.softmax = nn.Softmax(dim=1)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        dropout_after = max(1, math.floor(self.num_layers * 0.6))

        for layer_index, block in enumerate(self.blocks, start=1):
            x = block(x)
            x = self.avgpool(self.maxpool(x))

            if layer_index == dropout_after:
                x = self.drop_mid(x)

        return self.drop_end(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.linear(x)
        return self.softmax(x)


def load_density(npz_path: Path) -> np.ndarray:
    """Load the electron-density volume from a .npz file."""
    with np.load(npz_path, allow_pickle=False) as archive:
        if "data" not in archive.files:
            available = ", ".join(archive.files)
            raise KeyError(f"Expected key 'data' in {npz_path}. Available keys: {available}")
        density = archive["data"]

    if density.ndim == 3:
        volume = density
    elif density.ndim == 4 and density.shape[-1] == 1:
        volume = density[..., 0]
    elif density.ndim == 4 and density.shape[0] == 1:
        volume = density[0]
    else:
        raise ValueError(f"Expected a 3D density volume, got shape {density.shape}")

    if volume.size == 0 or not np.isfinite(volume).all():
        raise ValueError(f"Empty or non-finite density: {npz_path}")
    return volume.astype(np.float32, copy=False)


def centre_of_mass(volume: np.ndarray) -> np.ndarray:
    """Return the centre of mass in voxel-index coordinates."""
    weights = np.abs(volume)
    total_weight = float(weights.sum())
    nx, ny, nz = weights.shape

    if total_weight <= 0.0:
        return np.array(
            [(nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0],
            dtype=np.float64,
        )

    x_axis = np.arange(nx, dtype=np.float64)
    y_axis = np.arange(ny, dtype=np.float64)
    z_axis = np.arange(nz, dtype=np.float64)

    x_centre = float((weights.sum(axis=(1, 2)) * x_axis).sum() / total_weight)
    y_centre = float((weights.sum(axis=(0, 2)) * y_axis).sum() / total_weight)
    z_centre = float((weights.sum(axis=(0, 1)) * z_axis).sum() / total_weight)

    return np.array([x_centre, y_centre, z_centre], dtype=np.float64)


def paste_with_aligned_centre(
    target: np.ndarray,
    source: np.ndarray,
    source_centre: np.ndarray,
) -> None:
    """Paste source into target after aligning source centre to target centre."""
    target_shape = np.array(target.shape)
    source_shape = np.array(source.shape)
    target_centre = (target_shape - 1) / 2.0

    start = np.round(target_centre - source_centre).astype(int)
    target_start = np.maximum(start, 0)
    target_stop = np.minimum(start + source_shape, target_shape)

    if np.any(target_stop <= target_start):
        return

    source_start = np.maximum(0, -start)
    source_stop = source_start + (target_stop - target_start)

    tx0, ty0, tz0 = target_start
    tx1, ty1, tz1 = target_stop
    sx0, sy0, sz0 = source_start
    sx1, sy1, sz1 = source_stop

    target[tx0:tx1, ty0:ty1, tz0:tz1] = source[sx0:sx1, sy0:sy1, sz0:sz1]


def prepare_density_tensor(npz_path: Path, density_shape: tuple[int, int, int, int]) -> torch.Tensor:
    """Load, rescale, recentre, and batch the density for the CNN."""
    channels, nx, ny, nz = density_shape
    if channels != 1:
        raise ValueError("This inference script expects a single density channel.")

    volume = load_density(npz_path)

    max_abs_value = float(np.max(np.abs(volume)))
    if max_abs_value > 0.0:
        volume = volume / max_abs_value

    centred = np.zeros((nx, ny, nz), dtype=np.float32)
    paste_with_aligned_centre(centred, volume, source_centre=centre_of_mass(volume))

    # Shape expected by PyTorch Conv3d: batch, channel, x, y, z.
    return torch.from_numpy(centred[None, None, :, :, :]).float()


def infer_reference_path(npz_path: Path) -> Path | None:
    """Infer the conventional TDDFT spectrum path from density_M*.npz."""
    match = re.search(r"density_M(\d+)\.npz$", npz_path.name)
    if match is None:
        return None

    molecule_id = match.group(1)
    return npz_path.parent / f"tddft_spectrum_gamma_150meV_M{molecule_id}.dat"


def load_model(model_path: Path, device: torch.device, config_path: Path | None = None) -> nn.Module:
    """Load a trusted full model, or a state_dict with explicit CNNModel kwargs."""
    try:
        saved = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        saved = torch.load(model_path, map_location=device)
    if isinstance(saved, nn.Module):
        model = saved
    elif isinstance(saved, dict):
        config = json.loads(config_path.read_text()) if config_path else saved.get("model_kwargs")
        if isinstance(config, dict) and "model_kwargs" in config:
            config = config["model_kwargs"]
        required = {"num_layers", "kernel_size", "base_filters", "dropout_p", "input_shape", "out_features"}
        if not isinstance(config, dict) or not required.issubset(config):
            raise ValueError("State-dict checkpoints need complete CNNModel model_kwargs, in the bundle or --config JSON: " + ", ".join(sorted(required)))
        state = saved.get("state_dict", saved.get("model_state_dict", saved))
        model = CNNModel(**config)
        model.load_state_dict(state, strict=True)
    else:
        raise TypeError("Expected a full CNN model or a state_dict checkpoint with model_kwargs.")
    return model.to(device).eval()


def predict_spectrum(model: nn.Module, density_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """Run inference and return a one-dimensional spectrum."""
    with torch.no_grad():
        prediction = model(density_tensor.to(device))

    prediction = prediction.detach().cpu().numpy()

    if prediction.ndim != 2 or prediction.shape[0] != 1:
        raise RuntimeError(f"Unexpected model output shape: {prediction.shape}")

    return prediction[0].astype(np.float32, copy=False)


def build_output_path(outdir: Path, npz_path: Path) -> Path:
    """Create a readable output filename from the density filename."""
    stem = npz_path.stem.replace("density_", "")
    return outdir / f"spectrum_prediction_{stem}.dat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="External trained CNN checkpoint.")
    parser.add_argument("--config", type=Path, help="CNNModel constructor JSON for a plain state_dict.")
    parser.add_argument("--npz", required=True, type=Path, help="Any compatible single-channel density NPZ.")
    parser.add_argument("--outdir", default=REPO_ROOT / "predictions" / "density", type=Path)
    parser.add_argument("--ref", type=Path, help="Optional TDDFT reference. Otherwise check beside density_M*.npz.")
    parser.add_argument("--length", type=int, default=900, help="Number of leading model bins to use.")
    parser.add_argument("--energy-grid", type=Path, help="Text file whose first column is the model energy grid.")
    parser.add_argument("--energy-step-au", type=float, help="Explicit grid spacing from zero, if no grid file is supplied.")
    parser.add_argument("--reference-column", type=int, default=1, help="Reference intensity column (zero based).")
    parser.add_argument("--reference-sigma", type=float, default=10.0, help="Reference Gaussian width in bins; 0 disables smoothing.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def select_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu.")
    return torch.device(requested_device)


def predict_to_file(model, npz_path, outdir, device, length=900, reference_path=None,
                    energy_grid_path=None, energy_step_au=None, reference_column=1,
                    reference_sigma=10.0, model_path=None):
    """Shared entry point for individual densities and the four paper examples."""
    shape = tuple(getattr(model, "input_shape", DENSITY_SHAPE))
    density_tensor = prepare_density_tensor(Path(npz_path), shape)
    full_prediction = predict_spectrum(model, density_tensor, device)
    energy, prediction, reference = prepare_spectra(
        full_prediction, length=length, reference_path=reference_path,
        reference_column=reference_column, reference_sigma=reference_sigma,
        energy_grid_path=energy_grid_path, energy_step_au=energy_step_au,
    )
    metadata = {
        "model": str(model_path) if model_path else None,
        "density_npz": str(npz_path),
        "reference": str(reference_path) if reference_path else None,
        "model_output_bins": int(full_prediction.size), "selected_bins": int(length),
        "energy_min_au": float(energy[0]), "energy_max_au": float(energy[-1]),
        "normalization": "unit sum within selected leading bins",
        "density_preprocessing": "maximum absolute amplitude; integer centre-of-mass alignment",
        "input_shape": list(shape), "reference_sigma_bins": float(reference_sigma),
        "reference_column": int(reference_column),
    }
    path = build_output_path(Path(outdir), Path(npz_path))
    write_spectrum_table(path, energy, prediction, reference, metadata)
    return path


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model_path = args.model.expanduser().resolve()
    npz_path = args.npz.expanduser().resolve()
    model = load_model(model_path, device, args.config.expanduser().resolve() if args.config else None)
    reference = args.ref.expanduser().resolve() if args.ref else infer_reference_path(npz_path)
    if args.ref is None and reference is not None and not reference.is_file():
        reference = None
    path = predict_to_file(
        model, npz_path, args.outdir.expanduser().resolve(), device,
        length=args.length, reference_path=reference,
        energy_grid_path=args.energy_grid.expanduser().resolve() if args.energy_grid else None,
        energy_step_au=args.energy_step_au, reference_column=args.reference_column,
        reference_sigma=args.reference_sigma, model_path=model_path,
    )
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
