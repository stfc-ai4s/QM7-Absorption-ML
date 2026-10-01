"""Shared spectrum-table contract for the repository's inference scripts.

With a reference: energy_au, reference, prediction.
Without a reference: energy_au, prediction.
Energy is read from the supplied grid/reference, never inferred from output size.
"""
from pathlib import Path
import json

import numpy as np


def read_numeric_table(path):
    """Accept numeric files with # comments or one un-commented header line."""
    path = Path(path)
    skiprows = 0
    with path.open() as handle:
        for index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                [float(token) for token in stripped.split()]
            except ValueError:
                skiprows = index + 1
            break
    values = np.loadtxt(path, skiprows=skiprows, ndmin=2)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Empty or non-finite spectrum table: {path}")
    return values


def unit_sum(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name} must be a finite, non-negative one-dimensional spectrum.")
    total = float(values.sum())
    if total <= 0:
        raise ValueError(f"{name} must have positive total intensity.")
    return values / total


def prepare_spectra(prediction, length=900, reference_path=None, reference_column=-1,
                    reference_sigma=0.0, energy_grid_path=None, energy_step_au=None):
    """Select matching bins and normalise within the selected window.

    The first L output bins must correspond to the first L reference/grid rows.
    There is no interpolation, padding, or compression of a longer energy range.
    Gaussian reference smoothing, when requested, occurs before truncation, as
    in the supplied density predictor. Its width is in original reference bins.
    """
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if length < 2 or length > prediction.size:
        raise ValueError(f"Requested length {length}; model supplies {prediction.size} bins. Set --length explicitly.")
    if reference_sigma < 0:
        raise ValueError("Reference smoothing width must be non-negative.")
    reference = None
    reference_energy = None
    if reference_path is not None:
        table = read_numeric_table(reference_path)
        if len(table) < length:
            raise ValueError(f"Reference has {len(table)} rows but {length} are required.")
        if table.shape[1] == 1:
            values = table[:, 0]
        else:
            reference_energy = table[:, 0]
            values = table[:, reference_column]
        values = np.clip(values, 0.0, None)
        if reference_sigma:
            from scipy.ndimage import gaussian_filter1d
            values = gaussian_filter1d(values, sigma=reference_sigma)
        reference = unit_sum(values[:length], "Reference")

    if energy_grid_path is not None:
        energy = read_numeric_table(energy_grid_path)[:, 0]
    elif energy_step_au is not None:
        if not np.isfinite(energy_step_au) or energy_step_au <= 0:
            raise ValueError("--energy-step-au must be positive and finite.")
        energy = np.arange(length, dtype=np.float64) * energy_step_au
    elif reference_energy is not None:
        energy = reference_energy
    else:
        raise ValueError("Provide an energy/intensity reference, --energy-grid, or an explicit --energy-step-au matching training.")
    if len(energy) < length:
        raise ValueError(f"Energy grid has {len(energy)} rows but {length} are required.")
    energy = np.asarray(energy[:length], dtype=np.float64)
    if not np.isfinite(energy).all() or (np.diff(energy) <= 0).any():
        raise ValueError("Energy values must be finite and strictly increasing.")
    if reference_energy is not None and not np.allclose(energy, reference_energy[:length], rtol=1e-7, atol=1e-9):
        raise ValueError("Reference and requested energy grids disagree; supply aligned data from the training grid.")
    return energy, unit_sum(prediction[:length], "Prediction"), reference


def write_spectrum_table(path, energy, prediction, reference=None, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if reference is None:
        table = np.column_stack([energy, prediction])
        columns = "energy_au prediction"
    else:
        table = np.column_stack([energy, reference, prediction])
        columns = "energy_au reference prediction"
    header = columns
    if metadata:
        header += "\n" + json.dumps(metadata, sort_keys=True)
    np.savetxt(path, table, fmt="%.12e", header=header, comments="# ")
    return path
