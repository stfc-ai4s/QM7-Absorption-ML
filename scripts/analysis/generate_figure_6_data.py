#!/usr/bin/env python3
"""Generate the Pearson-only chemistry summary behind Figures 6 and 7.

Dependencies: numpy, pandas, rdkit.

All of these inputs are committed, so the default invocation reproduces the
summary CSV without any download. They live under data/figure_6/ because that
is where the analysis originated; the same CSV drives both figures.

  data/figure_6/val_preds_true.npz     pred/true/val_idx from the CNN run
  data/figure_6/indices.json           train_idx/val_idx of the same run
  data/figure_6/cache_index.json       the training cache's index.json
  data/figure_6/validation_molecule_numbers_ordered_with_smiles.csv
  data/figure_6/xyz/<number>/GEO_M<number>.xyz

To rebuild these from a training run of your own, copy val_preds_true.npz and
indices.json out of the chosen experiment directory, copy the cache's
index.json as cache_index.json keeping its complete samples list and original
ordering, and copy the numbered XYZ subfolders. Only validation molecules need
XYZ files here.

The NPZ must contain pred, true, and val_idx; all saved spectral bins are used.
The cache index's samples[val_idx[i]]['dens_npz'] identifies
density_M<number>.npz. Density and spectrum paths recorded inside this index
are not opened, so their original absolute paths can remain unchanged.

Run from the repository root with no additional arguments:
  python scripts/analysis/generate_figure_6_data.py

Defaults are anchored to this script's location, so running its absolute
path from another working directory also works. Optional --run-dir,
--cache-index, --xyz-root, --smiles-csv, and --output override these defaults;
explicit relative CLI paths are resolved from the working directory.

The NPZ's val_idx fixes the prediction-row order. If RUN_DIR/indices.json is
also present, its val_idx must agree. No row-order fallback is used for SMILES.
Element-presence flags use XYZ atoms; bond and functional-group flags use
RDKit on SMILES, exactly as in the original chemistry-analysis notebook.
Missing or invalid molecular metadata raises an error rather than being
assigned to the category-absent group.

Spectra are converted to float64, clipped at zero, and normalised to unit sum
before Pearson calculation, preserving the notebook's epsilon of 1e-12.
Zero-total spectra become uniform; constant spectra give Pearson zero, as
in the notebook. No inference, training, or other spectral metrics are run.

Default output:
  data/plotting/figure_6/category_delta_true_minus_false_pearson.csv

Use --output to save elsewhere; overriding --run-dir changes only the input
location. The default output is always the repository's Figure 6 plotting data.

Only columns consumed by the plotting script are written:
  category, n_false, n_true, median_false, median_true,
  delta_median_true_minus_false

True/False mean category present/absent. Categories with an empty group are
omitted, matching the notebook. Categories may overlap across comparisons.
"""

from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd


EPS = 1e-12
OUTPUT_NAME = "category_delta_true_minus_false_pearson.csv"

# ---------------------------------------------------------------------------
# Repository paths. Place this file in scripts/analysis/ so parents[2] is the
# repository root. These defaults are independent of the working directory.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = REPO_ROOT / "data" / "figure_6"
DEFAULT_RUN_DIR = INPUT_DIR  # Contains val_preds_true.npz and indices.json.
DEFAULT_CACHE_INDEX = INPUT_DIR / "cache_index.json"
DEFAULT_XYZ_ROOT = INPUT_DIR / "xyz"
DEFAULT_SMILES_CSV = INPUT_DIR / "validation_molecule_numbers_ordered_with_smiles.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "plotting" / "figure_6" / OUTPUT_NAME

# Only the functional groups actually included in the notebook's final plots.
FUNCTIONAL_GROUP_SMARTS = {
    "fg_carbonyl_CeqO": "[CX3]=[OX1]",
    "fg_alkene_CCdouble": "[#6]=[#6]",
    "fg_alkyne_CCtriple": "[#6]#[#6]",
    "fg_nitrile": "[CX2]#N",
    "fg_hydroxyl": "[OX2H]",
    "fg_ether": "[OD2]([#6])[#6]",
    "fg_amine_nonamide": "[NX3;!$(NC=O)]",
}
CATEGORY_COLUMNS = [
    "has_double_bond", "has_triple_bond", "has_aromatic_bond", "has_ring",
    "has_unsaturation", "has_only_single_bonds", "has_oxygen", "has_nitrogen",
    "has_fluorine", "has_oxygen_and_nitrogen", "has_hetero_N_or_O",
    *FUNCTIONAL_GROUP_SMARTS,
]
OUTPUT_COLUMNS = [
    "category", "n_false", "n_true", "median_false", "median_true",
    "delta_median_true_minus_false",
]


def load_predictions(run_dir: Path, cache_index_path: Path):
    """Load spectra and map their saved validation indices to molecule IDs."""
    with np.load(run_dir / "val_preds_true.npz", allow_pickle=False) as archive:
        missing = {"pred", "true", "val_idx"} - set(archive.files)
        if missing:
            raise ValueError(f"Prediction NPZ is missing arrays: {sorted(missing)}")
        pred = np.asarray(archive["pred"], dtype=np.float64)
        true = np.asarray(archive["true"], dtype=np.float64)
        val_idx = np.asarray(archive["val_idx"])
    if pred.ndim != 2 or pred.shape != true.shape or 0 in pred.shape:
        raise ValueError("pred and true must be non-empty arrays with the same (molecules, bins) shape.")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("Prediction and reference spectra must contain only finite values.")
    if val_idx.ndim != 1 or len(val_idx) != len(pred) or val_idx.dtype.kind not in "iu":
        raise ValueError("val_idx must contain one integer dataset index per spectrum row.")
    if len(np.unique(val_idx)) != len(val_idx):
        raise ValueError("Repeated validation indices would double-count molecules.")

    # Cross-check an existing split manifest without requiring a second copy
    # of the validation indices for this generator to run.
    indices_path = run_dir / "indices.json"
    if indices_path.is_file():
        recorded = json.loads(indices_path.read_text())["val_idx"]
        if not np.array_equal(val_idx, np.asarray(recorded)):
            raise ValueError("NPZ val_idx and indices.json val_idx differ; check the run files.")

    samples = json.loads(cache_index_path.read_text())["samples"]
    if np.any(val_idx < 0) or np.any(val_idx >= len(samples)):
        raise ValueError("Validation indices are outside the supplied cache index.")
    molecule_numbers = []
    for index in val_idx:
        density_path = str(samples[int(index)]["dens_npz"])
        match = re.search(r"density_M(\d+)\.npz$", density_path)
        if match is None:
            raise ValueError(f"Cannot identify molecule from: {density_path}")
        molecule_numbers.append(int(match.group(1)))
    if len(set(molecule_numbers)) != len(molecule_numbers):
        raise ValueError("Validation rows map to repeated molecule IDs in the cache index.")
    return pred, true, molecule_numbers


def pearson_per_molecule(pred, true):
    """Vectorised version of the notebook's normalised Pearson calculation."""
    def centred_probability(spectra):
        values = np.clip(np.asarray(spectra, dtype=np.float64), 0.0, None)
        totals = values.sum(axis=1, keepdims=True)
        values /= np.maximum(totals, EPS)
        values[totals[:, 0] <= EPS] = 1.0 / values.shape[1]
        return values - values.mean(axis=1, keepdims=True)

    p = centred_probability(pred)
    q = centred_probability(true)
    denominator = np.sqrt(np.sum(p * p, axis=1) * np.sum(q * q, axis=1)) + EPS
    return np.sum(p * q, axis=1) / denominator


def load_smiles(path: Path):
    """Read an explicit ID-to-SMILES table; never assume matching row order."""
    table = pd.read_csv(path)
    columns = {str(column).strip().lower(): column for column in table.columns}
    id_aliases = [
        "molecule_number", "mol_num", "mol_number", "moleculenumber", "mid",
        "molecule_id", "molecule", "qm7_index", "dataset_molecule_number",
    ]
    smiles_aliases = ["smiles", "canonical_smiles", "rdkit_smiles"]
    id_column = next((columns[key] for key in id_aliases if key in columns), None)
    smiles_column = next((columns[key] for key in smiles_aliases if key in columns), None)
    if id_column is None or smiles_column is None:
        raise ValueError("SMILES CSV needs molecule_number and smiles columns (common aliases are accepted).")
    table = table[[id_column, smiles_column]].copy()
    table.columns = ["molecule_number", "smiles"]
    ids = pd.to_numeric(table["molecule_number"], errors="raise").to_numpy()
    if not np.isfinite(ids).all() or np.any(ids != np.floor(ids)):
        raise ValueError("Molecule IDs in the SMILES CSV must be integers without missing values.")
    table["molecule_number"] = ids.astype(np.int64)
    if table["molecule_number"].duplicated().any():
        raise ValueError("SMILES CSV must contain a single row per molecule ID.")
    table["smiles"] = table["smiles"].astype("string").str.strip()
    return table.set_index("molecule_number")["smiles"]


def xyz_elements(path: Path):
    """Read only element symbols from an XYZ file, retaining its comment line."""
    with path.open() as handle:
        n_atoms = int(handle.readline().strip())
        if n_atoms <= 0:
            raise ValueError(f"XYZ file has no atoms: {path}")
        handle.readline()  # The second line is a comment, which may be blank.
        elements = set()
        for _ in range(n_atoms):
            fields = handle.readline().split()
            if len(fields) < 4:
                raise ValueError(f"Incomplete atom record in XYZ file: {path}")
            elements.add(fields[0])
    return elements


def chemical_categories(molecule_numbers, xyz_root: Path, smiles_csv: Path):
    """Compute only the flags used in the final category comparison."""
    from rdkit import Chem

    smiles_lookup = load_smiles(smiles_csv)
    patterns = {name: Chem.MolFromSmarts(smarts) for name, smarts in FUNCTIONAL_GROUP_SMARTS.items()}
    rows = []
    for mid in molecule_numbers:
        smiles = smiles_lookup.get(mid, pd.NA)
        if pd.isna(smiles) or not str(smiles).strip():
            raise ValueError(f"Missing SMILES for molecule {mid}.")
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"Invalid SMILES for molecule {mid}: {smiles}")
        elements = xyz_elements(xyz_root / str(mid) / f"GEO_M{mid}.xyz")
        bonds = list(mol.GetBonds())
        has_double = any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in bonds)
        has_triple = any(bond.GetBondType() == Chem.BondType.TRIPLE for bond in bonds)
        has_aromatic = any(bond.GetIsAromatic() for bond in bonds)
        rows.append({
            "has_double_bond": has_double,
            "has_triple_bond": has_triple,
            "has_aromatic_bond": has_aromatic,
            "has_ring": mol.GetRingInfo().NumRings() > 0,
            "has_unsaturation": has_double or has_triple or has_aromatic,
            # The notebook's no-bond value was missing and then filled False.
            "has_only_single_bonds": bool(bonds) and all(
                bond.GetBondType() == Chem.BondType.SINGLE for bond in bonds
            ),
            "has_oxygen": "O" in elements,
            "has_nitrogen": "N" in elements,
            "has_fluorine": "F" in elements,
            "has_oxygen_and_nitrogen": "O" in elements and "N" in elements,
            "has_hetero_N_or_O": "O" in elements or "N" in elements,
            **{name: bool(mol.HasSubstructMatch(pattern)) for name, pattern in patterns.items()},
        })
    return pd.DataFrame(rows, columns=CATEGORY_COLUMNS)


def summarise_categories(pearson, categories):
    rows = []
    for column in CATEGORY_COLUMNS:
        present = categories[column].to_numpy(dtype=bool)
        false_values, true_values = pearson[~present], pearson[present]
        if len(false_values) == 0 or len(true_values) == 0:
            continue
        median_false, median_true = np.median(false_values), np.median(true_values)
        rows.append({
            "category": column,
            "n_false": len(false_values), "n_true": len(true_values),
            "median_false": median_false, "median_true": median_true,
            "delta_median_true_minus_false": median_true - median_false,
        })
    if not rows:
        raise ValueError("No chemistry category has both a present group and an absent group.")
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS).sort_values(
        "delta_median_true_minus_false", kind="stable"
    ).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR,
                        help="Folder containing val_preds_true.npz and optional indices.json.")
    parser.add_argument("--cache-index", type=Path, default=DEFAULT_CACHE_INDEX)
    parser.add_argument("--xyz-root", type=Path, default=DEFAULT_XYZ_ROOT)
    parser.add_argument("--smiles-csv", type=Path, default=DEFAULT_SMILES_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output CSV; defaults to data/plotting/figure_6/ in the repository.")
    args = parser.parse_args()
    try:
        run_dir = args.run_dir.expanduser().resolve()
        pred, true, molecule_numbers = load_predictions(run_dir, args.cache_index.expanduser().resolve())
        pearson = pearson_per_molecule(pred, true)
        categories = chemical_categories(
            molecule_numbers, args.xyz_root.expanduser().resolve(), args.smiles_csv.expanduser().resolve()
        )
        summary = summarise_categories(pearson, categories)
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output, index=False)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Used {len(molecule_numbers)} validation molecules and {pred.shape[1]} spectral bins.")
    print(f"Saved {len(summary)} category comparisons: {output}")


if __name__ == "__main__":
    main()
