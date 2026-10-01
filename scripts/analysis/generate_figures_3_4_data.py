#!/usr/bin/env python3
"""Generate the four committed paper tables with one external density model.

Input under --dataset-root: <id>/density_M<id>.npz and
<id>/tddft_spectrum_gamma_150meV_M<id>.dat for IDs 481, 3935, 4170, 972.
These eight files are part of the released dataset, not of this repository;
see docs/DATA.md for the record and for where to place them.

--check-labels compares the SMILES in data/figures_3_4/molecules.json against
the Figure 6 validation table and stops on disagreement. It is off by default
because M4170 is a known, deliberate disagreement: the paper label is
C=C(CC)NC=[N] while the geometry-derived table gives C=C(CC)NC=N. The published
label is kept so that regenerated tables stay comparable with the paper
figures. Only the label differs; no spectrum is affected.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "inference"))
# Importing this name into __main__ also supports the existing CNNModel pickle.
from predict_spectrum_from_density import CNNModel, load_model, predict_to_file, select_device

DATA_DIR = REPO_ROOT / "data" / "figures_3_4"
SMILES_CSV = REPO_ROOT / "data" / "figure_6" / "validation_molecule_numbers_ordered_with_smiles.csv"


def verify_molecule_labels(entries, csv_path):
    """Catch the inconsistent Figure 4 labels against an explicit ID table."""
    from rdkit import Chem
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = {column.strip().lower(): column for column in (reader.fieldnames or [])}
        id_column = next((columns[key] for key in ["molecule_number", "mol_num", "mol_number", "moleculenumber", "mid", "molecule_id", "molecule", "qm7_index", "dataset_molecule_number"] if key in columns), None)
        smiles_column = next((columns[key] for key in ["smiles", "canonical_smiles", "rdkit_smiles"] if key in columns), None)
        if id_column is None or smiles_column is None:
            raise ValueError("SMILES CSV needs molecule-number and SMILES columns.")
        lookup = {}
        wanted = {int(entry["id"]) for entry in entries}
        for row in reader:
            mid = int(row[id_column])
            if mid in wanted:
                if mid in lookup:
                    raise ValueError(f"Duplicate molecule M{mid} in the SMILES CSV.")
                lookup[mid] = row[smiles_column]
    for entry in entries:
        mid = int(entry["id"])
        actual = Chem.MolFromSmiles(lookup.get(mid, ""))
        expected = Chem.MolFromSmiles(entry["smiles"])
        if mid not in lookup or actual is None or expected is None or Chem.MolToSmiles(actual) != Chem.MolToSmiles(expected):
            raise ValueError(f"M{mid}: paper label {entry['smiles']!r} disagrees with SMILES CSV {lookup.get(mid)!r}. Resolve the molecule mapping in molecules.json before regenerating the paper tables.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="External CNN checkpoint; it is not copied into the repository.")
    parser.add_argument("--config", type=Path, help="CNNModel kwargs for a plain state_dict checkpoint.")
    parser.add_argument("--dataset-root", type=Path, default=DATA_DIR / "inputs")
    parser.add_argument("--smiles-csv", type=Path, default=SMILES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--length", type=int, default=900)
    parser.add_argument("--reference-sigma", type=float, default=10.0)
    parser.add_argument("--reference-column", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--check-labels", action="store_true",
                        help="Verify molecules.json SMILES against the validation table. "
                             "Fails on the known M4170 label difference described above.")
    args = parser.parse_args()
    manifest = json.loads((DATA_DIR / "molecules.json").read_text())
    entries = manifest["molecules"]
    root = args.dataset_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if args.check_labels:
        verify_molecule_labels(entries, args.smiles_csv.expanduser().resolve())
    for entry in entries:
        mid = entry["id"]
        for filename in [f"density_M{mid}.npz", f"tddft_spectrum_gamma_150meV_M{mid}.dat"]:
            path = root / str(mid) / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing regeneration input: {path}")
    device = select_device(args.device)
    model_path = args.model.expanduser().resolve()
    model = load_model(model_path, device, args.config.expanduser().resolve() if args.config else None)
    for entry in entries:
        mid = entry["id"]
        path = predict_to_file(
            model, root / str(mid) / f"density_M{mid}.npz", output, device,
            length=args.length,
            reference_path=root / str(mid) / f"tddft_spectrum_gamma_150meV_M{mid}.dat",
            reference_column=args.reference_column, reference_sigma=args.reference_sigma,
            model_path=model_path,
        )
        print(f"Saved: {path}")
    digest = hashlib.sha256()
    with model_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    (output / "prediction_run.json").write_text(json.dumps({
        "model_sha256": digest.hexdigest(), "model": str(model_path),
        "molecule_ids": [entry["id"] for entry in entries],
        "selected_bins": args.length, "reference_sigma_bins": args.reference_sigma,
        "reference_column": args.reference_column,
        "note": "Compare the generated values with the archived paper tables using the same checkpoint and preprocessing.",
    }, indent=2) + "\n")
    # Needed for plotting when an explicit output directory was supplied.
    (output / "molecules.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
