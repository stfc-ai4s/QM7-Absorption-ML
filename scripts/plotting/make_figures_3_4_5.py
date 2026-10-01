#!/usr/bin/env python3
"""Plot Figures 3/4 from four committed DAT tables; Figure 5 is opt-in.

Run with no arguments for Figures 3 and 4, or --figures 3 4 5 for all three.
DAT columns: energy_au, reference, prediction. Energy is read from each table.
Model weights and density inputs are not needed for plotting saved tables.

    python scripts/plotting/make_figures_3_4_5.py --figures 3 4 5
"""
from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "figures_3_4"
NORMALIZATION_NPZ = REPO_ROOT / "data" / "figure_5" / "val_preds_true_raw.npz"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures"


def load_spectra(data_dir, manifest, requested_ids):
    entries = {int(entry["id"]): entry for entry in manifest["molecules"]}
    if len(entries) != len(manifest["molecules"]):
        raise ValueError("Duplicate molecule IDs in molecules.json.")
    records = {}
    for mid in requested_ids:
        path = data_dir / f"spectrum_prediction_M{mid}.dat"
        if not path.is_file():
            raise FileNotFoundError(f"Missing paper table: {path}. Copy your original table or run scripts/analysis/generate_figures_3_4_data.py with external weights.")
        with path.open() as handle:
            header = handle.readline().lstrip("# ").strip()
        if "energy" in header and "pred" in header and "ref" in header:
            fields = header.split()
            if len(fields) >= 3 and "pred" in fields[1]:
                raise ValueError(f"Wrong column order in {path}: expected energy, reference, prediction.")
        table = np.loadtxt(path, ndmin=2)
        if table.shape[1] != 3 or len(table) < 2 or not np.isfinite(table).all():
            raise ValueError(f"{path} must contain three finite columns: energy, reference, prediction.")
        if (np.diff(table[:, 0]) <= 0).any():
            raise ValueError(f"Energy must increase in {path}.")
        if (table[:, 1:] < 0).any() or (table[:, 1:].sum(axis=0) <= 0).any():
            raise ValueError(f"Reference and prediction must be non-negative with positive sums: {path}")
        # Saved values are plotted without rescaling or smoothing.
        records[mid] = {"energy": table[:, 0], "true": table[:, 1], "pred": table[:, 2],
                        "smiles": entries[mid]["smiles"]}
    return records


def save_example_metrics(records, output_dir):
    path = output_dir / "figures_3_4_metrics.csv"
    fields = ["molecule_id", "smiles", "pearson", "mae", "reference_sum", "prediction_sum", "bins", "energy_min_au", "energy_max_au"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mid, r in records.items():
            writer.writerow({"molecule_id": mid, "smiles": r["smiles"],
                             "pearson": float(np.corrcoef(r["true"], r["pred"])[0, 1]),
                             "mae": float(np.mean(np.abs(r["pred"] - r["true"]))),
                             "reference_sum": float(r["true"].sum()), "prediction_sum": float(r["pred"].sum()),
                             "bins": len(r["energy"]), "energy_min_au": r["energy"][0], "energy_max_au": r["energy"][-1]})
    return path


def molecule_image(smiles: str, size=(400, 220)):
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    rdDepictor.Compute2DCoords(mol)
    image = Draw.MolToImage(mol, size=size, kekulize=True, fitImage=True).convert("RGBA")

    arr = np.asarray(image).copy()
    white = np.all(arr[:, :, :3] > 245, axis=2)
    arr[white, 3] = 0
    return arr


def plot_normalization(data_path: Path, output_dir: Path, dpi: int):
    from sklearn.metrics import r2_score

    with np.load(data_path, allow_pickle=False) as z:
        pred = z["pred"].astype(float).reshape(-1)
        true = z["true"].astype(float).reshape(-1)
    if pred.shape != true.shape or pred.size < 2 or not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("Figure 5 needs matching finite pred/true arrays with at least two entries.")

    error = pred - true
    r2 = r2_score(true, pred)
    mn = min(true.min(), pred.min())
    mx = max(true.max(), pred.max())

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    axes[0].scatter(true, pred, s=20, alpha=0.6, label=f"R² = {r2:.4f}")
    axes[0].plot([mn, mx], [mn, mx], label="y=x reference")
    axes[0].set_xlabel("True", fontsize=16)
    axes[0].set_ylabel("Predicted", fontsize=16)
    axes[0].set_title("Validation: Predicted vs True Normalization Constant", fontsize=18)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(error, bins=20, alpha=0.8, histtype="step", lw=4)
    axes[1].set_xlabel("Error (pred - true)", fontsize=16)
    axes[1].set_ylabel("Count", fontsize=16)
    axes[1].set_title("Validation Error Histogram", fontsize=18)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / "figure_5.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_figure_3(records, manifest, output_dir: Path, dpi: int):
    examples = manifest["figure_3"]
    labels = ["a", "b", "c"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9.2), sharex=True)

    for ax, mid, label in zip(axes, examples, labels):
        record = records[mid]
        x, true, pred, smi = record["energy"], record["true"], record["pred"], record["smiles"]
        corr = np.corrcoef(true, pred)[0, 1]

        ax.plot(x, pred, label="Predicted", linewidth=1.8)
        ax.plot(x, true, label="Reference", linewidth=1.2)
        ax.margins(y=0.05)

        ax.text(0.01, 0.83, f"{label}) M{mid}: {smi}", transform=ax.transAxes,
                ha="left", va="top", fontsize=11)
        ax.text(0.01, 0.72, f"Corr = {corr:.4g}", transform=ax.transAxes,
                ha="left", va="top", fontsize=11)

        inset = ax.inset_axes([0.08, 0.34, 0.29, 0.40])
        inset.imshow(molecule_image(smi))
        inset.axis("off")

        ax.set_ylabel("Normalized Intensity")
        ax.grid(True, alpha=0.15)

    axes[-1].set_xlabel("Energy (a.u.)")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, loc="upper center",
               bbox_to_anchor=(0.85, 0.95), ncols=2)

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = output_dir / "figure_3.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


# -----------------------------------------------------------------------------
# Figure 4: two similar molecules
# -----------------------------------------------------------------------------

def plot_figure_4(records, manifest, output_dir: Path, dpi: int):
    a, b = [records[mid] for mid in manifest["figure_4"]]
    smiles_a, smiles_b = a["smiles"], b["smiles"]
    t1, t2 = a["true"], b["true"]
    p1, p2 = a["pred"], b["pred"]
    x1, x2 = a["energy"], b["energy"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True)

    axes[0].plot(x1, t1, linewidth=1.6, label=smiles_a)
    axes[0].plot(x2, t2, linewidth=1.6, label=smiles_b)
    axes[0].set_ylabel("Normalized Intensity")
    axes[0].set_yticks([])
    axes[0].set_title("True spectra comparison")
    axes[0].legend(frameon=False, ncols=2, loc="upper left")
    axes[0].grid(True, alpha=0.15)

    axes[1].plot(x1, p1, linewidth=1.6, label=smiles_a)
    axes[1].plot(x2, p2, linewidth=1.6, label=smiles_b)
    axes[1].set_xlabel("Energy (a.u.)")
    axes[1].set_ylabel("Normalized Intensity")
    axes[1].set_yticks([])
    axes[1].set_title("Predicted spectra comparison")
    axes[1].legend(frameon=False, ncols=2, loc="upper left")
    axes[1].grid(True, alpha=0.15)

    inset_a = axes[0].inset_axes([0.10, 0.48, 0.14, 0.34])
    inset_a.imshow(molecule_image(smiles_a, size=(350, 220)))
    inset_a.axis("off")

    inset_b = axes[0].inset_axes([0.26, 0.48, 0.14, 0.34])
    inset_b.imshow(molecule_image(smiles_b, size=(350, 220)))
    inset_b.axis("off")

    axes[0].text(0.17, 0.45, "A", transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=10)
    axes[0].text(0.33, 0.45, "B", transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=10)

    fig.tight_layout()
    out = output_dir / "figure_4.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--normalization-npz", type=Path, default=NORMALIZATION_NPZ)
    parser.add_argument("--figures", type=int, nargs="+", choices=[3, 4, 5], default=[3, 4])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dpi <= 0:
        parser.error("DPI must be positive.")
    saved = []
    if 3 in args.figures or 4 in args.figures:
        data_dir = args.data_dir.expanduser().resolve()
        manifest = json.loads((data_dir / "molecules.json").read_text())
        ids = list(dict.fromkeys(mid for number in [3, 4] if number in args.figures for mid in manifest[f"figure_{number}"]))
        records = load_spectra(data_dir, manifest, ids)
        if 3 in args.figures:
            saved.append(plot_figure_3(records, manifest, output_dir, args.dpi))
        if 4 in args.figures:
            saved.append(plot_figure_4(records, manifest, output_dir, args.dpi))
        saved.append(save_example_metrics(records, output_dir))
    if 5 in args.figures:
        path = args.normalization_npz.expanduser().resolve()
        if not path.is_file():
            parser.error(f"Figure 5 requires {path}; run --figures 3 4 until its predictions are available.")
        saved.append(plot_normalization(path, output_dir, args.dpi))
    for path in saved:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
