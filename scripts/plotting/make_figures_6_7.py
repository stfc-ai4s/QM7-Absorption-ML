#!/usr/bin/env python3
"""Plot Figures 6 and 7: the two Pearson-correlation chemistry comparisons.

Both figures are drawn from one full-precision summary CSV at:
    data/plotting/figure_6/category_delta_true_minus_false_pearson.csv

That CSV is committed. Regenerate it from the validation predictions with:
    python scripts/analysis/generate_figure_6_data.py
Extra columns in the CSV are ignored.

Required columns:
    category, n_false, n_true, median_false, median_true,
    delta_median_true_minus_false

False/True mean the chemical category is absent/present, respectively. The
difference is median_true - median_false; a negative difference indicates lower
median Pearson correlation for molecules in which the category is present.
Categories can overlap: each row is a separate present-versus-absent comparison.

Dependencies: numpy, pandas, matplotlib.
Run from the repository root:
    python scripts/plotting/make_figures_6_7.py
Or use an absolute script path from any directory. Defaults are anchored to
the script location. Explicit relative CLI paths use the working directory.

Optional arguments:
    --data-file /path/to/category_delta_true_minus_false_pearson.csv
    --output-dir /path/to/figures --dpi 300 --show

Outputs in figures/:
    figure_6.png    median Pearson correlation by chemical category
    figure_7.png    difference in median Pearson correlation, present minus absent

This script replots saved summary statistics. It does not recompute predictions,
Pearson correlations, or chemical features from the original validation data.
"""

from pathlib import Path
import argparse

import matplotlib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_FILE = (
    REPO_ROOT / "data/plotting/figure_6/category_delta_true_minus_false_pearson.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures"
DELTA_COLUMN = "delta_median_true_minus_false"
REQUIRED_COLUMNS = [
    "category", "n_false", "n_true", "median_false", "median_true", DELTA_COLUMN,
]


def load_summary(path: Path) -> pd.DataFrame:
    """Read saved statistics without recomputing or filtering any categories."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing summary CSV: {path}\n"
            "Copy category_delta_true_minus_false_pearson.csv from the notebook's "
            "chemistry_spectra_error_analysis folder, or supply --data-file."
        )
    data = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in data]
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(missing)}")
    data = data[REQUIRED_COLUMNS].copy()
    if data.empty or data["category"].isna().any():
        raise ValueError("The summary must contain non-empty category names.")
    data["category"] = data["category"].astype(str).str.strip()
    if data["category"].eq("").any() or data["category"].duplicated().any():
        raise ValueError("Category names must be non-empty and unique.")
    for column in REQUIRED_COLUMNS[1:]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if not np.isfinite(data[REQUIRED_COLUMNS[1:]].to_numpy(dtype=float)).all():
        raise ValueError("Summary statistics and group counts must be finite.")
    counts = data[["n_false", "n_true"]].to_numpy(dtype=float)
    if (counts <= 0).any() or (counts != np.floor(counts)).any():
        raise ValueError("Group counts must be positive integers.")
    if (data[["median_false", "median_true"]].abs() > 1).any().any():
        raise ValueError("Median Pearson correlations must lie between -1 and 1.")
    if not np.allclose(
        data[DELTA_COLUMN], data["median_true"] - data["median_false"],
        rtol=1e-7, atol=1e-10,
    ):
        raise ValueError(
            "Median differences disagree with the group medians. "
            "Use the original full-precision CSV saved by the notebook."
        )
    # Match the final notebook figures' ascending effect-size order.
    return data.sort_values(DELTA_COLUMN, kind="stable").reset_index(drop=True)


def category_labels(categories):
    """Use the notebook's label formatting consistently in both figures."""
    return (
        categories.str.replace("has_", "", regex=False)
        .str.replace("_", " ", regex=False)
        .str.title()
        .str.replace("Fg ", "", regex=False)
    )


def plot_group_medians(data):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    y = np.arange(len(data))
    height = 0.35
    for group, offset, label in [
        ("false", -height / 2, "Category absent (False)"),
        ("true", height / 2, "Category present (True)"),
    ]:
        values = data[f"median_{group}"].to_numpy()
        ax.barh(y + offset, values, height=height, label=label)
        for row_y, value, count in zip(y + offset, values, data[f"n_{group}"]):
            ax.annotate(
                f"n={int(count)}", (value, row_y), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=9,
            )
    values = data[["median_false", "median_true"]].to_numpy()
    xmin, xmax = values.min(), values.max()
    # Retain the notebook's zoomed scale; allow room for labels without a
    # hard-coded upper limit that could crop values in a different summary.
    ax.set_xlim(xmin - 0.01, xmax + max(0.002, 0.2 * (xmax - xmin)))
    ax.set_yticks(y)
    ax.set_yticklabels(category_labels(data["category"]))
    ax.set_xlabel("Median Pearson correlation")
    ax.set_title("Median Pearson correlation by chemical category", pad=32)
    ax.legend(
        frameon=False, loc="lower left", bbox_to_anchor=(0, 1.005),
        ncol=2, borderaxespad=0,
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_median_differences(data):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    y = np.arange(len(data))
    differences = data[DELTA_COLUMN].to_numpy()
    # Only medians are plotted: markers and annotations share the same row.
    ax.scatter(differences, y, s=70)
    ax.axvline(0, linewidth=1.2, linestyle="--", color="0.4")
    for row_y, difference, n_true, n_false in zip(
        y, differences, data["n_true"], data["n_false"]
    ):
        ax.annotate(
            f"n={int(n_true)}/{int(n_false)}", (difference, row_y),
            xytext=(5 if difference >= 0 else -5, 0),
            textcoords="offset points", va="center",
            ha="left" if difference >= 0 else "right", fontsize=9,
        )
    xmin = min(float(differences.min()), 0.0)
    xmax = max(float(differences.max()), 0.0)
    padding = 0.25 * (xmax - xmin) if xmax > xmin else 0.01
    ax.set_xlim(xmin - padding, xmax + padding)
    ax.set_yticks(y)
    ax.set_yticklabels(category_labels(data["category"]))
    ax.set_xlabel(
        "Difference in median Pearson correlation: present \N{MINUS SIGN} absent\n"
        "Count labels: n(present)/n(absent)"
    )
    ax.set_title("Median Pearson correlation difference by chemical category")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def make_plots(data_file: Path, output_dir: Path, dpi: int = 300, show: bool = False):
    """Save both figures and return their paths; display only when requested."""
    if dpi <= 0:
        raise ValueError("DPI must be a positive integer.")
    data = load_summary(Path(data_file).expanduser().resolve())
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved = []
    for plot_function, filename in [
        (plot_group_medians, "figure_6.png"),
        (plot_median_differences, "figure_7.png"),
    ]:
        figure = plot_function(data)
        path = output_dir / filename
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(path)
        if show:
            plt.show()
        plt.close(figure)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true", help="Also display the figures interactively.")
    args = parser.parse_args()
    try:
        saved = make_plots(args.data_file, args.output_dir, dpi=args.dpi, show=args.show)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    for path in saved:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
