#!/usr/bin/env bash
# Regenerate every paper figure from the data committed in this repository.
#
# No model weights and no electron-density cubes are needed: the figures are
# built from the saved prediction tables and validation predictions under
# data/. Regenerating those tables instead requires the released weights and
# dataset; see docs/DATA.md and docs/FIGURES.md.
#
#   bash scripts/reproduce_figures.sh              # write into figures/
#   bash scripts/reproduce_figures.sh /tmp/check   # write somewhere else
#
# Writing to a scratch directory and diffing against figures/ is the quickest
# way to confirm an environment reproduces the published output.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${REPO_ROOT}/figures}"
mkdir -p "${OUT_DIR}"

echo "==> Figures 3, 4 and 5 from data/figures_3_4 and data/figure_5"
python "${REPO_ROOT}/scripts/plotting/make_figures_3_4_5.py" \
    --figures 3 4 5 \
    --output-dir "${OUT_DIR}"

echo "==> Figure 6/7 chemistry summary from data/figure_6"
python "${REPO_ROOT}/scripts/analysis/generate_figure_6_data.py"

echo "==> Figures 6 and 7"
python "${REPO_ROOT}/scripts/plotting/make_figures_6_7.py" \
    --output-dir "${OUT_DIR}"

echo
echo "Done. Figures written to ${OUT_DIR}"
