#!/usr/bin/env python3
"""Convert Gaussian-cube electron densities into the NPZ files the scripts read.

The PSDI dataset ships each density as density_M<id>.cube. The density
predictors and the cache builder read density_M<id>.npz instead, with the volume
stored under the key ``data`` as an (nx, ny, nz) float32 array in cube order
(x slowest, z fastest). Each NPZ is written beside its cube.

    python scripts/convert_cubes_to_npz.py /your/scratch/qm7-spectra/dataset
    python scripts/convert_cubes_to_npz.py dataset/1/density_M1.cube

Directories are searched recursively for density_M*.cube. Existing NPZ files are
left alone unless --overwrite is given.
"""
import argparse
from pathlib import Path

import numpy as np


def read_cube(path):
    """Return the volumetric data of a cube file as an (nx, ny, nz) array."""
    with Path(path).open() as handle:
        lines = handle.read().split("\n")
    header = lines[2].split()
    n_atoms = int(header[0])
    shape = tuple(abs(int(lines[3 + axis].split()[0])) for axis in range(3))
    start = 6 + abs(n_atoms)
    if n_atoms < 0:
        # Orbital cubes carry one extra line listing the orbital indices.
        start += 1
    values = np.array(" ".join(lines[start:]).split(), dtype=np.float32)
    if values.size != int(np.prod(shape)):
        raise ValueError(f"{path}: expected {np.prod(shape)} values for grid {shape}, found {values.size}")
    return values.reshape(shape)


def find_cubes(paths):
    for path in map(Path, paths):
        if path.is_dir():
            yield from sorted(path.rglob("density_M*.cube"))
        else:
            yield path


def main():
    parser = argparse.ArgumentParser(description="Convert density_M<id>.cube files to density_M<id>.npz.")
    parser.add_argument("paths", nargs="+", help="Cube files, or directories to search recursively.")
    parser.add_argument("--overwrite", action="store_true", help="Replace NPZ files that already exist.")
    args = parser.parse_args()

    converted = skipped = 0
    for cube in find_cubes(args.paths):
        target = cube.with_suffix(".npz")
        if target.exists() and not args.overwrite:
            skipped += 1
            continue
        np.savez(target, data=read_cube(cube))
        converted += 1
    print(f"[OK] converted {converted}, skipped {skipped} existing")


if __name__ == "__main__":
    main()
