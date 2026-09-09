"""Minimal quick_visualize_q example using the bundled director data."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import nematics3d


DATA_DIR = Path(__file__).resolve().parent / "data"

n = np.load(DATA_DIR / "n_example_global.npy")
S = np.load(DATA_DIR / "S_example_global.npy")

Nx, Ny, Nz = 32, 64, 64
S_input = S[:Nx, :Ny, :Nz]
n_input = n[:Nx, :Ny, :Nz]

q_obj, figure = nematics3d.quick_visualize_q(
    S=S_input,
    n=n_input,
    box_periodic_flag=False,
    is_visualize_lines=True,
)
