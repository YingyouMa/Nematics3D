import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import Nematics3D


DATA_DIR = Path(__file__).resolve().parent / "data"

n = np.load(DATA_DIR / "n_example_global.npy")
S = np.load(DATA_DIR / "S_example_global.npy")
n = n[:60, :60, :60]
S = S[:60, :60, :60]

Q = Nematics3D.QFieldObject(S=S, n=n, name="testQ")
Q.act_lines_smooth()

figure = Nematics3D.PlotFigure(
    name="lines and directors",
    is_off_screen=True,
)  # render off-screen so the example can save the figure directly

Q.act_visualize_disclination_lines(
    figure=figure,
    line_color=(0.5, 0.5, 0.5),
    line_radius=0.3,
)

spacing = 3
trans = 6

Q.act_visualize_n_plane(
    figure=figure,
    is_extent=False,
    grid_normal=(1, 1, 1),
    grid_origin=(
        30 - trans,
        30 - trans,
        30 - trans,
    ),
    grid_size=100,
    grid_spacing=spacing,
    n_length=spacing,
)

figure.act_commit(elevation=0, azimuth=90, distance=150)

figure.act_savefig("../docs/example/informative/2.png")
