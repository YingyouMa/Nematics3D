import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import Nematics3D


DATA_DIR = Path(__file__).resolve().parent / "data"

n = np.load(DATA_DIR / "n_example_global.npy")
S = np.load(DATA_DIR / "S_example_global.npy")
n = n[:30, :30, :30]
S = S[:30, :30, :30]

Q = Nematics3D.QFieldObject(S=S, n=n, name="testQ")
Q.act_lines_smooth(min_line_length=20, window_length=10)

figure = Nematics3D.PlotFigure(
    name="near-defect director field",
)

Q.act_visualize_disclination_lines(
    figure=figure,
    line_color=(0.5, 0.5, 0.5),
    line_radius=0.1,
    extent_radius=0.05,
    min_line_length=20,
)

figure.act_commit(
    elevation=0,
    azimuth=90,
    distance=70,
)

figure.act_savefig("docs/example/informative/3.png")

# smooth0 = Q.lines[0].smooth
# Q.act_visualize_n_near_defect(
#     u_percent=0.3,
#     smooth=smooth0,
#     figure=figure,
#     is_extent=False,
# )

# figure.act_commit(
#     elevation=0,
#     azimuth=90,
#     distance=70,
# )

# figure.act_savefig("docs/example/informative/3.png")
