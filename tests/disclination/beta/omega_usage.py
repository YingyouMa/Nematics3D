from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path.cwd()
while not (REPO_ROOT / "src" / "nematics3d").exists():
    if REPO_ROOT.parent == REPO_ROOT:
        raise RuntimeError("Could not locate the repository root.")
    REPO_ROOT = REPO_ROOT.parent

SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import nematics3d

DATA_PATH = REPO_ROOT / "tests" / "disclination" / "beta" / "Q_1630.npy"
Q_data = np.load(DATA_PATH)[0]
Q_data = Q_data[168:185, 5:32, 10:35]

Q = nematics3d.QFieldObject(
    Q=Q_data,
    name="WT",
)

if len(Q.lines) != 1:
    raise RuntimeError(f"Expected exactly one line, got {len(Q.lines)}.")

Q.act_lines_smooth(window_length=28)
smooth = Q.lines[0].smooths[-1]

u_percent = 5
omega_result = smooth.act_calc_omega(u_percent)
beta = omega_result["beta"]
print(beta)

Q.act_visualize_disclination_lines(
    is_new=True,  # Create a new figure.
    min_line_length=0,  # Keep even short detected lines so this small local example is not filtered out.
    title="WT disclination lines",  # The title (also the name) of the figure
)

Q.act_visualize_n_near_defect(u_percent=u_percent)  # Draw the director field around the selected point on the defect line.

Q.figs.active_fig.act_commit(  # Preset viewing angle chosen by the author for a clear first look; feel free to rotate freely in the interactive window.
  azimuth=73,
  elevation=49,
  roll=-11,
  distance=20,
  focal_point=[ 6.31487462, 15.66747125,  6.04773563],
)


beta_func = smooth.act_create_linefunc(
    func=lambda u: smooth.act_calc_omega(u)["beta"],  # Function to sample: input is u_percent, output is beta.
    u_samples=np.arange(0, 100, 5),  # Sample positions in u_percent.
    name="beta",  # Registry name used to attach this line function under smooth.linefuncs.
)

smooth.visual_tube.act_commit(
    paint_by="scalars",  # Use scalar coloring instead of a single fixed tube color.
    resolver_source="u_percent",  # Pass each tube point's normalized line position to beta_func.
    scalars=beta_func,  # The line-function interpolator that returns beta along the smoothed line.
    scalar_bar_title="beta",  # Title shown on the scalar color bar.
)


Q.figs.active_fig.act_commit(  # Preset viewing angle chosen by the author for a clear first look; feel free to rotate freely in the interactive window.
  azimuth=71,
  elevation=20,
  roll=2.1,
  distance=43,
  focal_point=[ 3.86305393, 12.81694508, 10.51823688],
)