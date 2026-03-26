import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import nematics3d


DATA_DIR = Path(__file__).resolve().parent / "data"

# This workflow example starts from a precomputed Q-tensor field.
Q_data = np.load(DATA_DIR / "Q_example_workflow.npy")

# Step 1: build the central QFieldObject that will hold the field data,
# detected defect points, classified disclination lines, and later figures.
Q = nematics3d.QFieldObject(Q=Q_data, name="workflow_Q")

# Step 2: smooth the detected disclination lines before plotting them.
# The raw lines are discrete point chains, so plotting them directly often
# looks visibly jagged.
Q.act_lines_smooth(
    window_length=31,
    min_line_length=61,
)

# Step 3: visualize the smoothed disclination lines together with the box
# extent, using separate prefixed keyword groups for the two visual objects.
Q.act_visualize_disclination_lines(
    min_line_length=61,
    line_color=(0.45, 0.45, 0.45),
    line_radius=0.35,
    extent_color=(0.15, 0.15, 0.15),
    extent_radius=0.08,
)

Q.figs.active_fig.act_commit(
    azimuth=44,
    elevation=-1.6,
    roll=-3.2,
    distance=120,
    focal_point=[114.52592024, 35.7777258, 51.6029819],
)


# Step 4: add a director-field section on a plane. Here we reuse the same
# figure, so `is_extent=False` avoids drawing the same outer box again.
Q.act_visualize_n_plane(
    is_extent=False,  # the extent is already drawn in the current figure
    grid_normal=(0, 0, 1),
    grid_origin=(100, 50, 50),
    grid_size=200,
    grid_spacing=4,
    n_length=3.6,
    n_radius=0.25,
)

Q.figs.active_fig.act_commit(
    azimuth=90,
    elevation=34,
    roll=0,
    distance=2.2e02,
    focal_point=[98.63580312, 49.5, 49.5],
)

# grid_here = Q.figs.active_fig["n bulk of plane 'n-plane'"].owner.grid
# grid_here.act_commit(
#     origin=(100,50,75),
#     spacing=5
# )

bounds_local = nematics3d.as_bounds(
    (145, 175, 13, 51, 64, 94), name="small-loop bounds"
)

Q.act_visualize_disclination_lines(
    is_new=True,
    bounds=bounds_local,
    min_line_length=61,
    line_color=(0.45, 0.45, 0.45),
    line_radius=0.35,
    extent_color=(0.15, 0.15, 0.15),
    extent_radius=0.08,
)

Q.act_visualize_n_plane(
    bounds=bounds_local,
    grid_normal=(0, 0, 1),
    grid_origin=(160, 32, 80),
    grid_spacing=2.5,
    grid_size=60,
    n_length=2.8,
    n_radius=0.25,
)

Q.figs.active_fig.act_commit(azimuth=110, elevation=30)

bounds_local.opts.origin = (150, 13, 64)

bounds_polar = bounds_local.act_copy(name="small-loop polar bounds")
figure_polar = nematics3d.PlotFigure(name="small-loop polar view")

xmin, xmax, ymin, ymax, zmin, zmax = 145, 175, 13, 51, 64, 94
target_line = min(
    (
        line
        for line in Q.lines
        if line.kind == "loop"
        and xmin <= np.mean(line._calc_defect_coords[:, 0]) <= xmax
        and ymin <= np.mean(line._calc_defect_coords[:, 1]) <= ymax
        and zmin <= np.mean(line._calc_defect_coords[:, 2]) <= zmax
    ),
    key=lambda line: line._calc_defect_num,
)
smooth_target = target_line.smooths[0]

Q.act_visualize_disclination_lines(
    figure=figure_polar,
    bounds=bounds_polar,
    min_line_length=61,
    line_color=(0.45, 0.45, 0.45),
    line_radius=0.35,
    extent_color=(0.15, 0.15, 0.15),
    extent_radius=0.08,
)

Q.act_visualize_n_near_defect(
    u_percent=50,
    smooth=smooth_target,
    figure=figure_polar,
    bounds=bounds_polar,
    is_extent=True,
)

# Later steps in the README will continue from this object.
