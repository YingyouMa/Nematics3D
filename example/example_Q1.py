import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import Nematics3D

DATA_DIR = Path(__file__).resolve().parent / "data"

n = np.load(DATA_DIR / "n_example_global.npy")
S = np.load(DATA_DIR / "S_example_global.npy")

Q = Nematics3D.QFieldObject(S=S, n=n, box_periodic_flag=True, name="testQ")
Q.act_lines_smooth()
Q.act_visualize_disclination_lines(is_wrap=False, extent_color=(0.5, 0.5, 0.5))
Q.act_visualize_disclination_lines(
    is_new=True,
    is_wrap=True,
    line_shading_type="pbr",
    figure_azimuth=0,
    figure_elevation=45,
)

bounds_max = 80
bounds = Nematics3D.as_bounds((0, bounds_max, 0, bounds_max, 0, bounds_max))
figure = Nematics3D.PlotFigure(name="zoomed-in")
opts_line = Nematics3D.OptsTube(color=(0.5,0.5,0.5), radius=0.3)
Q.act_visualize_disclination_lines(figure=figure, bounds=bounds, opts_line=opts_line)

trans = 7.5
spacing = 2.5

Q.act_visualize_n_plane(
    grid_normal=(1, 1, 1),
    grid_spacing=spacing,
    grid_size=100,
    grid_origin=(bounds_max / 2 - trans, bounds_max / 2 - trans, bounds_max / 2 - trans),
    is_extent=True,
    bounds=bounds,
    figure=figure
)
Q.figs.active_fig.act_view_yz()
Q.figs.active_fig.opts.azimuth = 90
