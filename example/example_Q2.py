import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import nematics3d

DATA_DIR = Path(__file__).resolve().parent / "data"

n = np.load(DATA_DIR / "n_example_global.npy")
S = np.load(DATA_DIR / "S_example_global.npy")

# Q = nematics3d.QFieldObject(S=S, n=n, box_periodic_flag=True, name="testQ")
# Q.act_lines_smooth()

# figure = nematics3d.PlotFigure()
# Q.act_visualize_disclination_lines(figure=figure, line_radius=0.3, extent_radius=0.1)

# Q.act_visualize_n_plane(
#     grid_normal=(0, 0, 1),
#     grid_spacing=2.5,
#     grid_size=200,
#     grid_origin=(64, 64, 64),
#     figure=figure,
# )


def _auto_quick_Q_visual_params(field, grid_normal):
    shape = np.asarray(field.shape[:3], dtype=float)
    if shape.shape != (3,):
        raise ValueError("Input field must provide a 3D grid shape in its first axes.")

    grid_normal = np.asarray(grid_normal, dtype=float)
    if grid_normal.shape != (3,) or np.linalg.norm(grid_normal) == 0:
        raise ValueError("`grid_normal` must be a nonzero 3D vector.")

    base_size = 128.0
    scale = np.prod(shape / base_size) ** (1.0 / 3.0)
    smooth_window_length = int(round(41 * scale))
    if smooth_window_length % 2 == 0:
        smooth_window_length += 1
    smooth_window_length = max(smooth_window_length, 5)

    return {
        "smooth_min_line_length": max(int(round(61 * scale)), 2),
        "smooth_window_length": smooth_window_length,
        "visual_min_line_length": max(int(round(75 * scale)), 2),
        "line_radius": 0.3 * scale,
        "extent_radius": 0.1 * scale,
        "defect_radius": 0.675 * scale,
        "grid_origin": tuple(shape / 2.0),
        "grid_size": float(np.linalg.norm(shape)),
        "grid_spacing": 2.5 * scale,
        "n_length": 2.5 * scale,
        "n_radius": 0.3 * scale,
    }


def quick_visualize_Q(
    S=nematics3d.UNSET,
    n=nematics3d.UNSET,
    Q=nematics3d.UNSET,
    box_periodic_flag=False,
    name="Q",
    grid_normal=(0, 0, 1),
    is_visualize_lines=True,
):
    unset = nematics3d.UNSET
    is_Q_provided = Q is not None and Q is not unset
    is_S_provided = S is not None and S is not unset
    is_n_provided = n is not None and n is not unset

    if not is_Q_provided and not is_n_provided:
        raise ValueError("Provide `Q` or `n`.")

    field_for_shape = Q if is_Q_provided else n

    params = _auto_quick_Q_visual_params(field_for_shape, grid_normal)

    q_obj = nematics3d.QFieldObject(
        Q=Q if is_Q_provided else unset,
        S=S if is_S_provided else unset,
        n=n if is_n_provided else unset,
        box_periodic_flag=box_periodic_flag,
        name=name,
        default_miminum_line_length_smooth=params["smooth_min_line_length"],
        default_smooth_window_length=params["smooth_window_length"],
        default_miminum_line_length_visual=params["visual_min_line_length"],
    )
    if is_visualize_lines:
        q_obj.act_lines_smooth(
            min_line_length=params["smooth_min_line_length"],
            window_length=params["smooth_window_length"],
        )

    figure = nematics3d.PlotFigure()
    if is_visualize_lines:
        q_obj.act_visualize_disclination_lines(
            figure=figure,
            is_extent=False,
            min_line_length=params["visual_min_line_length"],
            line_radius=params["line_radius"],
        )
    else:
        nematics3d.PlotSphere(
            coords=q_obj.calc_defect_grid,
            name="defect points",
            category="defects",
            figure=figure,
            opts=nematics3d.OptsSphere(
                color=(0.5, 0.5, 0.5),
                radius=params["defect_radius"],
            ),
        )

    q_obj.calc_bounds.act_visualize(
        figure=figure,
        opts=nematics3d.OptsTube(radius=params["extent_radius"]),
        is_reset_camera=False,
    )

    q_obj.act_visualize_n_plane(
        is_extent=False,
        grid_normal=grid_normal,
        grid_spacing=params["grid_spacing"],
        grid_size=params["grid_size"],
        grid_origin=params["grid_origin"],
        n_length=params["n_length"],
        n_radius=params["n_radius"],
        figure=figure,
    )

    return q_obj, figure


Nx, Ny, Nz = 32, 64, 64
S_input = S[:Nx, :Ny, :Nz]
n_input = n[:Nx, :Ny, :Nz]
Q, figure = quick_visualize_Q(
    S_input, n_input, box_periodic_flag=False, is_visualize_lines=False
)
