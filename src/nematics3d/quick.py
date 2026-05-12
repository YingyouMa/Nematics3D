"""Quick convenience workflows for common Nematics3D visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .classes.q_field_object import QFieldObject
from .classes.visual.plot_figure import PlotFigure
from .classes.visual.plot_sphere import OptsSphere, PlotSphere
from .classes.visual.plot_tube import OptsTube
from .datatypes import UNSET


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
    S=UNSET,
    n=UNSET,
    Q_input=UNSET,
    box_periodic_flag=False,
    name="Q",
    grid_normal=(0, 0, 1),
    is_visualize_lines=True,
    save_path=None,
    is_off_screen=False,
):
    is_Q_input_provided = Q_input is not None and Q_input is not UNSET
    is_S_provided = S is not None and S is not UNSET
    is_n_provided = n is not None and n is not UNSET

    if not is_Q_input_provided and not is_n_provided:
        raise ValueError("Provide `Q_input` or `n`.")

    field_for_shape = Q_input if is_Q_input_provided else n
    params = _auto_quick_Q_visual_params(field_for_shape, grid_normal)

    q_obj = QFieldObject(
        Q=Q_input if is_Q_input_provided else UNSET,
        S=S if is_S_provided else UNSET,
        n=n if is_n_provided else UNSET,
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

    figure = PlotFigure(is_off_screen=is_off_screen)
    if is_visualize_lines:
        q_obj.act_visualize_disclination_lines(
            figure=figure,
            is_extent=False,
            min_line_length=params["visual_min_line_length"],
            line_radius=params["line_radius"],
        )
    else:
        PlotSphere(
            coords=q_obj.calc_defect_grid,
            name="defect points",
            category="defects",
            figure=figure,
            opts=OptsSphere(
                color=(0.5, 0.5, 0.5),
                radius=params["defect_radius"],
            ),
        )

    q_obj.calc_bounds.act_visualize(
        figure=figure,
        opts=OptsTube(radius=params["extent_radius"]),
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

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.act_savefig(save_path)

    return q_obj, figure
