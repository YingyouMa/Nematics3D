"""Convenience workflows for Q-tensor fields."""

from pathlib import Path

import numpy as np

from ..datatypes import UNSET, as_bool
from ..logging_decorator import logging_and_warning_decorator
from ..q_field.q_field_object import QFieldObject
from ..visual.plot_figure import PlotFigure
from ..visual.plot_sphere import OptsSphere, PlotSphere
from ..visual.plot_tube import OptsTube

__all__ = ["quick_visualize_q"]


_DIRECTOR_SPACING_CONFIG = {
    "dense": {
        "grid_spacing_scale": 1.0,
        "n_length_scale": 1.0,
        "n_radius_scale": 1.0,
    },
    "medium": {
        "grid_spacing_scale": 1.75,
        "n_length_scale": 1.2,
        "n_radius_scale": 1.1,
    },
    "sparse": {
        "grid_spacing_scale": 2.5,
        "n_length_scale": 1.45,
        "n_radius_scale": 1.2,
    },
}


def _resolve_director_spacing_level(level):
    try:
        return _DIRECTOR_SPACING_CONFIG[level]
    except (KeyError, TypeError) as exc:
        valid_levels = ", ".join(repr(key) for key in _DIRECTOR_SPACING_CONFIG)
        raise ValueError(
            f"`director_spacing` must be one of {valid_levels}, got {level!r}."
        ) from exc


def _auto_quick_q_visual_params(field):
    shape = np.asarray(np.shape(field)[:3], dtype=float)
    if shape.shape != (3,) or np.any(shape <= 0):
        raise ValueError("Input field must have three non-empty spatial axes.")

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


def _validate_grid_normal(grid_normal):
    grid_normal = np.asarray(grid_normal, dtype=float)
    if grid_normal.shape != (3,) or not np.all(np.isfinite(grid_normal)):
        raise ValueError("`grid_normal` must be a finite 3D vector.")
    if np.linalg.norm(grid_normal) == 0.0:
        raise ValueError("`grid_normal` must be nonzero.")
    return grid_normal


@logging_and_warning_decorator(start_finish_level=5)
def quick_visualize_q(
    S=UNSET,
    n=UNSET,
    Q=UNSET,
    box_periodic_flag=False,
    name="Q",
    grid_normal=(0, 0, 1),
    director_spacing="medium",
    is_visualize_lines=True,
    save_path=None,
    is_off_screen=False,
    logger=None,
):
    """Create a standard Q-field visualization with minimal configuration.

    Supply either a full Q-tensor field through ``Q`` or a director field
    through ``n``. ``S`` may accompany ``n``. The function constructs a
    :class:`QFieldObject`, detects its defects, draws either smoothed
    disclination lines or defect points, overlays the field bounds, and adds
    one director plane using automatically scaled visual parameters.

    Parameters
    ----------
    S, n, Q : array_like or UNSET, optional
        Field data used to construct the :class:`QFieldObject`. ``Q`` is
        mutually exclusive with ``n`` and ``S``. When ``Q`` is omitted, ``n``
        is required and ``S`` is optional.
    box_periodic_flag : bool or length-3 bool-like, optional
        Periodic-boundary specification passed to :class:`QFieldObject`.
    name : str, optional
        Name of the created Q-field object.
    grid_normal : array_like, shape (3,), optional
        Nonzero normal vector of the displayed director plane.
    director_spacing : {"dense", "medium", "sparse"}, optional
        Preset controlling director-plane sampling density and glyph size.
    is_visualize_lines : bool, optional
        Draw smoothed disclination lines when ``True``; otherwise draw raw
        detected defect points.
    save_path : path-like or None, optional
        Save the rendered figure when provided. Parent directories are created
        automatically.
    is_off_screen : bool, optional
        Create the figure in off-screen mode. When ``True`` and ``save_path``
        is omitted, the call produces no useful output and returns
        ``(None, None)`` after logging a warning.

    Returns
    -------
    q_obj : QFieldObject or None
        The constructed field object, or ``None`` for the ignored off-screen
        call described above.
    figure : PlotFigure or None
        The created figure, or ``None`` for the ignored off-screen call.
    """
    is_visualize_lines = as_bool(is_visualize_lines, name="is_visualize_lines")
    is_off_screen = as_bool(is_off_screen, name="is_off_screen")
    grid_normal = _validate_grid_normal(grid_normal)

    if is_off_screen and save_path is None:
        logger.warning(
            "quick_visualize_q was called with is_off_screen=True but "
            "save_path=None. No visible window or saved image will be produced, "
            "so this call is ignored."
        )
        return None, None

    is_Q_provided = Q is not None and Q is not UNSET
    is_S_provided = S is not None and S is not UNSET
    is_n_provided = n is not None and n is not UNSET

    if is_Q_provided and (is_n_provided or is_S_provided):
        raise ValueError("Provide either `Q` or (`n`, optional `S`), not both.")
    if is_S_provided and not is_n_provided:
        raise ValueError("`S` may only be provided together with `n`.")
    if not is_Q_provided and not is_n_provided:
        raise ValueError("Provide either `Q` or `n`.")

    field_for_shape = Q if is_Q_provided else n
    params = _auto_quick_q_visual_params(field_for_shape)
    director_spacing_config = _resolve_director_spacing_level(director_spacing)

    q_obj = QFieldObject(
        Q=Q if is_Q_provided else UNSET,
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
        grid_normal=tuple(grid_normal),
        grid_spacing=(
            params["grid_spacing"] * director_spacing_config["grid_spacing_scale"]
        ),
        grid_size=params["grid_size"],
        grid_origin=params["grid_origin"],
        n_length=params["n_length"] * director_spacing_config["n_length_scale"],
        n_radius=params["n_radius"] * director_spacing_config["n_radius_scale"],
        figure=figure,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.act_savefig(save_path)

    return q_obj, figure
