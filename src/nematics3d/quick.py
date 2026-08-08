"""Quick convenience workflows for common Nematics3D visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    import sys

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_REPO_ROOT / "src"))

    from nematics3d.classes.q_field_object import QFieldObject
    from nematics3d.classes.visual.plot_figure import PlotFigure
    from nematics3d.classes.visual.plot_sphere import OptsSphere, PlotSphere
    from nematics3d.classes.visual.plot_tube import OptsTube
    from nematics3d.datatypes import UNSET
    from nematics3d.logging_decorator import logging_and_warning_decorator
else:
    from .classes.q_field_object import QFieldObject
    from .classes.visual.plot_figure import PlotFigure
    from .classes.visual.plot_sphere import OptsSphere, PlotSphere
    from .classes.visual.plot_tube import OptsTube
    from .datatypes import UNSET
    from .logging_decorator import logging_and_warning_decorator


def _resolve_director_spacing_level(level):
    spacing_config_by_level = {
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

    try:
        return spacing_config_by_level[level]
    except KeyError as exc:
        valid_levels = ", ".join(repr(key) for key in spacing_config_by_level)
        raise ValueError(
            f"`director_spacing` must be one of {valid_levels}, got {level!r}."
        ) from exc


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

    if not is_Q_provided and not is_n_provided:
        raise ValueError("Provide `Q` or `n`.")

    field_for_shape = Q if is_Q_provided else n
    params = _auto_quick_Q_visual_params(field_for_shape, grid_normal)
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
        grid_normal=grid_normal,
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


def _find_repo_root_for_quick_demo():
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "example" / "data" / "Q_example_workflow.npy").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the repository root for the quick_visualize_q demo."
    )


def _run_quick_visualize_q_tutorial_demo():
    repo_root = _find_repo_root_for_quick_demo()
    data_path = repo_root / "example" / "data" / "Q_example_workflow.npy"
    output_dir = repo_root / "tutorials" / "output" / "quick_visualize_q"
    save_path = output_dir / "quick_py_main_preview.png"

    Q_data = np.load(data_path)

    # Edit these values directly when using this file as a quick local driver.
    demo_kwargs = {
        "Q": Q_data,
        "name": "quick_py_main_demo",
        "grid_normal": (0, 0, 1),
        "director_spacing": "sparse",
        "is_visualize_lines": True,
        "save_path": save_path,
        "is_off_screen": False,
    }

    q_obj, figure = quick_visualize_q(**demo_kwargs)
    print(f"Loaded tutorial data from: {data_path}")
    print(f"Saved preview image to: {save_path}")
    return q_obj, figure


if __name__ == "__main__":
    _run_quick_visualize_q_tutorial_demo()
