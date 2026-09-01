"""Color helpers for visualization."""

from collections.abc import Callable
from typing import Any

import numpy as np


def blue_red_in_white_bg() -> np.ndarray:
    """Return L2-normalized RGB colors from blue to red for a white background.

    The table contains 511 colors: 256 samples from blue to green followed by
    255 samples from green to red, with the shared green endpoint included only
    once. Each RGB vector is normalized to unit Euclidean norm so mixed colors
    remain visually strong on a white background.
    """
    t = np.linspace(0.0, 1.0, 256)

    blue_to_green = np.column_stack(
        (
            np.zeros_like(t),
            t,
            1.0 - t,
        )
    )
    green_to_red = np.column_stack(
        (
            t[1:],
            1.0 - t[1:],
            np.zeros_like(t[1:]),
        )
    )

    colors = np.vstack((blue_to_green, green_to_red))
    colors /= np.linalg.norm(colors, axis=1, keepdims=True)

    return colors


def director_color_pareto_034(n: np.ndarray) -> np.ndarray:
    """Map directors to RGB with the Stage-I Pareto solution at J_norm = 0.34.

    This is the current comparison candidate for the RP2 colormap project.  It
    uses the same Boy-surface polynomial as :func:`nematics3d.field.n_color_immerse`
    but replaces the empirical Nematics3D affine color transform with the
    numerically optimized affine map found on the J_norm = 0.34 Pareto point.

    Parameters
    ----------
    n : array_like, shape (..., 3)
        Director vectors. Nonzero vectors are normalized internally.

    Returns
    -------
    numpy.ndarray, shape (..., 3)
        RGB colors.
    """
    n = np.asarray(n, dtype=float)
    if n.shape[-1] != 3:
        raise ValueError("n must have shape (..., 3)")

    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("director vectors must be nonzero")
    n = n / norm

    x = n[..., 0]
    y = n[..., 1]
    z = n[..., 2]
    x2 = x * x
    y2 = y * y
    z2 = z * z

    p = np.empty(n.shape, dtype=float)
    p[..., 0] = 0.5 * (
        (2.0 * x2 - y2 - z2)
        + 2.0 * y * z * (y2 - z2)
        + z * x * (x2 - z2)
        + x * y * (y2 - x2)
    )
    p[..., 1] = (7.0 / 8.0) * (
        (y2 - z2) + z * x * (z2 - x2) + x * y * (y2 - x2)
    )
    p[..., 2] = (1.0 / 8.0) * (x + y + z) * (
        (x + y + z) ** 3 + 4.0 * (y - x) * (z - y) * (x - z)
    )

    A = np.array(
        [
            [0.52868237, 0.05496927, 0.19053080],
            [-0.31232790, 0.42594150, 0.18920397],
            [-0.21649449, -0.48049833, 0.19067196],
        ],
        dtype=float,
    )
    b = np.array([0.39328370, 0.39340467, 0.39325923], dtype=float)

    return np.einsum("...i,ji->...j", p, A) + b


def plot_director_color_sphere(
    color_func: Callable[[np.ndarray], Any],
    *,
    figure=None,
    radius: float = 1.0,
    theta_resolution: int = 120,
    phi_resolution: int = 120,
    axis_length: float = 1.55,
    is_off_screen: bool = False,
    figure_size: tuple[int, int] = (1800, 1800),
):
    """Plot a director-color sphere and x/y/z arrows for a colormap function.

    The sphere position itself is the director direction.  Each surface point
    ``n`` is colored with ``color_func(n)``.  The three positive Cartesian
    arrows are colored by the same function evaluated at ``e_x``, ``e_y`` and
    ``e_z``.  This makes different director-to-RGB maps directly comparable.

    Parameters
    ----------
    color_func : callable
        Function accepting an array of directors with shape ``(..., 3)`` and
        returning RGB values with matching shape ``(..., 3)``.  Examples are
        ``nematics3d.field.n_color_immerse`` (the original Nematics3D map) and
        :func:`director_color_pareto_034` (the current Pareto comparison map).
    figure : PlotFigure, optional
        Existing figure to draw into.  If omitted, a new ``PlotFigure`` is
        created.
    radius : float, optional
        Radius of the color sphere.
    theta_resolution, phi_resolution : int, optional
        PyVista sphere resolutions.
    axis_length : float, optional
        Length of the x/y/z arrows.
    is_off_screen : bool, optional
        Used only when this function creates the figure.
    figure_size : tuple[int, int], optional
        Figure size used only when this function creates the figure.

    Returns
    -------
    dict
        Dictionary with ``figure``, ``surface`` and ``axes`` entries.

    Examples
    --------
    >>> from nematics3d.field import n_color_immerse
    >>> scene_original = plot_director_color_sphere(n_color_immerse)
    >>> scene_pareto = plot_director_color_sphere(director_color_pareto_034)
    """
    # Local imports keep the lightweight color utilities usable without
    # importing PyVista and the visualization stack at module import time.
    import pyvista as pv

    from .plot_figure import PlotFigure
    from .plot_polydata import PlotPolyData
    from .plot_vector import PlotVector

    if figure is None:
        figure = PlotFigure(
            is_off_screen=is_off_screen,
            name="director_color_sphere",
            size=figure_size,
        )

    sphere = pv.Sphere(
        radius=radius,
        theta_resolution=theta_resolution,
        phi_resolution=phi_resolution,
    )
    points = np.asarray(sphere.points, dtype=float)
    directors = points / np.linalg.norm(points, axis=1, keepdims=True)

    colors = np.asarray(color_func(directors), dtype=float)
    if colors.shape != directors.shape:
        raise ValueError(
            "color_func must return RGB values with the same shape as its "
            f"director input; got {colors.shape}, expected {directors.shape}"
        )
    if not np.all(np.isfinite(colors)):
        raise ValueError("color_func returned non-finite RGB values")

    surface = PlotPolyData(
        polydata=sphere,
        figure=figure,
        name="director_color_sphere",
        category="director_color_sphere",
        color=colors,
        opacity=1.0,
        ambient=0.32,
        diffuse=0.82,
        specular=0.18,
        specular_power=18,
        is_pickable=False,
        is_reset_camera=False,
    )

    axis_directors = np.eye(3, dtype=float)
    axis_colors = np.asarray(color_func(axis_directors), dtype=float)
    if axis_colors.shape != (3, 3):
        raise ValueError(
            "color_func must map the three axis directors to a (3, 3) RGB array"
        )

    axes = PlotVector(
        coords=np.zeros((3, 3), dtype=float),
        orient=axis_length * axis_directors,
        figure=figure,
        name="director_color_axes",
        category="director_color_axes",
        color=axis_colors,
        opacity=1.0,
        length=lambda orient_length: orient_length,
        radius=0.04 * radius,
        tip_length_fraction=0.24,
        tip_radius_ratio=2.9,
        anchor="tail",
        sides=20,
        ambient=0.25,
        diffuse=0.88,
        specular=0.12,
        is_pickable=False,
        is_reset_camera=False,
    )

    figure.pl.view_vector((1.7, 1.55, 1.2), viewup=(0.0, 0.0, 1.0))
    figure.pl.camera.zoom(0.85)
    figure.pl.render()

    return {
        "figure": figure,
        "surface": surface,
        "axes": axes,
    }
