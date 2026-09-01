"""Color helpers for visualization."""

from collections.abc import Callable
from typing import Any

import numpy as np


def blue_red_in_white_bg() -> np.ndarray:
    """Return L2-normalized RGB colors from blue to red for a white background."""
    t = np.linspace(0.0, 1.0, 256)
    blue_to_green = np.column_stack((np.zeros_like(t), t, 1.0 - t))
    green_to_red = np.column_stack((t[1:], 1.0 - t[1:], np.zeros_like(t[1:])))
    colors = np.vstack((blue_to_green, green_to_red))
    colors /= np.linalg.norm(colors, axis=1, keepdims=True)
    return colors


def _boy_polynomial(n: np.ndarray) -> np.ndarray:
    """Return the original Nematics3D Boy polynomial before color transform."""
    n = np.asarray(n, dtype=float)
    if n.shape[-1] != 3:
        raise ValueError("n must have shape (..., 3)")
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("director vectors must be nonzero")
    n = n / norm

    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
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
    return p


def _apply_boy_affine(n: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    p = _boy_polynomial(n)
    result = np.einsum("...i,ji->...j", p, A) + b
    tol = 1e-10
    if np.any(result < -tol) or np.any(result > 1.0 + tol):
        raise ValueError("Director colormap produced RGB values outside [0, 1]")
    return np.clip(result, 0.0, 1.0)


def director_color_pareto_034(n: np.ndarray) -> np.ndarray:
    """Map directors to RGB with the previous sRGB Pareto solution at J_norm=0.34."""
    A = np.array(
        [
            [0.52868237, 0.05496927, 0.19053080],
            [-0.31232790, 0.42594150, 0.18920397],
            [-0.21649449, -0.48049833, 0.19067196],
        ],
        dtype=float,
    )
    b = np.array([0.39350715, 0.39366654, 0.39346801], dtype=float)
    return _apply_boy_affine(n, A, b)


def director_color_pareto_oklab_043(n: np.ndarray) -> np.ndarray:
    """Map directors to RGB with the selected OKLab Pareto knee at J_loc≈0.43.

    The affine map itself remains in encoded sRGB and therefore produces normal
    display RGB values.  The optimization used OKLab for both the axis-color
    distance and the local tangent-metric uniformity objective, while retaining
    the hard global sRGB gamut constraint.

    The selected point is the maximum-distance knee of the normalized Pareto
    curve over J_loc in [0.30, 0.76].  Before export, the red and blue affine
    rows were contracted by less than 4.4e-5 and shifted by about 1e-5 so the
    continuous channel extrema lie strictly inside [0, 1].
    """
    A = np.array(
        [
            [0.5015275525743265, 0.0814208604228778, 0.4289041134674454],
            [-0.2426021621724047, 0.3331598986797062, 0.2349957727825471],
            [-0.2761140886939694, -0.3089204069097675, 0.4083459067954693],
        ],
        dtype=float,
    )
    b = np.array(
        [0.3805938025338775, 0.4480306832558079, 0.4044714907877873],
        dtype=float,
    )
    return _apply_boy_affine(n, A, b)


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
    """Plot a director-color sphere and x/y/z arrows for a colormap function."""
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

    return {"figure": figure, "surface": surface, "axes": axes}
