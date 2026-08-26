"""Coordinate-grid generation helpers."""

from typing import Tuple

import numpy as np


def generate_coordinate_grid(
    shape_source: Tuple[int, ...], shape_target: Tuple[int, ...]
) -> np.ndarray:
    """Generate an N-dimensional coordinate grid over the source domain.

    Parameters
    ----------
    shape_source : tuple of int
        Shape of the original data in N dimensions.
    shape_target : tuple of int
        Desired shape of the resampled grid in N dimensions.

    Returns
    -------
    grid : numpy.ndarray
        Continuous source coordinates with shape ``(*shape_target, N)``.
    grid_int : numpy.ndarray
        Integer target-grid coordinates with the same shape as ``grid``.
    steps : numpy.ndarray
        Source-coordinate step along each dimension.
    """
    ndim = len(shape_source)
    if ndim != len(shape_target):
        raise ValueError(
            "shape_source and shape_target must have the same number of dimensions"
        )

    axes = [np.linspace(0, s - 1, t) for s, t in zip(shape_source, shape_target)]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid = np.stack(mesh, axis=-1)

    axes_int = [np.arange(t) for t in shape_target]
    mesh_int = np.meshgrid(*axes_int, indexing="ij")
    grid_int = np.stack(mesh_int, axis=-1)
    grid_int = np.asarray(grid_int)

    steps = np.array(
        [
            (s - 1) / (t - 1) if t > 1 else 0.0
            for s, t in zip(shape_source, shape_target)
        ],
        dtype=float,
    )

    return grid, grid_int, steps


def generate_fixed_step_grid(
    size1: float,
    size2: float,
    step1: float,
    step2: float,
    alignment: str = "bottom-left",
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Generate a two-dimensional coordinate grid with fixed step sizes.

    ``alignment='bottom-left'`` starts both axes at zero. ``'center'`` keeps
    zero as an actual grid point and expands symmetrically around it. Return
    the continuous grid, integer index grid, and effective covered sizes.
    """
    alignment = str(alignment)
    if alignment == "bottom-left":
        n1 = int(np.floor(size1 / step1)) + 1
        n2 = int(np.floor(size2 / step2)) + 1

        axis1 = np.arange(n1, dtype=float) * step1
        axis2 = np.arange(n2, dtype=float) * step2
        axis1_int = np.arange(n1)
        axis2_int = np.arange(n2)

        size1_eff = (n1 - 1) * step1
        size2_eff = (n2 - 1) * step2

    elif alignment == "center":
        n1_half = int(np.floor(size1 / step1 / 2))
        n2_half = int(np.floor(size2 / step2 / 2))

        axis1 = np.arange(-n1_half, n1_half + 1, dtype=float) * step1
        axis2 = np.arange(-n2_half, n2_half + 1, dtype=float) * step2
        axis1_int = np.arange(axis1.shape[0])
        axis2_int = np.arange(axis2.shape[0])

        size1_eff = 2 * n1_half * step1
        size2_eff = 2 * n2_half * step2
    else:
        raise ValueError(
            f"alignment must be 'bottom-left' or 'center', got {alignment!r}"
        )

    mesh = np.meshgrid(axis1, axis2, indexing="ij")
    grid = np.stack(mesh, axis=-1)

    mesh_int = np.meshgrid(axis1_int, axis2_int, indexing="ij")
    grid_int = np.stack(mesh_int, axis=-1)

    return grid, grid_int, (size1_eff, size2_eff)
