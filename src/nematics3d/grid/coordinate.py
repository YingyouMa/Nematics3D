"""Coordinate-grid generation helpers."""

from collections.abc import Sequence

import numpy as np

from ..datatypes import as_grid_shape, as_number, as_str


def generate_coordinate_grid(
    shape_source: Sequence[int],
    shape_target: Sequence[int],
) -> np.ndarray:
    """Generate target-grid coordinates in the source index space.

    Parameters
    ----------
    shape_source : sequence of int
        Shape of the source array in N dimensions.
    shape_target : sequence of int
        Shape of the target sampling grid in the same N dimensions.

    Returns
    -------
    numpy.ndarray
        Floating coordinate grid with shape ``(*shape_target, N)``. Each final
        axis entry gives the corresponding coordinate in the source index space.

    Notes
    -----
    When source and target shapes are identical, the result is the ordinary
    integer lattice represented as floating coordinates. Otherwise each target
    axis spans the full source-index interval ``[0, source_size - 1]``.

    A target dimension of length one samples source coordinate zero, matching
    ``numpy.linspace(0, source_size - 1, 1)``.
    """
    shape_source = as_grid_shape(shape_source, name="shape_source")
    shape_target = as_grid_shape(shape_target, name="shape_target")

    ndim = len(shape_source)
    if ndim != len(shape_target):
        raise ValueError(
            "shape_source and shape_target must have the same number of dimensions"
        )

    if shape_source == shape_target:
        return np.moveaxis(np.indices(shape_target, dtype=float), 0, -1)

    grid = np.empty((*shape_target, ndim), dtype=float)
    for axis_index, (source_size, target_size) in enumerate(
        zip(shape_source, shape_target)
    ):
        axis = np.linspace(0.0, source_size - 1.0, target_size, dtype=float)
        reshape = [1] * ndim
        reshape[axis_index] = target_size
        grid[..., axis_index] = axis.reshape(reshape)

    return grid


def generate_fixed_step_grid(
    size1: float,
    size2: float,
    step1: float,
    step2: float,
    alignment: str = "bottom-left",
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Generate a two-dimensional grid with fixed physical step sizes."""
    size1 = float(as_number(size1, name="size1", value_range=(0.0, np.inf)))
    size2 = float(as_number(size2, name="size2", value_range=(0.0, np.inf)))
    step1 = float(as_number(step1, name="step1", value_range=(1e-12, np.inf)))
    step2 = float(as_number(step2, name="step2", value_range=(1e-12, np.inf)))
    alignment = as_str(
        alignment,
        name="alignment",
        pool=("bottom-left", "center"),
    )

    if alignment == "bottom-left":
        n1 = int(np.floor(size1 / step1)) + 1
        n2 = int(np.floor(size2 / step2)) + 1
        origin_index1 = 0
        origin_index2 = 0
        size1_eff = (n1 - 1) * step1
        size2_eff = (n2 - 1) * step2
    else:
        n1_half = int(np.floor(size1 / (2.0 * step1)))
        n2_half = int(np.floor(size2 / (2.0 * step2)))
        n1 = 2 * n1_half + 1
        n2 = 2 * n2_half + 1
        origin_index1 = n1_half
        origin_index2 = n2_half
        size1_eff = 2.0 * n1_half * step1
        size2_eff = 2.0 * n2_half * step2

    grid_int = np.moveaxis(np.indices((n1, n2), dtype=np.intp), 0, -1)
    grid = grid_int.astype(float)
    grid[..., 0] = (grid[..., 0] - origin_index1) * step1
    grid[..., 1] = (grid[..., 1] - origin_index2) * step2

    return grid, grid_int, (float(size1_eff), float(size2_eff))
