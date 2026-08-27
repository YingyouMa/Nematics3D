"""Coordinate-grid generation helpers."""

from collections.abc import Sequence

import numpy as np


def _as_positive_int_shape(shape, *, name: str) -> tuple[int, ...]:
    """Return one validated shape made of positive integral dimensions."""
    if isinstance(shape, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of positive integers.")

    try:
        values = tuple(shape)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of positive integers.") from exc

    if not values:
        raise ValueError(f"{name} must contain at least one dimension.")

    result = []
    for i, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(
                f"{name}[{i}] must be an integer, got {type(value).__name__}."
            )
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name}[{i}] must be positive, got {value}.")
        result.append(value)

    return tuple(result)


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
    shape_source = _as_positive_int_shape(shape_source, name="shape_source")
    shape_target = _as_positive_int_shape(shape_target, name="shape_target")

    ndim = len(shape_source)
    if ndim != len(shape_target):
        raise ValueError(
            "shape_source and shape_target must have the same number of dimensions"
        )

    # Common fast path: no resampling is needed. np.indices allocates the
    # coordinate payload directly without constructing a separate meshgrid list.
    if shape_source == shape_target:
        return np.moveaxis(np.indices(shape_target, dtype=float), 0, -1)

    # Allocate only the final dense coordinate grid. Each 1D coordinate axis is
    # broadcast directly into its final component, avoiding N full-size meshgrid
    # temporaries before stacking.
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
