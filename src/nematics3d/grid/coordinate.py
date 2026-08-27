"""Coordinate-grid generation helpers."""

from collections.abc import Sequence

import numpy as np

from ..datatypes import as_number, as_str


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
) -> tuple[np.ndarray, tuple[float, float]]:
    """Generate integer topology for a 2D grid with fixed physical step sizes.

    Parameters
    ----------
    size1, size2 : real
        Requested non-negative physical extents along the two grid axes.
    step1, step2 : real
        Strictly positive finite physical spacings along the two grid axes.
    alignment : {"bottom-left", "center"}, optional
        ``"bottom-left"`` starts at one corner and grows in the positive
        directions. ``"center"`` keeps the origin on an actual grid point and
        grows symmetrically around it.

    Returns
    -------
    grid_int : numpy.ndarray
        Integer grid topology with shape ``(n1, n2, 2)`` and integer dtype.
        ``grid_int[i, j] == (i, j)``.
    size_eff : tuple of float
        Effective physical extents actually covered by the discrete grid. The
        requested size is rounded down to the largest extent compatible with
        the fixed spacing and selected alignment.

    Notes
    -----
    This helper intentionally returns only integer topology plus effective
    extents. Physical coordinates are constructed by the owning plane geometry
    from the topology, basis vectors, spacing, and origin.
    """
    size1 = float(as_number(size1, name="size1"))
    size2 = float(as_number(size2, name="size2"))
    step1 = float(as_number(step1, name="step1"))
    step2 = float(as_number(step2, name="step2"))
    alignment = as_str(
        alignment,
        name="alignment",
        pool=("bottom-left", "center"),
    )

    if size1 < 0.0 or size2 < 0.0:
        raise ValueError("size1 and size2 must be non-negative.")
    if step1 <= 0.0 or step2 <= 0.0:
        raise ValueError("step1 and step2 must be strictly positive.")

    if alignment == "bottom-left":
        n1 = int(np.floor(size1 / step1)) + 1
        n2 = int(np.floor(size2 / step2)) + 1
        size1_eff = (n1 - 1) * step1
        size2_eff = (n2 - 1) * step2
    else:  # alignment == "center"
        n1_half = int(np.floor(size1 / (2.0 * step1)))
        n2_half = int(np.floor(size2 / (2.0 * step2)))
        n1 = 2 * n1_half + 1
        n2 = 2 * n2_half + 1
        size1_eff = 2.0 * n1_half * step1
        size2_eff = 2.0 * n2_half * step2

    # Only the integer topology is consumed downstream. np.indices constructs
    # that payload directly, avoiding the old float meshgrid and a second
    # integer meshgrid before stacking.
    grid_int = np.moveaxis(np.indices((n1, n2), dtype=np.intp), 0, -1)
    return grid_int, (float(size1_eff), float(size2_eff))
