"""Defect-index semantic alias and runtime converter."""

import numpy as np

from .number import as_number


# A DefectIndex is an array of shape (N, 3) in lattice-index coordinates.
# Each row identifies a plaquette center through exactly one integer coordinate
# and two half-integer coordinates.
DefectIndex = np.ndarray


def as_defect_index(
    input_data,
    name: str = "defect indices",
    *,
    tolerance: float = 1e-8,
) -> DefectIndex:
    """Validate and canonicalize an array of defect indices.

    The returned array has shape (N, 3) and floating-point dtype. Every
    coordinate is snapped to its exact integer or half-integer lattice value,
    and every row contains exactly one integer and two half-integers.
    Empty input with shape (0, 3) is valid.
    """
    tolerance = as_number(
        tolerance,
        name="tolerance",
        value_range=(0.0, np.inf),
    )

    raw_values = np.asarray(input_data)
    if raw_values.ndim != 2 or raw_values.shape[1:] != (3,):
        raise ValueError(
            f"{name!r} must have shape (N, 3). Got shape {raw_values.shape}."
        )
    if not np.issubdtype(raw_values.dtype, np.number) or np.iscomplexobj(raw_values):
        raise TypeError(
            f"{name!r} must contain only real numbers. Got dtype "
            f"{raw_values.dtype}."
        )

    values = np.asarray(raw_values, dtype=float)
    if not np.all(np.isfinite(values)):
        invalid_rows = np.flatnonzero(~np.all(np.isfinite(values), axis=1))
        raise ValueError(
            f"{name!r} must contain only finite values. Invalid row indices "
            f"include {invalid_rows[:5].tolist()}."
        )

    doubled_values = 2.0 * values
    doubled_rounded = np.rint(doubled_values)
    is_on_half_grid = np.isclose(
        doubled_values,
        doubled_rounded,
        rtol=0.0,
        atol=2.0 * tolerance,
    )
    invalid_grid_rows = ~np.all(is_on_half_grid, axis=1)
    if np.any(invalid_grid_rows):
        row = int(np.flatnonzero(invalid_grid_rows)[0])
        raise ValueError(
            f"Every coordinate in {name!r} must be an integer or half-integer "
            f"within tolerance {tolerance}. Got {values[row]!r} at row {row}."
        )

    doubled_integer = doubled_rounded.astype(np.int64, copy=False)
    integer_count = np.sum((doubled_integer & 1) == 0, axis=1)
    invalid_structure_rows = integer_count != 1
    if np.any(invalid_structure_rows):
        row = int(np.flatnonzero(invalid_structure_rows)[0])
        raise ValueError(
            f"Every row in {name!r} must contain exactly one integer coordinate "
            f"and two half-integer coordinates. Got {values[row]!r} at row {row}."
        )

    return doubled_integer.astype(float) / 2.0
