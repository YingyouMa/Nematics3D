"""Periodic box sizes for the three spatial dimensions."""

import numpy as np

from .dimension_info import DimensionInfo, as_dimension_info


# One box size applies to x, y, and z; three sizes apply to the axes in order.
# Positive finite values mark periodic axes, while +inf marks non-periodic axes.
BoxSizePeriodic = DimensionInfo


def as_box_size_periodic(
    input_data: BoxSizePeriodic,
    name: str = "box_size_periodic",
) -> np.ndarray:
    """Normalize periodic box sizes into a floating ``(x, y, z)`` array.

    A positive finite value is the period of its axis. Positive infinity marks
    a non-periodic axis. One input value is shared by all axes, while three
    values are assigned to x, y, and z in order.
    """
    raw_values = np.asarray(input_data, dtype=object)
    if any(isinstance(value, (bool, np.bool_)) for value in raw_values.reshape(-1)):
        raise TypeError(f"{name!r} must contain box sizes, not boolean values.")

    values = as_dimension_info(input_data, name=name)
    values = values.astype(float, copy=False)
    if np.isnan(values).any():
        raise ValueError(f"{name!r} must not contain NaN. Got {values!r}.")
    if np.isneginf(values).any():
        raise ValueError(
            f"{name!r} may use only positive infinity for non-periodic axes. "
            f"Got {values!r}."
        )
    if np.any(values <= 0.0):
        raise ValueError(f"{name!r} finite box sizes must be positive. Got {values!r}.")
    return values.copy()
