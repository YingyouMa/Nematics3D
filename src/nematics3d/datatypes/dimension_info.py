"""Shared or axis-specific information for the three spatial dimensions."""

import numbers
from typing import Sequence, Union

import numpy as np

from .bool import as_bool
from .number import Number


# One value applies to x, y, and z; three values apply to the axes in order.
DimensionInfo = Union[Number, Sequence[Number]]


def as_dimension_info(
    input_data: DimensionInfo,
    name: str = "dimension info",
    *,
    is_bool: bool = False,
) -> np.ndarray:
    """Expand shared or axis-specific information into an ``(x, y, z)`` array.

    Parameters
    ----------
    input_data : DimensionInfo
        One real scalar shared by all axes, or exactly three real values
        assigned to the x, y, and z axes, respectively.
    name : str, optional
        Parameter name used in validation messages.
    is_bool : bool, optional
        Require every axis value to be boolean or numerically equal to zero or
        one, then return an array with boolean dtype.

    Returns
    -------
    numpy.ndarray
        An independent one-dimensional array with shape ``(3,)``.

    Raises
    ------
    TypeError
        If the input contains non-real values or ``is_bool`` is not boolean.
    ValueError
        If the input is neither a scalar nor an array with shape ``(3,)``, or
        ``is_bool=True`` and a value is not zero or one.
    """
    is_bool = as_bool(is_bool, name="is_bool")

    raw_value = np.asarray(input_data)
    if raw_value.ndim == 0:
        raw_value = np.repeat(raw_value[None], 3)
    elif raw_value.shape != (3,):
        raise ValueError(
            f"{name!r} must be one value or exactly three values for the x, y, "
            f"and z axes. Got shape {raw_value.shape}."
        )

    if raw_value.dtype.kind == "O":
        if not all(isinstance(value, numbers.Real) for value in raw_value):
            raise TypeError(f"{name!r} must contain only real values.")
        raw_value = np.asarray(raw_value, dtype=float)
    elif not (
        np.issubdtype(raw_value.dtype, np.number)
        or np.issubdtype(raw_value.dtype, np.bool_)
    ) or np.iscomplexobj(raw_value):
        raise TypeError(
            f"{name!r} must contain only real values. Got dtype {raw_value.dtype}."
        )

    if is_bool:
        if not np.isin(raw_value, (0, 1)).all():
            raise ValueError(
                f"{name!r} must contain only boolean values or numeric 0/1. "
                f"Got {raw_value!r}."
            )
        return raw_value.astype(bool, copy=True)

    return raw_value.copy()
