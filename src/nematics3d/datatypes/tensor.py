"""Semantic tensor annotation and runtime converter."""

import numbers
from typing import Sequence, Union

import numpy as np

from ..logging_decorator import logging_and_warning_decorator


# Tensor(shape) is a semantic annotation for an array with the indicated shape.
# The shape is documented for readers; as_tensor() enforces it at runtime.
def Tensor(shape):  # noqa: N802 - compact semantic annotation used as Tensor(shape)
    return Union[Sequence, np.ndarray]


@logging_and_warning_decorator(start_finish_level=5)
def as_tensor(
    input_data,
    shape,
    name="input data",
    replace=None,
    logger=None,
):
    """Validate a real, finite tensor with exactly the requested shape.

    Parameters
    ----------
    input_data : Tensor(shape)
        Input tensor.
    shape : tuple of int
        Required tensor shape. Every dimension must be a positive integer.
    name : str, optional
        Human-readable input name used in error and recovery messages.
    replace : Tensor(shape) or None, optional
        Fallback tensor used when ``input_data`` is invalid. The replacement
        must satisfy the same validation rules.

    Returns
    -------
    numpy.ndarray
        Floating-point tensor with the requested shape.

    Raises
    ------
    TypeError
        If ``shape`` is not a tuple of integers or a value is not real.
    ValueError
        If ``shape`` is empty or contains a non-positive dimension, the input
        shape is wrong, or a value is not finite.
    """
    if not isinstance(shape, tuple):
        raise TypeError(f"'shape' must be a tuple of integers. Got {shape!r}.")
    if not shape:
        raise ValueError("'shape' must contain at least one dimension.")
    if any(
        isinstance(dimension, bool) or not isinstance(dimension, numbers.Integral)
        for dimension in shape
    ):
        raise TypeError(f"'shape' must contain only integers. Got {shape!r}.")
    shape = tuple(int(dimension) for dimension in shape)
    if any(dimension <= 0 for dimension in shape):
        raise ValueError(f"'shape' dimensions must be positive. Got {shape!r}.")

    def validate(value):
        raw_value = np.asarray(value)
        if raw_value.shape != shape:
            raise ValueError(
                f"{name!r} must have shape {shape}. Got shape {raw_value.shape}."
            )
        if not all(isinstance(component, numbers.Real) for component in raw_value.flat):
            raise TypeError(f"{name!r} must contain only real numbers. Got {value!r}.")

        tensor = np.asarray(raw_value, dtype=float)
        if not np.isfinite(tensor).all():
            raise ValueError(
                f"{name!r} must contain only finite values. Got {value!r}."
            )
        return tensor

    try:
        return validate(input_data)
    except (TypeError, ValueError):
        if replace is None:
            raise

        logger.exception(f"Invalid {name!r}; attempting the configured replacement.")
        logger.recovery(f"Use {replace!r} as {name!r} in the following.")
        return validate(replace)
