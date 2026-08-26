"""Semantic vector annotation and runtime converter."""

import numbers
from typing import Sequence, Union

import numpy as np

from ..logging_decorator import logging_and_warning_decorator


# Vect(d) is a semantic annotation for a vector with d components. The
# dimension is documented for readers; as_vector() enforces it at runtime.
def Vect(d):  # noqa: N802 - compact semantic annotation used as Vect(d)
    return Union[Sequence[numbers.Real], np.ndarray]


@logging_and_warning_decorator(start_finish_level=5)
def as_vector(
    input_data,
    d=3,
    name="input data",
    is_normalized=False,
    is_zero_allowed=True,
    replace=None,
    logger=None,
):
    """Validate and normalize a vector with exactly ``d`` components.

    Parameters
    ----------
    input_data : Vect(d)
        Input vector.
    d : int, optional
        Required number of components. Any positive dimension is supported.
    name : str, optional
        Human-readable input name used in error and recovery messages.
    is_normalized : bool, optional
        Whether to return a unit vector. A zero vector cannot be normalized.
    is_zero_allowed : bool, optional
        Whether a zero vector is valid when normalization is not requested.
    replace : Vect(d) or None, optional
        Fallback vector used when ``input_data`` is invalid. The replacement
        must satisfy the same validation rules.

    Returns
    -------
    numpy.ndarray
        Floating-point vector with shape ``(d,)``.

    Raises
    ------
    TypeError
        If ``d`` is not an integer or a value is not a real number.
    ValueError
        If ``d`` is not positive, the shape is wrong, a value is not finite,
        or the zero-vector rules are violated.
    """
    if isinstance(d, bool) or not isinstance(d, numbers.Integral):
        raise TypeError(f"'d' must be an integer. Got {d!r}.")
    d = int(d)
    if d <= 0:
        raise ValueError(f"'d' must be positive. Got {d}.")

    def validate(value):
        raw_value = np.asarray(value)
        if raw_value.shape != (d,):
            raise ValueError(
                f"{name!r} must have shape ({d},). Got shape {raw_value.shape}."
            )
        if not all(isinstance(component, numbers.Real) for component in raw_value):
            raise TypeError(f"{name!r} must contain only real numbers. Got {value!r}.")

        vector = np.asarray(raw_value, dtype=float)
        if not np.isfinite(vector).all():
            raise ValueError(
                f"{name!r} must contain only finite values. Got {value!r}."
            )

        norm = float(np.linalg.norm(vector))
        if is_normalized and norm <= 1e-12:
            raise ValueError(
                f"{name!r} must be non-zero when normalization is requested."
            )
        if not is_zero_allowed and norm <= 1e-12:
            raise ValueError(f"{name!r} must be a non-zero vector. Got {value!r}.")
        if is_normalized:
            vector = vector / norm
        return vector

    try:
        return validate(input_data)
    except (TypeError, ValueError):
        if replace is None:
            raise

        logger.exception(f"Invalid {name!r}; attempting the configured replacement.")
        logger.recovery(f"Use {replace!r} as {name!r} in the following.")
        return validate(replace)
