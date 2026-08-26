"""Real-number semantic alias and runtime converter."""

import numbers

import numpy as np

from ..logging_decorator import logging_and_warning_decorator
from .bool import as_bool


# Reader-facing semantic alias for real scalar values. Runtime conversion
# deliberately rejects booleans even though bool is a numbers.Real subclass.
Number = numbers.Real


def as_value_range(value_range, *, name: str = "value range") -> tuple[float, float]:
    """Validate and normalize an inclusive numeric interval."""
    raw_range = np.asarray(value_range)
    if raw_range.shape != (2,):
        raise ValueError(f"{name!r} must have shape (2,). Got shape {raw_range.shape}.")
    if not np.issubdtype(raw_range.dtype, np.number) or np.iscomplexobj(raw_range):
        raise TypeError(
            f"{name!r} must contain only real numbers. Got dtype " f"{raw_range.dtype}."
        )

    lower, upper = (float(value) for value in raw_range)
    if np.isnan(lower) or np.isnan(upper):
        raise ValueError(f"{name!r} must not contain NaN. Got {value_range!r}.")
    if upper <= lower:
        raise ValueError(f"{name!r} must be strictly increasing. Got {value_range!r}.")
    return lower, upper


@logging_and_warning_decorator(start_finish_level=5)
def as_number(
    input_data,
    name: str = "input data",
    *,
    is_integer: bool = False,
    is_nan_allowed: bool = False,
    is_infinite_allowed: bool = False,
    value_range=None,
    is_clipped: bool = False,
    replace=None,
    logger=None,
) -> int | float:
    """Validate and normalize one real number.

    Ordinary values are returned as Python ``float`` objects. Integer mode
    accepts integer-valued real inputs and returns a Python ``int``. NaN,
    infinity, and boolean inputs are rejected by default. If ``replace`` is
    provided, it must satisfy exactly the same contract as the original input.
    """
    is_integer = as_bool(is_integer, name="is_integer")
    is_nan_allowed = as_bool(is_nan_allowed, name="is_nan_allowed")
    is_infinite_allowed = as_bool(
        is_infinite_allowed,
        name="is_infinite_allowed",
    )
    is_clipped = as_bool(is_clipped, name="is_clipped")

    normalized_range = None
    if value_range is not None:
        normalized_range = as_value_range(
            value_range,
            name=f"{name} value range",
        )
        if is_integer and is_clipped:
            finite_bounds = [bound for bound in normalized_range if np.isfinite(bound)]
            if any(not bound.is_integer() for bound in finite_bounds):
                raise ValueError(
                    f"{name!r} uses integer clipping, so every finite value-range "
                    f"boundary must be integer-valued. Got {value_range!r}."
                )

    def validate(value):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name!r} must be a real number, not boolean.")
        if not isinstance(value, numbers.Real):
            raise TypeError(f"{name!r} must be a real number. Got {value!r}.")

        numeric_value = float(value)
        if np.isnan(numeric_value) and not is_nan_allowed:
            raise ValueError(f"{name!r} must not be NaN.")
        if np.isinf(numeric_value) and not is_infinite_allowed:
            raise ValueError(f"{name!r} must be finite. Got {value!r}.")

        if is_integer:
            if not np.isfinite(numeric_value) or not numeric_value.is_integer():
                raise TypeError(
                    f"{name!r} must be an integer-valued finite number. "
                    f"Got {value!r}."
                )
            result: int | float = int(numeric_value)
        else:
            result = numeric_value

        if normalized_range is not None and not (
            normalized_range[0] <= result <= normalized_range[1]
        ):
            if not is_clipped:
                raise ValueError(
                    f"{name!r} must be in the inclusive range "
                    f"[{normalized_range[0]}, {normalized_range[1]}]. Got {result}."
                )
            result = min(max(result, normalized_range[0]), normalized_range[1])
            result = int(result) if is_integer else float(result)
            logger.warning(f"Clipped {name!r} to {result!r}.")

        return result

    try:
        return validate(input_data)
    except (TypeError, ValueError):
        if replace is None:
            raise
        logger.exception(f"Invalid {name!r}; attempting the configured replacement.")
        logger.recovery(f"Use {replace!r} as {name!r} in the following.")
        return validate(replace)
