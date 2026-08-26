"""Runtime converter for scalar boolean inputs."""

import numbers

import numpy as np

from ..logging_decorator import logging_and_warning_decorator


@logging_and_warning_decorator(start_finish_level=5)
def as_bool(
    input_data,
    name: str = "input data",
    *,
    replace=None,
    logger=None,
) -> bool:
    """Validate and normalize one boolean-like scalar.

    Python and NumPy booleans are accepted directly. Real numeric scalars are
    accepted only when equal to zero or one. The result is always a Python
    ``bool``. If ``replace`` is provided, it must satisfy the same contract.
    """

    def validate(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, numbers.Real):
            if value in (0, 1):
                return bool(value)
            raise ValueError(
                f"{name!r} must be numerically equal to 0 or 1. Got {value!r}."
            )
        raise TypeError(
            f"{name!r} must be a boolean or numerically equal to 0 or 1. "
            f"Got {value!r}."
        )

    try:
        return validate(input_data)
    except (TypeError, ValueError):
        if replace is None:
            raise
        logger.exception(f"Invalid {name!r}; attempting the configured replacement.")
        logger.recovery(f"Use {replace!r} as {name!r} in the following.")
        return validate(replace)
