"""String runtime validation helper."""

from ..logging_decorator import logging_and_warning_decorator


def _validate_str(value, *, name="input_data", pool=None):
    """Validate one string against an optional allowed-value pool."""
    if not isinstance(value, str):
        raise TypeError(
            f"{name!r} must be a string. "
            f"Got {value!r} with type {type(value).__name__}."
        )
    if pool is not None and value not in pool:
        raise ValueError(f"{name!r} must be one of {pool!r}. Got {value!r}.")
    return value


@logging_and_warning_decorator(start_finish_level=5)
def as_str(input_data, name="input_data", pool=None, replace=None, logger=None):
    """Validate a string with an optional allowed-value pool and fallback.

    ``replace`` is validated by the same rules as ``input_data`` before it is
    used as a recovery value.
    """
    try:
        return _validate_str(input_data, name=name, pool=pool)
    except (TypeError, ValueError):
        if replace is None:
            raise

        logger.exception(f"Invalid value for {name!r}.")
        replacement = _validate_str(replace, name=f"{name} replacement", pool=pool)
        logger.recovery(
            f"Use replacement {replacement!r} for {name!r} in the following."
        )
        return replacement
