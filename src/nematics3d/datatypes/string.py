"""String runtime validation helper."""

from ..logging_decorator import logging_and_warning_decorator


@logging_and_warning_decorator(start_finish_level=5)
def as_str(input_data, name="input_data", pool=None, replace=None, logger=None):
    """Validate a string with an optional allowed-value pool and fallback.

    Parameters
    ----------
    input_data : Any
        Value to validate.
    name : str, optional
        Human-readable name used in error messages.
    pool : iterable, optional
        Allowed string values. If ``None``, no membership check is performed.
    replace : Any, optional
        Fallback returned when validation fails. For backward compatibility,
        the replacement itself is not validated.
    """
    try:
        if not isinstance(input_data, str):
            raise TypeError(
                f"{name!r} must be a string. "
                f"Got {input_data!r} with type {type(input_data).__name__}."
            )
        if pool is not None and input_data not in pool:
            raise ValueError(
                f"{name!r} must be one of {pool!r}. Got {input_data!r}."
            )
    except (TypeError, ValueError):
        if replace is None:
            raise

        logger.exception(f"Invalid value for {name!r}.")
        logger.recovery(
            f"Use replacement {replace!r} for {name!r} in the following."
        )
        return replace

    return input_data
