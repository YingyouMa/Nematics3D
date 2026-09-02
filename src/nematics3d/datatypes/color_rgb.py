"""RGB color semantic alias and runtime validation helpers."""

import numpy as np

from ..logging_decorator import logging_and_warning_decorator


ColorRGB = tuple[float, float, float]


def _validate_rgb_values(values, *, name: str) -> np.ndarray:
    """Return one validated floating-point RGB array."""
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError(f"{name} must contain real numeric values.")

    values = values.astype(float, copy=True)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} must contain RGB values in [0, 1].")
    return values


def _normalize_rgb(values: np.ndarray, *, norm_order, axis=None) -> np.ndarray:
    """Normalize RGB values using the package's historical sum-of-powers rule."""
    norms = np.sum(values**norm_order, axis=axis)

    if axis is None:
        if norms < 1e-3:
            return np.zeros_like(values)
        return values / norms

    result = values.copy()
    safe = norms >= 1e-3
    result[safe] /= norms[safe, None]
    result[~safe] = 0.0
    return result


@logging_and_warning_decorator(start_finish_level=5)
def as_ColorRGB(
    input_data,
    name="input data",
    is_norm=False,
    norm_order=2,
    replace=None,
    logger=None,
):
    """Validate and normalize one RGB color, returning a 3-float tuple."""
    try:
        if not isinstance(input_data, (tuple, list, np.ndarray)):
            raise ValueError(f"{name} must be an RGB sequence with shape (3,).")

        values = np.asarray(input_data)
        if values.shape != (3,):
            raise ValueError(f"{name} must have shape (3,). Got shape {values.shape}.")
        values = _validate_rgb_values(values, name=name)

    except (TypeError, ValueError):
        if replace is None:
            raise

        logger.exception("Please check RGB data.")
        logger.recovery(f"Set color={replace} in the following.")
        values = np.asarray(replace)
        if values.shape != (3,):
            raise ValueError(f"replace must have shape (3,). Got shape {values.shape}.")
        values = _validate_rgb_values(values, name="replace")

    if is_norm:
        values = _normalize_rgb(values, norm_order=norm_order)

    return tuple(map(float, values))


@logging_and_warning_decorator(start_finish_level=5)
def as_ColorRGB_array(
    input_data,
    name="input data",
    is_norm=False,
    norm_order=2,
    replace=None,
    logger=None,
):
    """Validate and normalize an ``(N, 3)`` RGB array."""
    if not isinstance(input_data, (list, tuple, np.ndarray)):
        raise ValueError(
            f"{name} must be an array-like with shape (N, 3). "
            f"Got {type(input_data).__name__}."
        )

    values = np.asarray(input_data)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3). Got shape {values.shape}.")

    n = values.shape[0]
    try:
        values = _validate_rgb_values(values, name=name)

    except (TypeError, ValueError):
        if replace is None:
            raise

        logger.exception("Please check color array values.")
        logger.recovery(f"Set color array to {replace} in the following.")
        replace_arr = np.asarray(replace)

        if replace_arr.shape == (3,):
            replace_arr = np.tile(replace_arr, (n, 1))
        elif replace_arr.shape != (n, 3):
            raise ValueError(
                f"replace must have shape (3,) or ({n}, 3). "
                f"Got shape {replace_arr.shape}."
            )
        values = _validate_rgb_values(replace_arr, name="replace")

    if is_norm:
        values = _normalize_rgb(values, norm_order=norm_order, axis=1)

    return values
