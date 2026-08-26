"""Runtime validators for real-valued lattice fields and lattice masks."""

import numpy as np

from .bool import as_bool
from .number import as_number, as_value_range


GeneralField = np.ndarray


def _as_shape(shape, *, name: str) -> tuple[int, ...]:
    """Validate one exact array shape without silently truncating dimensions."""
    try:
        shape = tuple(shape)
    except TypeError as exc:
        raise TypeError(f"{name!r} must be an iterable of positive integers.") from exc

    validated = []
    for i, dim in enumerate(shape):
        value = as_number(dim, name=f"{name}[{i}]", is_integer=True)
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name!r} dimensions must be positive. Got {shape}.")
        validated.append(value)
    return tuple(validated)


def as_real_lattice_field(
    input_data,
    name: str = "field values",
    *,
    extra_ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
    is_finite: bool = True,
    value_range=None,
    bounded: bool = False,
) -> GeneralField:
    """Convert input into a real-valued NumPy lattice field.

    The first three axes are interpreted as lattice axes. ``extra_ndim`` may
    require an exact number of trailing component axes. Numeric real-valued
    input is returned as floating point, without an unnecessary copy when the
    input already has a compatible floating dtype. ``shape`` may require one
    exact positive shape. ``value_range`` either rejects out-of-range values or
    clips them when ``bounded=True``.

    When ``is_finite=False``, NaN and infinity are permitted. A simultaneous
    ``value_range`` constraint applies only to finite values; NaN is therefore
    preserved rather than treated as an out-of-range value.
    """
    is_finite = as_bool(is_finite, name=f"{name} is_finite")
    bounded = as_bool(bounded, name=f"{name} bounded")

    values = np.asarray(input_data)
    if values.ndim < 3:
        raise ValueError(
            f"{name!r} must have at least 3 lattice axes. "
            f"Got shape {values.shape} instead."
        )

    if extra_ndim is not None:
        extra_ndim = int(
            as_number(extra_ndim, name=f"{name} extra_ndim", is_integer=True)
        )
        if extra_ndim < 0:
            raise ValueError(
                f"{name!r} extra_ndim must be non-negative. Got {extra_ndim}."
            )
        expected_ndim = 3 + extra_ndim
        if values.ndim != expected_ndim:
            raise ValueError(
                f"{name!r} must have shape (Nx, Ny, Nz)"
                f"{', ...' if extra_ndim > 0 else ''} with exactly {extra_ndim} "
                f"extra dimension{'s' if extra_ndim != 1 else ''}. "
                f"Got shape {values.shape} instead."
            )

    if shape is not None:
        shape = _as_shape(shape, name=f"{name} shape")
        if values.shape != shape:
            raise ValueError(
                f"{name!r} must have shape {shape}. Got shape {values.shape} instead."
            )

    if any(dim <= 0 for dim in values.shape):
        raise ValueError(
            f"{name!r} must not contain empty axes. Got shape {values.shape} instead."
        )

    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"{name!r} must contain numeric values. Got {values.dtype}.")
    if np.iscomplexobj(values):
        raise TypeError(
            f"{name!r} must be real-valued; complex fields are unsupported."
        )

    values = values.astype(float, copy=False)

    if is_finite and not np.all(np.isfinite(values)):
        raise ValueError(f"{name!r} must be finite everywhere.")

    if value_range is not None:
        lo, hi = as_value_range(value_range, name=f"{name} value_range")
        below = values < lo
        above = values > hi
        if np.any(below) or np.any(above):
            if not bounded:
                finite_values = values[np.isfinite(values)]
                value_min = float(np.min(finite_values))
                value_max = float(np.max(finite_values))
                raise ValueError(
                    f"{name!r} must stay within [{lo}, {hi}]. "
                    f"Got finite value range [{value_min}, {value_max}]."
                )
            values = np.clip(values, lo, hi)

    return values


MaskField = np.ndarray


def as_lattice_mask(
    input_data,
    name: str = "lattice mask",
    *,
    shape: tuple[int, ...] | None = None,
) -> MaskField:
    """Convert input into a boolean 3D lattice validity mask.

    Boolean arrays are accepted directly after shape validation. Numeric input
    must contain exactly 0/1 values. The returned array always has boolean dtype.
    """
    values = np.asarray(input_data)

    if values.dtype == np.bool_:
        if values.ndim != 3:
            raise ValueError(
                f"{name!r} must have exactly 3 lattice axes. Got shape {values.shape}."
            )
        if any(dim <= 0 for dim in values.shape):
            raise ValueError(
                f"{name!r} must not contain empty axes. Got shape {values.shape} instead."
            )
        if shape is not None:
            expected_shape = _as_shape(shape, name=f"{name} shape")
            if values.shape != expected_shape:
                raise ValueError(
                    f"{name!r} must have shape {expected_shape}. "
                    f"Got shape {values.shape} instead."
                )
        return values.copy()

    values = as_real_lattice_field(
        values,
        name=name,
        extra_ndim=0,
        shape=shape,
        is_finite=True,
        value_range=(0.0, 1.0),
    )
    if not np.all((values == 0.0) | (values == 1.0)):
        raise ValueError(
            f"{name!r} must contain only boolean-like values (True/False or 0/1)."
        )
    return values.astype(bool)
