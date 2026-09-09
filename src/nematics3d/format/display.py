"""Human-readable formatting helpers."""

from numbers import Integral, Real

import numpy as np


def fmt_value(v, ndigits=2, is_1d_single_line=False):
    """Format a real scalar or numeric ndarray with fixed decimal places."""
    if isinstance(ndigits, (bool, np.bool_)) or not isinstance(ndigits, Integral):
        raise TypeError("`ndigits` must be a non-negative integer.")
    ndigits = int(ndigits)
    if ndigits < 0:
        raise ValueError("`ndigits` must be non-negative.")
    if not isinstance(is_1d_single_line, (bool, np.bool_)):
        raise TypeError("`is_1d_single_line` must be a boolean.")

    def _format_number(value):
        return f"{float(value):.{ndigits}f}"

    if isinstance(v, np.ndarray):
        if not (
            np.issubdtype(v.dtype, np.integer) or np.issubdtype(v.dtype, np.floating)
        ):
            raise TypeError("`v` must be a real scalar or a real numeric ndarray.")

        if v.ndim == 0:
            return _format_number(v.item())
        if v.ndim == 1 and is_1d_single_line:
            return "[" + ", ".join(_format_number(value) for value in v) + "]"

        return np.array2string(
            v.astype(float, copy=False),
            separator=", ",
            formatter={"float_kind": _format_number},
        )

    if isinstance(v, (bool, np.bool_)) or not isinstance(v, Real):
        raise TypeError("`v` must be a real scalar or a real numeric ndarray.")
    return _format_number(v)


def repr_format(v, *, precision: int = 4, max_array_size: int = 12):
    """Format a value for compact repository-style repr output."""
    if isinstance(v, np.generic):
        v = v.item()

    if isinstance(v, float):
        return f"{v:.{precision}g}"

    if isinstance(v, np.ndarray):
        if v.size > max_array_size:
            return f"<ndarray shape={v.shape}, too many elements to display>"
        return np.array2string(v, precision=precision, separator=", ")

    return repr(v)


def repr_field_line(
    key: str,
    value,
    width: int,
    *,
    indent: str = "  ",
    precision: int = 4,
    max_array_size: int = 12,
    trailing_comma: bool = True,
):
    """Format one aligned ``key = value`` repr line."""
    prefix = f"{indent}{key:<{width}} = "
    value_text = repr_format(
        value,
        precision=precision,
        max_array_size=max_array_size,
    ).replace("\n", "\n" + " " * len(prefix))
    suffix = "," if trailing_comma else ""
    return f"{prefix}{value_text}{suffix}"


__all__ = ["fmt_value", "repr_field_line", "repr_format"]
