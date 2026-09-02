"""General-purpose formatting helpers without a more specific module home."""

from numbers import Integral, Real

import numpy as np


__all__ = ["fmt_value"]


def fmt_value(v, ndigits=2, is_1d_single_line=False):
    """Format a real scalar or numeric ndarray with fixed decimal places.

    Arrays normally use NumPy's structured display and summarization behavior.
    Set ``is_1d_single_line=True`` to force a one-dimensional array to be
    rendered completely on one line without NumPy line wrapping or omission.
    """
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
            np.issubdtype(v.dtype, np.integer)
            or np.issubdtype(v.dtype, np.floating)
        ):
            raise TypeError("`v` must be a real scalar or a real numeric ndarray.")

        if v.ndim == 0:
            return _format_number(v.item())
        if v.ndim == 1 and is_1d_single_line:
            return "[" + ", ".join(_format_number(value) for value in v) + "]"

        values = v.astype(float, copy=False)
        return np.array2string(
            values,
            separator=", ",
            formatter={"float_kind": _format_number},
        )

    if isinstance(v, (bool, np.bool_)) or not isinstance(v, Real):
        raise TypeError("`v` must be a real scalar or a real numeric ndarray.")
    return _format_number(v)
