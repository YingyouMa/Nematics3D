"""Read-only NumPy array conversion helper."""

import numpy as np

from .bool import as_bool


def as_readonly_array(input_data, *, dtype=float, copy: bool = True) -> np.ndarray:
    """Return a read-only NumPy array with explicit copy semantics.

    Parameters
    ----------
    input_data
        Array-like input.
    dtype
        NumPy dtype passed to ``np.asarray``. The default remains ``float`` for
        compatibility with the numerical-state use cases in Nematics3D.
    copy
        If ``True`` (default), return an independent array. If ``False``, avoid
        copying underlying data when possible, but still return a separate
        read-only view so the caller's original ndarray is not made read-only.

    Returns
    -------
    np.ndarray
        A NumPy array with ``flags.writeable == False``.
    """
    copy = as_bool(copy, name="copy")
    values = np.asarray(input_data, dtype=dtype)
    values = values.copy() if copy else values.view()
    values.setflags(write=False)
    return values
