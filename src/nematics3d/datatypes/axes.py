"""Runtime validation helper for 3D orthonormal axes frames."""

import numbers

import numpy as np


def as_axes(
    input_data,
    name: str = "axes",
    *,
    atol: float = 1e-8,
    is_right_handed: bool = True,
) -> np.ndarray:
    """Validate a 3D orthonormal frame stored as column vectors.

    Parameters
    ----------
    input_data : array-like
        Candidate frame with shape ``(3, 3)``. Each column represents one axis.
    name : str, optional
        Human-readable name used in error messages.
    atol : float, optional
        Non-negative finite absolute tolerance used for the orthonormality check.
    is_right_handed : bool, optional
        If ``True``, a valid left-handed frame is converted to a right-handed
        frame by flipping its final axis.

    Returns
    -------
    numpy.ndarray
        A floating-point ``(3, 3)`` orthonormal frame. The returned array is an
        independent copy and is right-handed when ``is_right_handed=True``.
    """
    if not isinstance(atol, numbers.Real) or isinstance(atol, (bool, np.bool_)):
        raise TypeError(f"{name!r} atol must be a real number. Got {atol!r}.")
    atol = float(atol)
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError(f"{name!r} atol must be finite and non-negative. Got {atol!r}.")

    if not isinstance(is_right_handed, (bool, np.bool_)):
        raise TypeError(
            f"{name!r} is_right_handed must be a boolean. "
            f"Got {is_right_handed!r}."
        )

    values = np.asarray(input_data)
    if values.shape != (3, 3):
        raise ValueError(
            f"{name!r} must have shape (3, 3). Got shape {values.shape}."
        )
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise TypeError(f"{name!r} must contain real numeric values.")

    axes = values.astype(float, copy=True)
    if not np.all(np.isfinite(axes)):
        raise ValueError(f"{name!r} must contain only finite values.")

    gram = axes.T @ axes
    if not np.allclose(gram, np.eye(3), atol=atol, rtol=0.0):
        raise ValueError(
            f"{name!r} must be an orthonormal 3D frame stored as columns."
        )

    if is_right_handed and np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0

    return axes
