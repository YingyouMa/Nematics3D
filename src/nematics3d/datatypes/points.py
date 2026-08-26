"""Runtime converter for collections of Cartesian points."""

import numbers

import numpy as np

from .bool import as_bool


def as_points(
    input_data,
    d=3,
    name="input data",
    *,
    is_finite=True,
    is_empty=True,
    is_unique=False,
    min_num=None,
) -> np.ndarray:
    """Validate and normalize a collection of ``d``-dimensional points.

    A single point with shape ``(d,)`` is promoted to ``(1, d)``. Empty input
    is normalized to ``(0, d)`` when ``d`` is specified. The returned
    floating-point array is always independent of the input.

    Parameters
    ----------
    input_data : array-like
        One point or a collection of points.
    d : int or None, optional
        Required point dimension. ``None`` accepts any dimension.
    name : str, optional
        Human-readable input name used in error messages.
    is_finite : bool, optional
        Whether all coordinates must be finite.
    is_empty : bool, optional
        Whether a collection containing no points is allowed.
    is_unique : bool, optional
        Whether duplicate points are removed.
    min_num : int or None, optional
        Minimum number of points required after optional deduplication.
    """
    if d is not None:
        if isinstance(d, (bool, np.bool_)) or not isinstance(d, numbers.Integral):
            raise TypeError(f"'d' must be an integer or None. Got {d!r}.")
        d = int(d)
        if d <= 0:
            raise ValueError(f"'d' must be positive. Got {d}.")

    is_finite = as_bool(is_finite, name="is_finite")
    is_empty = as_bool(is_empty, name="is_empty")
    is_unique = as_bool(is_unique, name="is_unique")

    if min_num is not None:
        if isinstance(min_num, (bool, np.bool_)) or not isinstance(
            min_num, numbers.Integral
        ):
            raise TypeError(f"'min_num' must be an integer or None. Got {min_num!r}.")
        min_num = int(min_num)
        if min_num < 0:
            raise ValueError(f"'min_num' must be non-negative. Got {min_num}.")

    raw_points = np.asarray(input_data)
    if raw_points.size == 0 and raw_points.ndim <= 1:
        point_dimension = 0 if d is None else d
        raw_points = np.empty((0, point_dimension), dtype=float)
    elif raw_points.ndim == 1:
        raw_points = raw_points[np.newaxis, :]

    if raw_points.ndim != 2:
        raise ValueError(
            f"{name!r} must have shape (N, D). Got shape={raw_points.shape}."
        )
    if d is not None and raw_points.shape[1] != d:
        raise ValueError(
            f"{name!r} must have shape (N, {d}). Got shape={raw_points.shape}."
        )

    if raw_points.dtype.kind not in "iuf":
        if raw_points.dtype.kind != "O" or not all(
            isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))
            for value in raw_points.flat
        ):
            raise TypeError(f"{name!r} must contain only real numbers.")

    points = np.array(raw_points, dtype=float, copy=True)
    if is_finite and not np.all(np.isfinite(points)):
        raise ValueError(f"{name!r} must contain only finite values.")
    if is_unique:
        points = np.unique(points, axis=0)
    if not is_empty and len(points) == 0:
        raise ValueError(f"{name!r} must contain at least one point.")
    if min_num is not None and len(points) < min_num:
        raise ValueError(
            f"{name!r} must contain at least {min_num} point(s). Got {len(points)}."
        )
    return points
