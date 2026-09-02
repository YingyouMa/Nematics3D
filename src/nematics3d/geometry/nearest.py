"""Nearest-point geometry queries."""

import numpy as np


__all__ = ["find_nearest_point"]


def find_nearest_point(query_pt, coords, is_return_idx=False):
    """Find the point in ``coords`` nearest to ``query_pt``.

    Euclidean distance is used in an arbitrary number of spatial dimensions.
    ``query_pt`` must be one-dimensional and ``coords`` must have shape
    ``(n_points, n_dimensions)`` with at least one point. All coordinates must
    be finite real numbers. If several points are equally near, the first one
    in ``coords`` is returned.

    Parameters
    ----------
    query_pt : array-like
        Query point with shape ``(n_dimensions,)``.
    coords : array-like
        Candidate points with shape ``(n_points, n_dimensions)``.
    is_return_idx : bool, default=False
        If True, also return the index of the nearest point.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, int]
        A copy of the nearest point, represented as floating-point values,
        optionally together with its index in ``coords``.
    """
    if not isinstance(is_return_idx, (bool, np.bool_)):
        raise TypeError("`is_return_idx` must be a boolean.")

    query = np.asarray(query_pt, dtype=float)
    points = np.asarray(coords, dtype=float)

    if query.ndim != 1:
        raise ValueError(
            f"`query_pt` must be one-dimensional. Got shape {query.shape}."
        )
    if points.ndim != 2:
        raise ValueError(f"`coords` must be two-dimensional. Got shape {points.shape}.")
    if points.shape[0] == 0:
        raise ValueError("`coords` must contain at least one point.")
    if points.shape[1] != query.shape[0]:
        raise ValueError(
            "`query_pt` and `coords` must have the same coordinate dimension. "
            f"Got {query.shape[0]} and {points.shape[1]}."
        )
    if not np.all(np.isfinite(query)):
        raise ValueError("`query_pt` must contain only finite values.")
    if not np.all(np.isfinite(points)):
        raise ValueError("`coords` must contain only finite values.")

    delta = points - query
    distance_squared = np.einsum("ij,ij->i", delta, delta)
    index = int(np.argmin(distance_squared))
    point = points[index].copy()

    return (point, index) if is_return_idx else point
