"""Nearest-point geometry queries."""

import numpy as np


__all__ = ["closest_point_on_polyline", "find_nearest_point"]


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


def closest_point_on_polyline(query_pt, poly_pts):
    """Find the point on a polyline nearest to ``query_pt``.

    The polyline is the union of the line segments joining consecutive rows of
    ``poly_pts``. Euclidean distance is used in an arbitrary number of spatial
    dimensions. Repeated consecutive points are allowed and represent
    zero-length segments.

    Parameters
    ----------
    query_pt : array-like
        Query point with shape ``(n_dimensions,)``.
    poly_pts : array-like
        Polyline vertices with shape ``(n_points, n_dimensions)`` and at least
        one point.

    Returns
    -------
    numpy.ndarray
        A floating-point copy of the nearest point on the polyline.
    """
    query = np.asarray(query_pt, dtype=float)
    points = np.asarray(poly_pts, dtype=float)

    if query.ndim != 1:
        raise ValueError(
            f"`query_pt` must be one-dimensional. Got shape {query.shape}."
        )
    if points.ndim != 2:
        raise ValueError(
            f"`poly_pts` must be two-dimensional. Got shape {points.shape}."
        )
    if points.shape[0] == 0:
        raise ValueError("`poly_pts` must contain at least one point.")
    if points.shape[1] != query.shape[0]:
        raise ValueError(
            "`query_pt` and `poly_pts` must have the same coordinate dimension. "
            f"Got {query.shape[0]} and {points.shape[1]}."
        )
    if not np.all(np.isfinite(query)):
        raise ValueError("`query_pt` must contain only finite values.")
    if not np.all(np.isfinite(points)):
        raise ValueError("`poly_pts` must contain only finite values.")

    if points.shape[0] == 1:
        return points[0].copy()

    segment_start = points[:-1]
    segment_delta = points[1:] - segment_start
    query_delta = query - segment_start

    segment_length_squared = np.einsum(
        "ij,ij->i", segment_delta, segment_delta
    )
    projection_numerator = np.einsum(
        "ij,ij->i", query_delta, segment_delta
    )

    projection_fraction = np.zeros_like(segment_length_squared)
    nonzero_segment = segment_length_squared > 0.0
    projection_fraction[nonzero_segment] = (
        projection_numerator[nonzero_segment]
        / segment_length_squared[nonzero_segment]
    )
    np.clip(projection_fraction, 0.0, 1.0, out=projection_fraction)

    projected_points = (
        segment_start + segment_delta * projection_fraction[:, None]
    )
    projection_delta = projected_points - query
    distance_squared = np.einsum(
        "ij,ij->i", projection_delta, projection_delta
    )

    return projected_points[int(np.argmin(distance_squared))].copy()
