"""Legacy compatibility helpers still used by active Nematics3D modules.

This module intentionally contains only names that remain referenced by active
source/tests. Helpers already moved to dedicated modules are re-exported here;
the remaining legacy implementations are preserved until their callers migrate.
"""

import numpy as np

from .format import fmt_value
from .geometry import (
    find_rotation_axis,
    get_box_corners,
    rotation_matrix_from_vectors,
)
from .logging_decorator import logging_and_warning_decorator

__all__ = [
    "closest_point_on_polyline",
    "find_nearest_point",
    "find_rotation_axis",
    "fmt_value",
    "get_box_corners",
    "get_square",
    "get_square_each",
    "mark_points_membership",
    "rotation_matrix_from_vectors",
    "select_grid_in_box",
]


def get_square_each(size, num, dim=2):
    """
    Generate the coordinates of a square's boundary

    This function constructs a square boundary based on the given size and the number
    of discrete points along each edge. The output contains the coordinates of these
    points in 2D or 3D space, depending on the specified dimension.

    The boundary always starts with [0,0,0] as the bottom-left corner and goes clockwisely.

    If in 3D, the x-coordinates of the boundary is 0

    Parameters
    ----------
    size : float
           The length of one side of the square.

    num : int
          The number of points along each edge of the square. Must be greater than or equal to 2.

    dim : int, optional
          The dimension of the space in which the square is represented.
          - If `dim=2` (default), the square is generated in 2D space.
          - If `dim=3`, the square is generated in 3D space, with the x-coordinate set to 0.

    Returns
    -------
    result : numpy.ndarray, (4*num-4, dim)
             Array containing the coordinates of the points forming the boundary of the square.
             The points are ordered in a clockwise manner starting from origin.

    Notes
    -----
    - The traversal starts at (0,0) and goes right, up, left, then down.
    - In 3D mode, the square lies in the YZ-plane with x=0.

    Examples
    --------
    >>> get_square_each(2, 3, dim=3)
        array([[0., 0., 0.],
               [0., 1., 0.],
               [0., 2., 0.],
               [0., 2., 1.],
               [0., 2., 2.],
               [0., 1., 2.],
               [0., 0., 2.],
               [0., 0., 1.]])
    """

    corners = np.array([[0, 0], [size, 0], [size, size], [0, size]])

    edges = []
    for i in range(4):
        p0, p1 = corners[i], corners[(i + 1) % 4]
        edge = np.linspace(p0, p1, num - 1, endpoint=False)  # ä¸è¦é‡å¤é¡¶ç‚¹
        edges.append(edge)
    coords = np.vstack(edges)

    if dim == 3:
        coords = np.hstack([np.zeros((coords.shape[0], 1)), coords])

    return coords


def get_square(size_list, num_list, origin_list=[[0, 0, 0]], dim=3):
    """
    Generate the coordinates of multiple squares' boundaries in a specified dimension.

    This function constructs boundaries for multiple squares based on given sizes,
    numbers of points along edges, and positions of the bottom-left corner.
    The resulting coordinates are combined into a single array.

    Parameters
    ----------
    size_list : list or numpy.ndarray
                List or array of side lengths for the squares.
                Each element specifies the side length of one square.

    num_list : list or numpy.ndarray
               List or array of the number of points along each edge of the squares.
               Each element corresponds to the respective square's `size_list`.

    origin_list : list or numpy.ndarray, (N, 3), optional
                  List or array specifying the origin for each square, as the positions of bottom-left corner.
                  N is the number of origins
                  Default is [[0, 0, 0]].

    dim : int, optional
          The dimension of the space in which the squares are represented.
          - If `dim=2` , the squares are generated in 2D space.
          - If `dim=3` (default), the squares are generated in 3D space, with the x-coordinates set to 0.

    Returns
    -------
    result : numpy.ndarray, (total_points, dim)
             Array containing the coordinates of the points forming the boundaries of all the squares.
             Points from each square are ordered as returned by get_square_each().

    Raises
    ------
    ValueError
        If the lengths of `size_list`, `num_list`, and `origin_list` do not match.
    """

    if isinstance(size_list, int):
        size_list = np.array([size_list])
    if isinstance(num_list, int):
        num_list = np.array([num_list])

    if not len(size_list) == len(num_list) == np.shape(origin_list)[0]:
        raise ValueError("length of size_list and num_list must be the same")

    results = []
    for size, num, origin in zip(size_list, num_list, origin_list):
        temp = get_square_each(size, num, dim) + origin
        results.append(temp)
    result = np.vstack(results)

    return result


def select_grid_in_box(
    grid: np.ndarray,
    corners_limit: np.ndarray | None,
    is_return_mask: bool = False,
    logger=None,
):
    """
    Filter a set of 3D points by an oriented rectangular box, with an optional membership mask.

    The box is specified by four corner points:
      - corners_limit[0] is the origin corner O
      - corners_limit[1], corners_limit[2], corners_limit[3] define three edge directions
        (via vectors e1 = P1-O, e2 = P2-O, e3 = P3-O)
    A point x is considered inside the box if its projections onto the unit edge directions
    lie within [0, |ei|] (up to a small tolerance).

    Parameters
    ----------
    grid : np.ndarray, shape (N, 3)
        Input 3D point cloud.

    corners_limit : np.ndarray | None, shape (>=4, 3)
        Oriented box definition. If None, no spatial filtering is applied.

    is_return_mask : bool, default False
        If True, also return a boolean mask of shape (N,) indicating membership in the box
        relative to the original input `grid`.

    Returns
    -------
    grid_selected : np.ndarray, shape (M, 3)
        Points inside the box (or the original `grid` if corners_limit is None).
        M <= N.

    mask : np.ndarray, shape (N,), dtype bool
        Returned only if is_return_mask is True.
        - If corners_limit is None: mask is all True (no filtering).
        - If grid is empty: mask is an empty boolean array.
        - Otherwise: True where the corresponding original point lies in the box.

    Notes
    -----
    - The membership check is performed in the local coordinate system spanned by the box edges.
    - A tolerance of 1e-9 is used to include points extremely close to box faces.
    - Warnings are emitted when:
        (a) `grid` is empty;
        (b) no points fall inside the box (when corners_limit is not None).
    """

    grid = np.asarray(grid)
    n = grid.shape[0] if grid.ndim >= 1 else 0

    # Prepare a mask in all exit paths when the caller requests it.
    if n == 0:
        logger.warning("Input `grid` is empty; returning an empty selection.")
        mask_empty = np.zeros((0,), dtype=bool)
        return (grid, mask_empty) if is_return_mask else grid

    if corners_limit is None:
        mask_all = np.ones((n,), dtype=bool)
        return (grid, mask_all) if is_return_mask else grid

    corners_limit = np.asarray(corners_limit)
    if (
        corners_limit.ndim != 2
        or corners_limit.shape[1] != 3
        or corners_limit.shape[0] < 4
    ):
        raise ValueError(
            f"`corners_limit` must have shape (>=4, 3). Got {corners_limit.shape} instead."
        )

    # Define edge directions and extents from the origin corner.
    axes = [corners_limit[i] - corners_limit[0] for i in range(1, 4)]
    lengths = [np.linalg.norm(axis) for axis in axes]

    # Guard against degenerate boxes (zero-length edges).
    if any(L <= 0.0 for L in lengths):
        raise ValueError(
            f"Degenerate `corners_limit`: box edge length(s) must be positive. Got lengths={lengths}."
        )

    unit_axes = [axis / L for axis, L in zip(axes, lengths)]

    # Project points into the box coordinate system.
    rel = grid - corners_limit[0]
    coords = np.stack([rel @ u for u in unit_axes], axis=1)

    tol = 1e-9
    bounds = np.array(lengths, dtype=coords.dtype)
    mask = np.all((coords >= -tol) & (coords <= bounds + tol), axis=1)

    grid_selected = grid[mask]
    if grid_selected.shape[0] == 0:
        logger.warning(
            "No points from `grid` fall inside the specified box defined by `corners_limit`:\n"
            f"{corners_limit}"
        )

    return (grid_selected, mask) if is_return_mask else grid_selected


def mark_points_membership(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask indicating whether each row in points1 appears in points2.

    Requirements:
    - points1 and points2 must have the same shape[1] (same number of columns).
    - Exact match (no tolerance) semantics.
    """
    a = np.ascontiguousarray(points1)
    b = np.ascontiguousarray(points2)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"points1 and points2 must be 2D arrays. Got {a.ndim=} and {b.ndim=}."
        )
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"points1 and points2 must have the same number of columns. Got {a.shape[1]=} vs {b.shape[1]=}."
        )

    row_dtype = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    a_view = a.view(row_dtype).ravel()
    b_view = b.view(row_dtype).ravel()

    return np.isin(a_view, b_view).reshape(-1)


def find_nearest_point(query_pt, coords, is_return_idx=False):
    """
    Find the nearest point in coords to query_pt (Euclidean).

    Parameters
    ----------
    query_pt : array-like, shape (d,)
        Query point in world coordinates.
    coords : array-like, shape (N, d)
        Candidate points.
    is_return_idx: bool,
        Whether to return the index of the nearest point.

    Returns
    -------
    nearest : np.ndarray, shape (d,)
        The nearest point in coords.
    idx : int
        The index of the nearest point.
        This is returned only if is_return_idx=True
    """
    q = np.asarray(query_pt, dtype=float).reshape(-1)
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != len(q):
        raise ValueError(
            f"`coords` shape is {pts.shape},"
            f"while normalized `query_pt` shape is {q.shape}"
        )

    d = pts - q
    d2 = np.einsum("ij,ij->i", d, d)
    idx = int(np.argmin(d2))
    return (pts[idx], idx) if is_return_idx else pts[idx]


def closest_point_on_polyline(query_pt: np.ndarray, poly_pts: np.ndarray) -> np.ndarray:
    """
    Compute the closest point on a polyline to a specific query point in 3D.

    The algorithm treats the polyline as a series of independent segments,
    projects the query point onto each segment, clips the projection to the
    segment boundaries, and identifies the globally closest result.

    Parameters
    ----------
    query_pt : (3,) array
        Coordinates of the query point (x, y, z).
    poly_pts : (N, 3) array
        Ordered vertices defining the polyline.

    Returns
    -------
    closest : (3,) array
        The coordinates of the point on the polyline closest to query_pt.
    """
    q = np.asarray(query_pt, dtype=float)
    pts = np.asarray(poly_pts, dtype=float)

    if pts.shape[0] == 1:
        return pts[0].copy()

    a = pts[:-1]
    b = pts[1:]
    ab = b - a
    aq = q - a

    ab2 = np.einsum("ij,ij->i", ab, ab)
    ab2 = np.where(ab2 <= 1e-30, 1e-30, ab2)

    t = np.einsum("ij,ij->i", aq, ab) / ab2
    t = np.clip(t, 0.0, 1.0)

    proj = a + ab * t[:, None]
    diff = proj - q
    d2 = np.einsum("ij,ij->i", diff, diff)

    idx = int(np.argmin(d2))
    return proj[idx]
