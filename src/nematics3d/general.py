"""
Miscellaneous shared utilities. If a helper does not clearly belong to a more specific
module yet, it can live here for now.
"""

import numpy as np
from typing import Union, Sequence, Iterable, Tuple, Hashable, Mapping, Optional
from .datatypes import Tensor, Vect
from nematics3d.format import fmt_value, is_equal, is_equal_array, is_given_str
from nematics3d.logging_decorator import logging_and_warning_decorator
from .geometry import (
    find_plane_normal,
    find_rotation_axis,
    rotation_matrix_from_vectors,
)


def make_hash_table(
    input: Iterable[Iterable[Hashable]],
) -> dict[Tuple[Hashable, ...], int]:
    """
    Create a hash table that maps each item (as a tuple) to its index in the input.

    Parameters
    ----------
    input : Iterable of Iterable[Hashable]
        A sequence of items, where each item is itself an iterable of hashable elements
        (e.g., list of ints or strings). Each item is converted to a tuple for hashing.

    Returns
    -------
    hash_table : dict[Tuple[Hashable, ...], int]
        A dictionary mapping each item's tuple form to its index in the input sequence.
        If a key is not found in the table, it returns NaN (via defaultdict fallback).

    Examples
    --------
    >>> make_hash_table([[1, 2], [3, 4]])
    {(1, 2): 0, (3, 4): 1}
    """

    from collections import defaultdict

    hash_table = defaultdict(lambda: np.nan)
    for idx, item in enumerate(input):
        item_hash = tuple(item)  # Ensure it's hashable
        hash_table[item_hash] = idx

    return hash_table


def search_in_reservoir(
    items: Sequence[Sequence[float]],
    reservoir: Union[Sequence[Sequence[float]], Mapping[tuple, float]],
    is_reservoir_hash: bool = False,
) -> np.ndarray:
    """
    Search a list of items within a reservoir, and return the corresponding index or value.

    If `reservoir` is a sequence of vectors (e.g., list of coordinates), a hash table will be built
    to map each vector (converted to tuple) to its index.
    If `reservoir` is already a hash table (dictionary-like), it will be used directly.

    Parameters
    ----------
    items : Sequence of Sequence of float
        A list of query items, where each item is an iterable of numbers (e.g., a 3D coordinate).
        Each item will be converted to a tuple for hash lookup.

    reservoir : Sequence of items or dict-like
        Either:
        - A list of reservoir items (to be hashed by `make_hash_table`), or
        - A precomputed dictionary mapping tuple(item) â†’ float or int (if `is_reservoir_hash` is True)

    is_reservoir_hash : bool, optional
        If True, `reservoir` is assumed to be a dictionary-like hash table.
        If False (default), a hash table will be constructed from the reservoir sequence.

    Returns
    -------
    result : np.ndarray of shape (len(items),)
        Array of mapped values from the reservoir hash table.
        Typically, if the hash maps to indices, this gives the index of each item in the reservoir.

    Raises
    ------
    KeyError
        If any item in `items` is not found in the reservoir.
    """
    if not is_reservoir_hash:
        reservoir_hash_table = make_hash_table(reservoir)
    else:
        reservoir_hash_table = reservoir

    result = np.zeros(len(items), dtype=float)

    for idx, item in enumerate(items):
        item_key = tuple(item)
        # if item_key not in reservoir_hash_table:
        #     raise KeyError(f"Item {item_key} not found in reservoir.")
        result[idx] = reservoir_hash_table[item_key]

    return result


def nearest_neighbor_order(
    points: Union[np.ndarray, Sequence[Sequence[float]]],
) -> list[int]:
    """
    Determine a greedy nearest-neighbor visiting order for a set of points.

    Starting from the first point (index 0), this function iteratively chooses the
    closest unvisited point as the next point in the sequence, forming an approximate
    path through all points (not necessarily optimal like TSP).

    Parameters
    ----------
    points : array-like of shape (N, D)
        Coordinates of the points to be ordered.
        N is the number of points, and D is the spatial dimension (e.g., 2 or 3).

    Returns
    -------
    order : list of int
        The indices of points in the order they are visited based on greedy nearest-neighbor search.
    """
    from scipy.spatial.distance import cdist

    num_points = len(points)

    # Calculate the pairwise distance matrix
    dist = cdist(points, points)

    # Initialize variables for tracking visited points and the order
    visited = np.zeros(num_points, dtype=bool)
    visited[0] = True
    order = [0]

    # Determine nearest neighbors iteratively
    for i in range(num_points - 1):
        current_point = order[-1]
        nearest_neighbor = np.argmin(dist[current_point, :] + visited * np.max(dist))
        order.append(nearest_neighbor)
        visited[nearest_neighbor] = True

    return order


def sort_line_indices(points):
    """
    Sort the indices of defects within a line based on their nearest neighbor order.

    Parameters
    ----------
    points : array-like of shape (N, D)
        Coordinates of the points to be ordered.
        N is the number of points, and D is the spatial dimension (e.g., 2 or 3).

    Returns
    -------
    output : numpy.ndarray, (N, D)
        Array representing the sorted indices of the line based on nearest neighbor order.
        N is the number of points, and D is the dimension (usually 2 or 3).
    """

    output = points[nearest_neighbor_order(points)]
    return output


def blue_red_in_white_bg() -> np.ndarray:
    """
    Generate a normalized blue-to-red colormap suitable for white backgrounds.

    This function produces a colormap with 511 RGB entries, transitioning from blue to red
    with green as an intermediate step. The RGB values are scaled between 0 and 1 and
    normalized by their L2 norm to enhance contrast against a white background.

    Returns
    -------
    colormap : np.ndarray, shape (511, 3)
        Array representing the normalized RGB colormap.

    Notes
    -----
    - The colormap is constructed as:
        * index 0â€“255: blue â†’ cyan â†’ green (increasing G, decreasing B)
        * index 255â€“510: green â†’ yellow â†’ red (decreasing G, increasing R)
    - L2 normalization is applied to each RGB vector for perceptual contrast
      when visualized on white backgrounds.
    """

    colormap = np.zeros((511, 3))
    colormap[:256, 1] = np.arange(256)
    colormap[:256, 2] = 255 - np.arange(256)
    colormap[255:, 1] = 255 - np.arange(256)
    colormap[255:, 0] = np.arange(256)
    colormap = colormap / 255
    colormap = colormap / np.linalg.norm(colormap, axis=-1, keepdims=True)

    return colormap


def sample_far(num: int) -> np.ndarray:
    """
    Generate a sequence of `num` sample points between 0 and 1 that are maximally spaced.

    The algorithm begins with [0, 1], and each subsequent point is placed to maximize
    its distance from previous samples, using a binary partitioning rule.

    Parameters
    ----------
    num : int
        Number of sample points to generate.

    Returns
    -------
    result : numpy.ndarray
        Array of `num` float values in the range [0, 2], representing spatially spread samples.

    Use Case
    --------
    - Used to sample points that are "far apart" in value space
    - Good for progressive refinement, plotting, or hierarchical sampling
    """

    result_init = [0, 1]
    if num <= 2:
        result = np.array(result_init[:num])
        return result

    n = np.arange(2, num)
    a = 2 ** np.trunc(np.log2(n - 1) + 1)
    b = 2 * n - a - 1

    result = np.zeros(num)

    result[0] = 0
    result[1] = 1
    result[2:] = b / a

    return result


def get_box_corners(Lx: float, Ly: float, Lz: float) -> np.ndarray:
    """
    Return the 8 corner coordinates of a rectangular box
    from (0, 0, 0) to (Lx, Ly, Lz).

    Parameters
    ----------
    Lx, Ly, Lz : float
        Lengths of the box along x, y, z axes.

    Returns
    -------
    corners : list of tuple of float
        List of 8 corner coordinates in (x, y, z) form.
        Order is:
            (0, 0, 0), (Lx, 0, 0), (0, Ly, 0), (0, 0, Lz),
            (Lx, Ly, 0), (Lx, 0, Lz), (0, Ly, Lz), (Lx, Ly, Lz)
    """
    corners = np.array(
        [
            [0, 0, 0],
            [Lx, 0, 0],
            [0, Ly, 0],
            [0, 0, Lz],
            [Lx, Ly, 0],
            [Lx, 0, Lz],
            [0, Ly, Lz],
            [Lx, Ly, Lz],
        ],
        dtype=np.float64,
    )
    return corners


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


@logging_and_warning_decorator()
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


def split_points(
    points1: np.ndarray, points2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split points in `points1` into two groups based on whether each row also appears in `points2`.

    Description
    -----------
    Treat each row as one point in D-dimensional space (D can be any positive integer).
    The function returns:
      - `only_in_points1`: rows in `points1` that do NOT appear in `points2`
      - `also_in_points2`: rows in `points1` that also appear in `points2`
    Row equality uses exact value comparison on all columns.

    Parameters
    ----------
    points1 : np.ndarray, shape (N, D) or (0, D)
        The primary set of points to be split by membership.
    points2 : np.ndarray, shape (M, D) or (0, D)
        The reference set of points used for membership testing.

    Returns
    -------
    only_in_points1 : np.ndarray, shape (K, D)
        Points in `points1` but not in `points2`.
    also_in_points2 : np.ndarray, shape (L, D)
        Points in `points1` that also appear in `points2`.
    Note that K + L may be <= N if `points1` contains duplicate rows (duplicates are preserved in output as present in set ops result).

    Raises
    ------
    ValueError
        If inputs cannot be reshaped to 2D with the same number of columns, or dtypes are incompatible.

    Notes
    -----
    - Works for arbitrary D (not just 3).
    - Exact equality is used. For floating-point data, consider pre-rounding if your points come from
      numerical computations with tiny differences (e.g., `np.round(points, decimals=9)`).
    - Each C-contiguous row is temporarily reinterpreted as one ``np.void`` blob, so ``setdiff1d`` and
      ``intersect1d`` compare whole rows instead of comparing elements independently.
    - Empty arrays are fully supported as long as they have shape (0, D).

    Examples
    --------
    >>> p1 = np.array([[0, 0], [1, 1], [2, 2]], dtype=int)
    >>> p2 = np.array([[1, 1], [3, 3]], dtype=int)
    >>> only, both = split_points_by_membership(p1, p2)
    >>> only
    array([[0, 0],
           [2, 2]])
    >>> both
    array([[1, 1]])
    """
    # --- Normalize inputs to 2D arrays ---
    a = np.asarray(points1)
    b = np.asarray(points2)

    if a.ndim == 1:
        if a.size == 0:
            raise ValueError(
                "points1 is 1D empty; use shape (0, D) for empty point sets."
            )
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        if b.size == 0:
            # allow empty, but must know D from points1
            if a.ndim != 2:
                raise ValueError("Cannot infer dimensionality for empty points2.")
            b = np.empty((0, a.shape[1]), dtype=a.dtype)
        else:
            b = b.reshape(-1, 1)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Both inputs must be 2D arrays of shape (N, D) and (M, D).")

    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Dimensionality mismatch: points1 has D={a.shape[1]}, points2 has D={b.shape[1]}."
        )

    # If dtypes differ, try a common dtype cast (e.g., int vs. int64). This keeps behavior predictable.
    common_dtype = np.result_type(a, b)
    a = a.astype(common_dtype, copy=False)
    b = b.astype(common_dtype, copy=False)

    # Ensure C-contiguous before viewing rows as bytes
    a_c = np.ascontiguousarray(a)
    b_c = np.ascontiguousarray(b)

    # --- Row-wise view via np.void (each row becomes one "scalar" for set ops) ---
    rowsize = a_c.dtype.itemsize * a_c.shape[1]
    a_view = a_c.view((np.void, rowsize)).reshape(-1)
    b_view = b_c.view((np.void, rowsize)).reshape(-1)

    # --- Set operations on rows ---
    # rows in a but not in b
    only_view = np.setdiff1d(a_view, b_view, assume_unique=False)
    # rows common to both a and b (membership of a against b)
    both_view = np.intersect1d(a_view, b_view, assume_unique=False)

    # Map back to 2D arrays
    only_in_points1 = only_view.view(common_dtype).reshape(-1, a_c.shape[1])
    also_in_points2 = both_view.view(common_dtype).reshape(-1, a_c.shape[1])

    return only_in_points1, also_in_points2


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


def pop_exclusive(kwargs: dict, k1: str, k2: str):  # 1
    """
    Pop exactly one of (k1, k2) from kwargs if present.

    Returns
    -------
    found : bool
        True if exactly one of the keys existed.
    value : Any
        The popped value if found, otherwise None.

    Raises
    ------
    TypeError
        If both keys exist simultaneously.
    """
    has1 = k1 in kwargs
    has2 = k2 in kwargs
    if has1 and has2:
        raise TypeError(
            f"Ambiguous input: both {k1!r} and {k2!r} were provided. "
            f"Please pass only one."
        )
    if has1:
        return True, kwargs.pop(k1)
    if has2:
        return True, kwargs.pop(k2)
    return False, None


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


# def find_neighbor_coord(x, reservoir, dist_large, dist_small=0, strict=(0, 0)):
#     from scipy.spatial.distance import cdist

#     if np.array(x).ndim == 1:
#         x = [x]

#     epsilon = np.nextafter(0, 1)
#     dist = cdist(x, reservoir)

#     condition_small = dist >= dist_small + strict[0] * epsilon
#     condition_large = dist <= dist_large - strict[0] * epsilon

#     return np.where(condition_large * condition_small)


# def get_rotation_axis(vectors):

#     cross_bulk = np.cross(vectors[:, :-1], vectors[:, 1:], axis=-1)
#     cross_end = np.cross(vectors[:, -1], vectors[:, 0], axis=-1)
#     cross_all = np.concatenate([cross_bulk, cross_end[:, np.newaxis]], axis=1)
#     cross_mean = np.mean(cross_all, axis=1)

#     cross_mean = cross_mean / np.linalg.norm(cross_mean, axis=1, keepdims=True)

#     return cross_mean


# def get_tangent(points, is_periodic=False, is_norm=True):
#     """
#     Calculate the tangent vectors at each point of a given set of points.

#     This function computes the tangent vectors for a series of points in space.
#     It supports periodic boundary conditions and optionally normalizes the tangent vectors.

#     Parameters
#     ----------
#     points : numpy.ndarray, (N, D)
#              Array of points where tangents are calculated.
#              `N` is the number of points,
#              `D` is the dimension of each point.

#     is_periodic : bool, optional
#                   Indicates whether the points form a periodic structure.
#                   - If `True`, the tangent at the first and last points is calculated
#                     using periodic boundary conditions.
#                   - If `False` (default), the tangent at the first and last points is calculated
#                     using forward and backward differences, respectively.

#     is_norm : bool, optional
#               Indicates whether to normalize the tangent vectors.
#               - If `True` (default), each tangent vector is normalized to unit length.

#     Returns
#     -------
#     tangents : numpy.ndarray, (N, D)
#                Array of tangent vectors corresponding to the input points. The shape matches
#                the input, with each row representing the tangent vector at the corresponding point.

#     Dependencies
#     ------------
#     - NumPy: 1.26.4
#     """

#     if is_periodic:
#         tangents = (np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)) / 2
#     else:
#         tangents = np.zeros_like(points)
#         tangents[1:-1] = (points[2:] - points[:-2]) / 2
#         tangents[0] = points[1] - points[0]
#         tangents[-1] = points[-1] - points[-2]

#     if is_norm:
#         size = np.linalg.norm(tangents, axis=1, keepdims=True)
#         size += 1e-6
#         tangents = tangents / size

#     return tangents


# def get_curvature(points, is_periodic=False):
#     """
#     Calculate the curvature at each point of a given set of points.

#     This function computes the curvature of a curve defined by a series of points in space.
#     It uses the tangent vectors to estimate the rate of change of direction along the curve.
#     Periodic boundary conditions are supported.

#     Parameters
#     ----------
#     points : numpy.ndarray, (N, D)
#              Array of points where curvature is calculated.
#              `N` is the number of points,
#              `D` is the dimension of each point.

#     is_periodic : bool, optional
#                   Indicates whether the points form a periodic structure.
#                   - If `True`, curvature at the first and last points is calculated
#                     using periodic boundary conditions.
#                   - If `False` (default), the curvature at the first and last points is approximated
#                     using forward and backward differences, respectively.

#     Returns
#     -------
#     curvatures : numpy.ndarray, (N,)
#                  Array of curvature values corresponding to the input points.
#                  Each value represents the magnitude of the rate of change of the tangent vector.

#     Dependencies
#     ------------
#     - NumPy: 1.26.4
#     """

#     tangents = get_tangent(points, is_periodic=is_periodic, is_norm=False)
#     tangents_size = np.linalg.norm(tangents, axis=1, keepdims=True)
#     tangents = tangents / tangents_size

#     dT_ds = get_tangent(tangents, is_periodic=is_periodic, is_norm=False)
#     dT_ds_size = np.linalg.norm(dT_ds, axis=1, keepdims=False)
#     curvatures = dT_ds_size / tangents_size[:, 0]

#     return curvatures

# @logging_and_warning_decorator()
# def calc_colors(colors, num_points, data: Optional[np.ndarray] = None, logger=None):
#     if colors is None:
#         colors = np.ones((num_points, 3))
#     elif callable(colors):
#         colors = colors(data)
#     elif isinstance(colors, (list, tuple, np.ndarray)):
#         colors = np.asarray(colors)
#         if np.size(colors) == 3:
#             colors = as_ColorRGB(colors)
#             colors = [colors for i in range(num_points)]
#         else:
#             if np.shape(colors) != (num_points, 3):
#                 msg = f"The array-like input of colors must be in shape {(num_points, 3)}. Got {np.shape(colors)} instead.\n"
#                 msg += "In the following, set directors to be white."
#                 logger.warning(msg)
#                 colors = [(1, 1, 1) for i in range(num_points)]
#             else:
#                 colors = [(color) for color in colors]
#     return colors


# @logging_and_warning_decorator()
# def calc_opacity(opacity, num_points, data: Optional[np.ndarray] = None, logger=None):
#     if opacity is None:
#         opacity = np.ones(num_points)
#     elif isinstance(opacity, (int, float)):
#         if opacity > 1 or opacity < 0:
#             msg = f"opacity must be in [0,1]. Got {opacity} insetad.\n"
#             msg += "In the following, set opacity of directors as 1."
#             logger.warning(msg)
#             opacity = np.ones(num_points)
#         else:
#             opacity = np.zeros(num_points) + opacity
#     elif callable(opacity):
#         opacity = opacity(data)
#     else:
#         opacity = np.asarray(opacity)
#         if np.max(opacity) > 1 or np.min(opacity) < 0:
#             msg = f"opacity must be in [0,1]. Got ({np.min(opacity), np.max(opacity)}) insetad.\n"
#             msg += "In the following, set opacity of directors as 1."
#             logger.warning(msg)
#             opacity = np.ones(num_points)
#         elif len(opacity) != num_points:
#             msg = f"The array-like input of opacity must be in length {num_points}. Got {len(opacity)} instead.\n"
#             msg += "In the following, set opacity of directors as 1."
#             logger.warning(msg)
#             opacity = np.ones(num_points)
#     return opacity
