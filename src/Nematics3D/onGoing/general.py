import numpy as np
import time
from typing import Union, Sequence, Iterable, Tuple, Hashable, Mapping


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
        - A precomputed dictionary mapping tuple(item) → float or int (if `is_reservoir_hash` is True)

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
        * index 0–255: blue → cyan → green (increasing G, decreasing B)
        * index 255–510: green → yellow → red (decreasing G, increasing R)
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


# def find_neighbor_coord(x, reservoir, dist_large, dist_small=0, strict=(0, 0)):
#     from scipy.spatial.distance import cdist

#     if np.array(x).ndim == 1:
#         x = [x]

#     epsilon = np.nextafter(0, 1)
#     dist = cdist(x, reservoir)

#     condition_small = dist >= dist_small + strict[0] * epsilon
#     condition_large = dist <= dist_large - strict[0] * epsilon

#     return np.where(condition_large * condition_small)


# def get_square_each(size, num, dim=2):
#     """
#     Generate the coordinates of a square's boundary

#     This function constructs a square boundary based on the given size and the number
#     of discrete points along each edge. The output contains the coordinates of these
#     points in 2D or 3D space, depending on the specified dimension.

#     The boundary always starts with [0,0,0] as the bottom-left corner and goes clockwisely.

#     If in 3D, the x-coordinates of the boundary is 0

#     Parameters
#     ----------
#     size : float
#            The length of one side of the square.

#     num : int
#           The number of points along each edge of the square. Must be greater than or equal to 2.

#     dim : int, optional
#           The dimension of the space in which the square is represented.
#           - If `dim=2` (default), the square is generated in 2D space.
#           - If `dim=3`, the square is generated in 3D space, with the x-coordinate set to 0.

#     Returns
#     -------
#     result : numpy.ndarray, (4*num-4, dim)
#              Array containing the coordinates of the points forming the boundary of the square.
#              The points are ordered in a clockwise manner starting from origin.

#     Dependencies
#     ------------
#     - NumPy: 1.26.4
#     """

#     edge1 = [np.linspace(0, size, num), np.zeros(num)]
#     edge2 = [np.zeros(num) + size, np.linspace(0, size, num)]
#     edge3 = [np.linspace(size, 0, num), np.zeros(num) + size]
#     edge4 = [np.zeros(num), np.linspace(size, 0, num)]

#     line1 = np.concatenate([edge1[0][:-1], edge2[0][:-1], edge3[0][:-1], edge4[0][:-1]])
#     line2 = np.concatenate([edge1[1][:-1], edge2[1][:-1], edge3[1][:-1], edge4[1][:-1]])
#     result = np.vstack([line1, line2]).T

#     if dim == 3:
#         result = np.hstack([np.array([np.zeros(len(line1))]).T, result])

#     return result


# def get_square(size_list, num_list, origin_list=[[0, 0, 0]], dim=3):
#     """
#     Generate the coordinates of multiple squares' boundaries in a specified dimension.

#     This function constructs boundaries for multiple squares based on given sizes,
#     numbers of points along edges, and positions of the bottom-left corner.
#     The resulting coordinates are combined into a single array.

#     Parameters
#     ----------
#     size_list : list or numpy.ndarray
#                 List or array of side lengths for the squares.
#                 Each element specifies the side length of one square.

#     num_list : list or numpy.ndarray
#                List or array of the number of points along each edge of the squares.
#                Each element corresponds to the respective square's `size_list`.

#     origin_list : list or numpy.ndarray, (N, 3), optional
#                   List or array specifying the origin for each square, as the positions of bottom-left corner.
#                   N is the number of origins
#                   Default is [[0, 0, 0]].

#     dim : int, optional
#           The dimension of the space in which the squares are represented.
#           - If `dim=2` , the squares are generated in 2D space.
#           - If `dim=3` (default), the squares are generated in 3D space, with the x-coordinates set to 0.

#     Returns
#     -------
#     result : numpy.ndarray, (total_points, dim)
#              Array containing the coordinates of the points forming the boundaries of all the squares.
#              Points from each square are ordered as returned by get_square_each().

#     Raises
#     ------
#     NameError
#         If the lengths of `size_list`, `num_list`, and `origin_list` do not match.

#     Dependencies
#     ------------
#     - NumPy: 1.26.4
#     """

#     if isinstance(size_list, int):
#         size_list = np.array([size_list])
#     if isinstance(num_list, int):
#         num_list = np.array([num_list])

#     if not len(size_list) == len(num_list) == np.shape(origin_list)[0]:
#         raise NameError("length of size_list and num_list must be the same")

#     result = np.empty((0, 3))

#     for i in range(len(size_list)):
#         temp = get_square_each(size_list[i], num_list[i], dim)
#         temp = temp + np.broadcast_to(origin_list[i], np.shape(temp))
#         result = np.vstack((result, temp))

#     return result


# def get_plane(points):
#     #! how good are points lying in a plane
#     #! average rotation vector
#     """
#     Calculate the normal vector of the best-fit plane to a set of 3D points
#     using Singular Value Decomposition (SVD).

#     Parameters
#     ----------
#     points : numpy.ndarray, (..., N, 3)
#              Array containing the 3D coordinates of the points.
#              The last dimension represents the coordinates (x, y, z).
#              It will find the averaged normal vector for each group of N points.

#     Returns
#     -------
#     normal_vector : numpy.ndarray, (..., 3)
#                     Array representing the normal vector of the best-fit plane.

#     Dependencies
#     ------------
#     - numpy: 1.22.0
#     """
#     ndim = points.ndim
#     if ndim == 2:
#         points = np.array([points])

#     # Calculate the center of the points
#     center = points.mean(axis=-2)

#     # Translate the points to be relative to the center
#     N = np.shape(points)[-2]
#     relative = points - np.tile(
#         center[:, np.newaxis, :], (*(np.ones(points.ndim - 2).astype(int)), N, 1)
#     )

#     # Perform Singular Value Decomposition (SVD) on the transposed relative points
#     svd = np.linalg.svd(np.swapaxes(relative, -1, -2), full_matrices=False)[0]

#     # Extract the left singular vector corresponding to the smallest singular value
#     normal_vector = svd[:, :, -1]

#     if ndim == 2:
#         normal_vector = normal_vector[0]

#     return normal_vector


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
