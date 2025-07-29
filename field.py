# ------------------------------------
# Analysis of Q field in 3D
# Yingyou Ma, Physics @ Brandeis, 2023
# ------------------------------------

from itertools import product
import time
from typing import Tuple, Optional

import numpy as np

from .general import *
from .datatypes import QField, QField9, as_QField9, nField, SField, GeneralField,  DimensionFlagInput, as_dimension_info
from .logging_decorator import logging_and_warning_decorator


@logging_and_warning_decorator()
def diagonalizeQ(qtensor: QField, logger=None) -> Tuple[SField, nField]:
    #! biaxial
    """
    Diagonalize a Q-tensor field to compute scalar order parameter (S) and director field (n).

    Accepts Q in either:
    - 5-component representation with shape (N, M, L, 5), or
    - full 3×3 matrix representation with shape (N, M, L, 3, 3)

    This function computes:
    - the largest eigenvalue of Q (λ_max), and
    - its corresponding eigenvector (director n).

    The scalar order parameter S is defined as 1.5 × λ_max.

    This function would be extended to cases of biaxial and/or negative S in the future.

    Parameters
    ----------
    qtensor : QField
        The Q-tensor field to be diagonalized.

    Returns
    -------
    S : SField
        Scalar order parameter, shape (N, M, L,)
        Defined as 1.5 × the largest eigenvalue of the Q tensor.

    n : nField
        Director field (unit vector), shape (N, M, L, 3)

    Raises
    ------
    TypeError
        If the input is not a float-type NumPy array.
    ValueError
        If the input shape is not a valid QField5 or QField9 structure.
    """
    Q: QField9 = as_QField9(qtensor)

    # Compute tensor invariants
    logger.info("Computing tensor invariants (p, q, r).")
    start = time.time()
    p = 0.5 * np.einsum("...ab, ...ba -> ...", Q, Q)
    q = np.linalg.det(Q)
    r = 2 * np.sqrt(p / 3)
    logger.info(f"Tensor invariants computed in {time.time() - start:.2f} seconds.")

    # Largest eigenvalue λ (before scaling)
    logger.info("Computing largest eigenvalue λ_max.")
    start = time.time()
    cos_arg = 4 * q / r**3
    cos_arg = np.clip(cos_arg, -1.0, 1.0)  # ensure valid domain
    lambda_max = r * np.cos((1 / 3) * np.arccos(cos_arg))
    logger.info(f"λ_max computed in {time.time() - start:.2f} seconds.")

    # Director corresponding to λ_max
    logger.info("Computing director field n.")
    start = time.time()
    n_raw = np.array(
        [
            Q[..., 0, 2] * (Q[..., 1, 1] - lambda_max) - Q[..., 0, 1] * Q[..., 1, 2],
            Q[..., 1, 2] * (Q[..., 0, 0] - lambda_max) - Q[..., 0, 1] * Q[..., 0, 2],
            Q[..., 0, 1] ** 2
            - (Q[..., 0, 0] - lambda_max) * (Q[..., 1, 1] - lambda_max),
        ]
    )
    n_unit = n_raw / np.linalg.norm(n_raw, axis=0)
    n: nField = np.moveaxis(n_unit, 0, -1)  # (..., 3)
    logger.info(f"Director field computed in {time.time() - start:.2f} seconds.")

    # Scale eigenvalue to get S
    S: SField = 1.5 * lambda_max

    return S, n


def getQ(n: nField, S: Optional[SField] = None) -> QField9:
    #! biaxial
    """
    Compute the Q-tensor field from a given director field and optional scalar order parameter.

    This function constructs a symmetric, traceless, uniaxial Q-tensor of the form:
        Q_ij = S * (n_i n_j - δ_ij / 3)

    If `S` is not provided, the tensor is computed assuming S = 1.

    Parameters
    ----------
    n : nField
        Director field of shape (..., 3).

    S : SField, optional
        Scalar order parameter field of shape (...,). If provided, scales the Q-tensor accordingly.

    Returns
    -------
    Q : QField9
        The computed Q-tensor field of shape (..., 3, 3), symmetric and traceless.
    """
    Q = np.einsum("...i, ...j -> ...ij", n, n)
    Q = Q - np.diag((1, 1, 1)) / 3
    if S is not None:
        Q = np.einsum("..., ...ij -> ...ij", S, Q)

    return Q



def add_periodic_boundary(data: GeneralField, 
                          is_boundary_periodic: DimensionFlagInput = 0) -> GeneralField:
    """
    Extend a physical field with periodic boundary slices in specified dimensions.

    This function appends one extra grid slice along each of the periodic dimensions.
    The added slice is a copy of the first slice along that axis, ensuring periodic continuity.
    If a dimension is non-periodic, it is left unchanged.

    Parameters
    ----------
    data : GeneralField
        Input physical field of shape (N, M, L, ...), where (N, M, L) are spatial dimensions,
        and the remaining axes represent vector/tensor components or other per-voxel data.

    is_boundary_periodic : DimensionFlagInput, optional
        A 3-element flag indicating which spatial dimensions are periodic.
        - Can be a scalar (broadcasted), or
        - A list/tuple/array of booleans with shape (3,)
        - Default is 0 (all dimensions non-periodic)

    Returns
    -------
    output : GeneralField
        Extended field with one additional slice along each periodic dimension.
        Shape becomes:
            (N + is_periodic[0], M + is_periodic[1], L + is_periodic[2], ...)
    """
    is_boundary_periodic = as_dimension_info(is_boundary_periodic)

    if np.sum(is_boundary_periodic) != 0:
        N, M, L = np.shape(data)[:3]  # Extract the first three dimensions
        output = np.zeros(
            (
                N + is_boundary_periodic[0],
                M + is_boundary_periodic[1],
                L + is_boundary_periodic[2],
                *(np.shape(data)[3:]),
            )
        )  # Preserve additional dimensions
        output[:N, :M, :L] = data  # Copy original data into the new array

        # Copy first slices to last.
        if is_boundary_periodic[0]:
            output[N] = output[0]
        if is_boundary_periodic[1]:
            output[:, M] = output[:, 0]
        if is_boundary_periodic[2]:
            output[:, :, L] = output[:, :, 0]
    else:
        output = data

    return output


def generate_coordinate_grid(source_shape: Tuple[int, ...],
                             target_shape: Tuple[int, ...],
                             transform: Optional[np.ndarray] = None,
                             offset: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Generate a dense N-dimensional coordinate grid for interpolation or resampling,
    with optional linear transformation (rotation/scaling) and translation (offset).

    Parameters
    ----------
    source_shape : tuple of int
        Shape of the original data in N dimensions.

    target_shape : tuple of int
        Desired shape of the resampled grid in N dimensions.

    transform : np.ndarray, optional
        Linear transformation matrix of shape (N, N).
        If None, defaults to identity (no transform).

    offset : np.ndarray, optional
        Offset vector (translation) of shape (N,).
        If None, defaults to zero vector (no translation).

    Returns
    -------
    grid : np.ndarray
        Coordinate grid of shape (*target_shape, N), where each entry is a vector
        of coordinates in the original space after transformation and translation.

    Raises
    ------
    ValueError
        If shapes are inconsistent or invalid.

    Examples
    --------
    >>> grid = generate_coordinate_grid((100, 200), (10, 20))
    >>> grid.shape
    (10, 20, 2)

    >>> # Rotate 45° and shift center by (10, 5)
    >>> theta = np.pi / 4
    >>> rot = np.array([[np.cos(theta), -np.sin(theta)],
    ...                 [np.sin(theta),  np.cos(theta)]])
    >>> offset = np.array([10, 5])
    >>> grid = generate_coordinate_grid((100, 100), (10, 10), transform=rot, offset=offset)
    """
    ndim = len(source_shape)
    if ndim != len(target_shape):
        raise ValueError("source_shape and target_shape must have the same number of dimensions")

    # Build sampling axes
    axes = [
        np.linspace(0, s - 1, t)
        for s, t in zip(source_shape, target_shape)
    ]
    mesh = np.meshgrid(*axes, indexing='ij')         # shape: (N, *target_shape)
    grid = np.stack(mesh, axis=-1)                   # shape: (*target_shape, N)

    # Linear transform
    if transform is not None:
        transform = np.asarray(transform)
        if transform.shape != (ndim, ndim):
            raise ValueError(f"transform must have shape ({ndim}, {ndim})")
        grid = np.einsum('...i,ij->...j', grid, transform)

    # Translation
    if offset is not None:
        offset = np.asarray(offset)
        if offset.shape != (ndim,):
            raise ValueError(f"offset must have shape ({ndim},)")
        grid = grid + offset

    return grid


def find_mirror_point_boundary(
    point, box_size_periodic=[np.inf, np.inf, np.inf], is_self=True
):
    """
    For point near the periodic boundary condition, find all the mirror points across the periodic boundary.

    It only works for the index between [-1,0] and [N-1, N], where N is the maximum index in the corresponding dimension.
    For example, for point = (-1, 127, 127.5) and box_size = [128, np.inf, 128], the return should be a numpy array of
    [(127.0, 127.0, 127.5),
    (127.0, 127.0, -0.5),
    (-1.0, 127.0, 127.5),
    (-1.0, 127.0, -0.5)]

    Parameters
    ----------
    point : array like, (,3)
            Index of the given point

    box_size_periodic : array of three floats, or one float, optional
                        The number of indices in each dimension, x, y, z.
                        If box_size is x, it will be interprepted as [x,x,x].
                        If one of the boundary is not periodic, the corresponding value in box_size is np.inf.
                        For example, if the box is periodic in x and y dimension, and the possible maximum index is X and Y,
                        box_size should be [X+1, Y+1, np.inf].
                        Default is [np.inf, np.inf, np.inf], which means the function only return the point itself.

    is_self : bool, optional
              If the original point itself is included in the output.
              Default is True

    Returns
    -------
    return_name : mirror_points, (N,3)
                  The indices of all the mirror points

    Dependencies
    ------------
    - Numpy : 1.26.4

    Called by
    ---------
    - .disclination.defect_neighbor_possible_get()
    """

    from itertools import product

    box_size = array_from_single_or_list(box_size_periodic)
    point = np.array(point)

    point = np.where(box_size == np.inf, point, point % box_size)

    mirrors = [[value] for value in point]
    for i, mirror in enumerate(mirrors):
        N = box_size[i]
        value = point[i]
        if N != np.inf:
            if -1 <= value <= 0:
                mirror.append(value + N)
            elif N - 1 <= value <= N:
                mirror.append(value - N)

    mirror_points = np.array(list(product(*mirrors)))

    if not is_self:
        mirror_points = mirror_points[1:]

    return mirror_points


def unwrap_trajectory(points, box_size_periodic=[np.inf, np.inf, np.inf]):
    """
    Unwrap the points which compose a line crossing the periodic boundary

    When a line cross the periodic boundary and the indices of each point is wrap,
    we will see a sudden "jump" between the indices of points which on different sides of the boundary.
    This function is designed to erase such jump to make the indices of points continuous,
    as unwrapping the points.

    Parameters
    ----------
    points : array like, (N,3)
             The indices of points composing the line.
             N is the number of points

    box_size_periodic : array of three floats, or one float, optional
                        The number of indices in each dimension, x, y, z.
                        If box_size is x, it will be interprepted as [x,x,x].
                        If one of the boundary is not periodic, the corresponding value here is np.inf.
                        For example, if the box is periodic in x and y dimension, and the possible maximum index is X and Y,
                        box_size_periodic should be [X+1, Y+1, np.inf].
                        Default is [np.inf, np.inf, np.inf], which means the function only return the point itself.


    Returns
    -------
    points_unwrap : numpy.ndarray, (N,3)
                    The indices of unwrapped points.

    Dependencies
    ------------
    - Numpy : 1.26.4
    - .general.array_from_single_or_list()

    Called by
    ---------
    - .disclination.defect_classify_into_lines_edge()
    """

    box_size_periodic = array_from_single_or_list(box_size_periodic)
    points = np.array(points)
    deltas = np.diff(points, axis=0)

    for i in range(3):
        if box_size_periodic[i] != np.inf:
            deltas[:, i] = np.where(
                deltas[:, i] > box_size_periodic[i] // 2,
                deltas[:, i] - box_size_periodic[i],
                deltas[:, i],
            )
            deltas[:, i] = np.where(
                deltas[:, i] < -box_size_periodic[i] // 2,
                deltas[:, i] + box_size_periodic[i],
                deltas[:, i],
            )

    points_unwrap = np.concatenate([[points[0]], points[0] + np.cumsum(deltas, axis=0)])

    return points_unwrap


@time_record
def interpolateQ(n, result_points, S=0, is_boundary_periodic=0):

    from scipy.interpolate import interpn

    init_Q = getQ(n, S=S, is_boundary_periodic=is_boundary_periodic)

    init_shape = np.array(np.shape(init_Q)[:3])
    init_points = (
        np.arange(init_shape[0]),
        np.arange(init_shape[1]),
        np.arange(init_shape[2]),
    )

    result_Q = np.zeros((*(np.shape(result_points)[:-1]), 5))
    result_Q[..., 0] = interpn(init_points, init_Q[..., 0, 0], result_points)
    result_Q[..., 1] = interpn(init_points, init_Q[..., 0, 1], result_points)
    result_Q[..., 2] = interpn(init_points, init_Q[..., 0, 2], result_points)
    result_Q[..., 3] = interpn(init_points, init_Q[..., 1, 1], result_points)
    result_Q[..., 4] = interpn(init_points, init_Q[..., 1, 2], result_points)

    return result_Q


@time_record
def interpolateQ_all(n, add_point, S=0, is_boundary_periodic=0):

    add_point = array_from_single_or_list(add_point)
    init_shape = np.array(np.shape(n)[:3])
    result_shape = (init_shape - 1) * (add_point + 1) + 1

    result_points = np.array(
        list(
            product(
                np.linspace(0, init_shape[0] - 1, result_shape[0]),
                np.linspace(0, init_shape[1] - 1, result_shape[1]),
                np.linspace(0, init_shape[2] - 1, result_shape[2]),
            )
        )
    )
    result_points = result_points.reshape((*result_shape, 3))

    result_Q = interpolateQ(
        n, result_points, S=S, is_boundary_periodic=is_boundary_periodic
    )

    return result_Q


def get_ghost_point(point, box_size):
    result = [np.array(point).copy()]

    for i, val in enumerate(point):
        if val >= box_size[i] - 1 or val <= 0:
            current_length = len(result)
            for j in range(current_length):
                new_vector = result[j].copy()
                if val <= 0:
                    new_vector[i] = val + box_size[i]
                else:
                    new_vector[i] = val - box_size[i]
                result.append(new_vector)

    return np.array(result)


def subbox_slices(
    min_vertex,
    max_vertex,
    margin_ratio=0,
    min_limit=[-np.inf, -np.inf, -np.inf],
    max_limit=[np.inf, np.inf, np.inf],
    box_grid_size=np.inf,
):
    #! find which functions are using select_subbox()
    """
    For a numpy array (N, M, L, ...) as values on 3D grid, generate the slice of a subbox,
    whose diagonal vertices (the minimum and maximum value in each dimension) are given.
    For example, assuming there is an orthogonal grid (called data) with shape (30, 20, 15),
    and we want to select the subbox where x in [10, 20], y in [6, 10], z in [4, 7],

    sl0, sl1, sl2, = subbox_slices([10, 5, 2], [20, 10, 7])
    data[sl0, sl1, sl2] is what we need

    This function also supports expand of the vertices to give a bigger subbox, which could be used in, for example, interpolation.
    The periodic boundary condition is also considered.

    Parameters
    ----------

    min_vertex : array of three positive ints,
                 The minimum index index in each dimension, (xmin, ymin, zmin)

    max_vertex : array of three positive ints,
                 The maximum index index in each dimension, (xmax, ymax, zmax)

    margin_ratio : float or array of three floats, optional
                   The subbox is magnified by how many times from initial in each dimension.
                   If only one float x is given, it is interprepted as (x,x,x)
                   Default is 0, no expansion.

    min_limit : array of three floats, optional
                Ihe limit of minimum value in each dimension.
                If the expanded subbox breaks the limit, it will stay at the limit.
                For example, if max_limit = [0, 10, 20], and the min_vertex of the expanded subbox is [-5, 5, 25],
                the min_vertex will turn to be [0, 10, 20].
                Default is [-np.inf, -np.inf, -np.inf], no limit in minimum

    max_limit : array of three floats, optional
                Ihe limit of maximum value in each dimension.
                If the expanded subbox breaks the limit, it will stay at the limit.
                For example, if max_limit = [50, 75, 100], and the max_vertex of the expanded subbox is [40, 90, 90],
                the min_vertex will turn to be [40, 75, 90].
                Default is [-np.inf, -np.inf, -np.inf], no limit in maximum

    box_grid_size : positive int or array of three positive ints
                    The maximum index in each dimension of the whole box.
                    Used to cover periodic boundary.
                    If there is no periodic boundary condition in a certain dimension, then set it to np.inf here.
                    If only one float x is given, it is interprepted as (x,x,x).
                    Default is np.inf, as no periodic boudnary condition in any dimension

    Returns
    -------

    sl0 : the slice in the first dimension (x-axis)

    sl1 : the slice in the second dimension (y-axis)

    sl2 : the slice in the third dimension (z-axis)

    subbox : array of ints, (2,3)
             np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]]), where each value is derived after expansion
    """

    min_vertex = np.array(min_vertex)
    max_vertex = np.array(max_vertex)

    if margin_ratio != 0:
        if len(np.shape([margin_ratio])) == 1:
            margin_ratio = np.array([margin_ratio] * 3)
        xrange, yrange, zrange = max_vertex - min_vertex
        margin = (np.array([xrange, yrange, zrange]) * margin_ratio / 2).astype(int)
        min_vertex = min_vertex - margin
        max_vertex = max_vertex + margin

    xmin, ymin, zmin = np.max(np.vstack((min_vertex, min_limit)), axis=0).astype(int)
    xmax, ymax, zmax = np.min(np.vstack((max_vertex, max_limit)), axis=0).astype(int)

    if len(np.shape([box_grid_size])) == 1:
        N, M, L = np.array([box_grid_size] * 3)
    else:
        N, M, L = box_grid_size

    sl0 = np.array(range(xmin, xmax + 1)).reshape(-1, 1, 1) % N
    sl1 = np.array(range(ymin, ymax + 1)).reshape(1, -1, 1) % M
    sl2 = np.array(range(zmin, zmax + 1)).reshape(1, 1, -1) % L

    return sl0, sl1, sl2, np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])


# ----------------------------------------------------
# The biaxial analysis of directors within a local box
# ----------------------------------------------------


def local_box_diagonalize(n_box):

    # Derive and take the average of the local Q tensor with the director field around the loop
    Q = np.einsum("abci, abcj -> abcij", n_box, n_box)
    Q = np.average(Q, axis=(0, 1, 2))
    Q = Q - np.diag((1, 1, 1)) / 3

    # Diagonalisation and sort the eigenvalues.
    eigval, eigvec = np.linalg.eig(Q)
    eigvec = np.transpose(eigvec)
    eigidx = np.argsort(eigval)[::-1]
    eigval = eigval[eigidx]
    eigvec = eigvec[eigidx]

    return eigvec, eigval


# ---------------------------------------------------------------------------
# Interpolate the directors within a local box containing a disclination loop
# The box is given by its unit vector and vertex coordinates
# ---------------------------------------------------------------------------


def interpolate_subbox(
    vertex_indices,
    axes_unit,
    loop_box,
    n,
    S,
    whole_box_grid_size,
    margin_ratio=2,
    num_min=20,
    ratio=[1, 1, 1],
):

    #! check which functions are using interpolate_subox
    """ """

    diagnal = vertex_indices[1] - vertex_indices[0]
    num_origin = np.einsum("i, ji -> j", diagnal, axes_unit)
    axes = np.einsum("i, ij -> ij", np.abs(num_origin) / num_origin, axes_unit)
    num_origin = np.abs(num_origin)
    num_scale = num_min / np.min(num_origin) * np.array(ratio)
    numx, numy, numz = np.round(num_scale * num_origin).astype(int)

    box = list(product(np.arange(numx + 1), np.arange(numy + 1), np.arange(numz + 1)))
    box = np.array(box)
    box = np.einsum("ai, ij -> aj", box[:, :3], (axes.T / num_scale).T)
    box = box + vertex_indices[0]

    sl0, sl1, sl2, ortho_box = subbox_slices(
        loop_box, whole_box_grid_size, margin_ratio=margin_ratio
    )
    n_box = n[sl0, sl1, sl2]
    S_box = S[sl0, sl1, sl2]

    Q_box = np.einsum("abci, abcj -> abcij", n_box, n_box)
    Q_box = Q_box - np.eye(3) / 3
    Q_box = np.einsum("abc, abcij -> abcij", S_box, Q_box)

    xmin, ymin, zmin = ortho_box[:, 0]
    xmax, ymax, zmax = ortho_box[:, 1]
    points = (
        np.arange(xmin, xmax + 1),
        np.arange(ymin, ymax + 1),
        np.arange(zmin, zmax + 1),
    )

    from scipy.interpolate import interpn

    def interp_Q(index1, index2):
        result = interpn(points, Q_box[..., index1, index2], box)
        result = np.reshape(result, (numx + 1, numy + 1, numz + 1))
        return result

    Q_out = np.zeros((numx + 1, numy + 1, numz + 1, 3, 3))
    Q_out[..., 0, 0] = interp_Q(0, 0)
    Q_out[..., 0, 1] = interp_Q(0, 1)
    Q_out[..., 0, 2] = interp_Q(0, 2)
    Q_out[..., 1, 1] = interp_Q(1, 1)
    Q_out[..., 1, 2] = interp_Q(1, 2)
    Q_out[..., 1, 0] = Q_out[..., 0, 1]
    Q_out[..., 2, 0] = Q_out[..., 0, 2]
    Q_out[..., 2, 1] = Q_out[..., 1, 2]
    Q_out[..., 2, 2] = -Q_out[..., 0, 0] - Q_out[..., 1, 1]

    Q_out = np.einsum("ab, ijkbc, dc -> ijkad", axes_unit, Q_out, axes_unit)

    return Q_out


# ---------------------------------------
# Exponential decay function, for fitting
# ---------------------------------------


def exp_decay(x, A, t):
    return A * np.exp(-x / t)


# -------------------------------------------------------------------------------------
# Change the correlation function into radial coordinate and fit with exponential decay
# -------------------------------------------------------------------------------------


def corr_sphere_fit(
    corr,
    max_init,
    width=200,
    skip_init=25,
    lp0=0.5,
    iterate=2,
    skip_ratio=1,
    max_ratio=10,
):

    from scipy.optimize import curve_fit

    N = np.shape(corr)[0]
    corr = corr[:max_init, :max_init, :max_init].reshape(-1)

    box = list(product(np.arange(max_init), np.arange(max_init), np.arange(max_init)))
    box = np.array(box).reshape((max_init, max_init, max_init, 3))
    r = np.sum(box**2, axis=-1).reshape(-1)
    r = np.sqrt(r) / N * width
    index = r.argsort()
    r = r[index]
    corr = corr[index]

    popt, pcov = curve_fit(exp_decay, r, corr, p0=[corr[0], lp0])
    skip = skip_init

    for i in range(iterate):
        skip_length = popt[1] * skip_ratio
        max_length = popt[1] * max_ratio
        select = (r > skip_length) * (r < max_length)
        popt, pcov = curve_fit(
            exp_decay, r[select], corr[select], p0=[corr[0], popt[1]]
        )
        skip = np.sum(r <= skip_length)

    corr = corr[r < max_length]
    r = r[r < max_length]
    perr = np.sqrt(np.diag(pcov))[1]

    return popt, r, corr, skip, perr


# ------------------------------------------------------
# Derive the persistent length of S by Fourier transfrom
# ------------------------------------------------------


def calc_lp_S(
    S, max_init, width=200, skip_init=25, iterate=2, skip_ratio=1, max_ratio=10, lp0=0.5
):

    from scipy.optimize import curve_fit

    N = np.shape(S)[0]

    S_fourier = np.fft.fftn(S - np.average(S))
    S_spectrum = np.absolute(S_fourier) ** 2
    S_corr = np.real(np.fft.ifftn(S_spectrum)) / N**3

    popt, r, corr, skip, perr = corr_sphere_fit(
        S_corr,
        max_init,
        width=width,
        skip_init=skip_init,
        lp0=lp0,
        iterate=iterate,
        skip_ratio=skip_ratio,
        max_ratio=max_ratio,
    )

    return popt, r, corr, skip, perr


# -------------------------------------------------------------
# Calculate the persistent length of n with Legendre polynomial
# -------------------------------------------------------------


def calc_lp_n(
    n, max_init, width=200, skip_init=25, iterate=2, skip_ratio=1, max_ratio=10, lp0=0.5
):

    from scipy.optimize import curve_fit

    N = np.shape(n)[0]

    Q = np.einsum("nmli, nmlj -> nmlij", n, n)

    Q_fourier = np.fft.fftn(Q - np.average(Q, axis=(0, 1, 2)), axes=(0, 1, 2))
    Q_spectrum = np.absolute(Q_fourier) ** 2
    Q_corr = np.real(np.fft.ifftn(Q_spectrum, axes=(0, 1, 2))) / N**3
    Q_corr = np.sum(Q_corr, axis=(-1, -2))

    popt, r, corr, skip, perr = corr_sphere_fit(
        Q_corr,
        max_init,
        width=width,
        skip_init=skip_init,
        lp0=lp0,
        iterate=iterate,
        skip_ratio=skip_ratio,
        max_ratio=max_ratio,
    )

    return popt, r, corr, skip, perr


# ----------------------------------------------------------------------------
# The default color function for the orientations of directors in 3D nematics
# ----------------------------------------------------------------------------


def n_color_func_default(n):

    theta = np.arccos(n[..., 0])
    phi = np.arctan2(n[..., 2], n[..., 1])

    scalars = (1 - np.cos(2 * theta)) * (np.sin(phi % np.pi) + 0.3)
    # scalars = (1-np.cos(2*theta))*(np.sin(2*phi)+0.3)

    return scalars


def visualize_n(
    n,
    loc=None,
    n_length=1,
    n_shape="cylinder",
    n_opacity=1,
    n_color=(0, 0, 0),
    n_width=1,
):

    from mayavi import mlab

    num = np.shape(n)[0]
    if loc is None:
        loc = np.zeros((num, 3))
    elif np.size(loc) == 3:
        loc = np.broadcast_to(loc, (num, 3))

    cord1 = loc[:, 0] - n[..., 0] * n_length / 2
    cord2 = loc[:, 1] - n[..., 1] * n_length / 2
    cord3 = loc[:, 2] - n[..., 2] * n_length / 2

    nx = n[..., 0]
    ny = n[..., 1]
    nz = n[..., 2]

    scalars = 0
    if isinstance(n_color, str):
        if n_color == "default":
            scalars = n_color_func_default(n)
            object = mlab.quiver3d(
                cord1,
                cord2,
                cord3,
                nx,
                ny,
                nz,
                mode=n_shape,
                scalars=scalars,
                scale_factor=n_length,
                opacity=n_opacity,
                line_width=n_width,
                color=tuple(n_color),
            )
            object.glyph.color_mode = "color_by_scalar"
        if n_color == "immerse":
            for i in range(len(nx)):
                object = mlab.quiver3d(
                    cord1[i],
                    cord2[i],
                    cord3[i],
                    nx[i],
                    ny[i],
                    nz[i],
                    mode=n_shape,
                    scale_factor=n_length,
                    opacity=n_opacity,
                    line_width=n_width,
                    color=tuple(n_color[i]),
                )
    if isinstance(n_color, (tuple, list)):
        object = mlab.quiver3d(
            cord1,
            cord2,
            cord3,
            nx,
            ny,
            nz,
            mode=n_shape,
            scale_factor=n_length,
            opacity=n_opacity,
            line_width=n_width,
            color=tuple(n_color),
        )

    return object, scalars


# ---------------------------------------------------------------------------------
# Visualize the diretors and defects (or the low S region) with given n and S field
# ---------------------------------------------------------------------------------
def visualize_nematics_field(
    n=[0],
    S=[0],
    defect_indices=None,
    plotn=True,
    plotS=True,
    plotdefects=False,
    space_index_ratio=1,
    sub_space=1,
    origin=(0, 0, 0),
    expand_ratio=1,
    n_opacity=1,
    n_interval=(1, 1, 1),
    n_shape="cylinder",
    n_width=2,
    n_plane_index=[],
    n_color_func=n_color_func_default,
    n_length_ratio_to_dist=0.8,
    n_is_colorbar=True,
    n_colorbar_params={},
    n_colorbar_range="default",
    S_threshold=0.45,
    S_opacity=1,
    S_plot_params={},
    S_is_colorbar=True,
    S_colorbar_params={},
    S_colorbar_range=(0, 1),
    defect_opacity=0.8,
    defect_size_ratio_to_dist=2,
    defect_color=(0, 0, 0),
    is_boundary_periodic=False,
    defect_threshold=0,
    defect_print_time=False,
    new_figure=True,
    bgcolor=(1, 1, 1),
    fgcolor=(0, 0, 0),
    figsize=(1920, 1360),
    is_axes=False,
    defect_n_opacity=-1,
    n_is_color_immerse=False,
):

    #! float n_interval with interpolation
    #! different styles for different planes
    #! plotdefects smooth
    #! axes without number
    #! with class
    #! warning about boundary condition of classfication of defect lines
    #! introduction of n_plane_index
    #! defect at cencter
    #! make_plot_directors to visualize_n
    #! change if to is

    """
    Visualize a 3D nematics field using Mayavi.
    The function provides options to plot the director field, scalar order parameter contours, and defect lines.
    The locations of defect points are automatically analyzed by the input director field.

    Parameters
    ----------
    n : numpy array, N x M x L x 3, optional
        The director of each grid point.
        Default is [0], indicating no data of n.
        The shape of n, (N, M, L), must match the shape of S if S is plotted with n or defects.

    S : numpy array, N x M x L, optional
        the scalar order parameter of each grid point.
        Default is [0], which means no data of S.
        The shape of n, (N, M, L), must match the shape of S if S is plotted with n or defects.

    defect_indices : numpy array, optional
                     The indices of defects  of n in the plotted region.
                     If defect_indices = None, the function will detect the defects automatically.
                     Or defect_indices could be manually input in this function.
                     defect_indices should follow the format discribed by .disclination.defect_detect().

    plotn : bool, optional
            If True, plot the director field.
            Default is True.

    plotS : bool, optional
            If True, plot contour of scalar order parameter.
            Default is True.

    plotdefects : bool, optional
                  If True, plot the defect points calculated by the director field.
                  Default is False.

    space_index_ratio : float or array of three floats, optional
                        Ratio between the unit of real space to the unit of grid indices.
                        If the box size is N x M x L and the size of grid of n and S is n x m x l,
                        then space_index_ratio should be (N/n, M/m, L/l).
                        If a single float x is provided, it is interpreted as (x, x, x).
                        Default is 1.

    sub_space : int or array of int as ((xmin, xmax), (ymin, ymax), (zmin, zmax)), optional
                Specifies a sub-box to visualize.
                If an array, it provides the range of sub box by the indices of grid
                If an int, it represents a sub box in the center of the whole box, where the zoom rate is sub_space.
                Default is 1, as the whole box is visualized.

    origin : array of three floats, optional
             Origin of the plot, translating the whole system in real space
             Default is (0, 0, 0), as the system is not translated

    expand_ratio : float or array of three floats, optional
                   Ratio to stretch or compress the plot along each dimension.
                   Usually used to see some structure more clearly.
                   If a single float x is provided, it is interpreted as (x, x, x).
                   Default is 1, as no stretch or compress.
                   Warning: if expand_ratio is not 1 or (1,1,1), the plot will be deformed, so that it does NOT stand for the real space anymore.

    n_opacity : float, optional
                Opacity of the director field.
                Default is 1.

    n_interval : int or array of three int, optional
                 Interval between directors in each dimension, specified in grid indices.
                 For example, if n_interval = (1,2,2), it will plot all directors in x-dimension but plot directors alternatively in y and z-dimension.
                 So large n_internval leads to dilute directors in the plot.
                 If a single int x is provided, it is interpreted as (x, x, x).
                 Default is (1, 1, 1), plotting all directors.

    n_shape : str, optional
              Shape of the director glyphs. It's usually 'cylinder' or '2ddash'.
              Default is 'cylinder'.
              Warning: some of glyph shapes can not be controlled by n_width.
              For example, n_width works for n_shape='2ddash' but does not works for n_shape='cylinder'.

    n_width : float, optional
              Width of the director glyphs.
              It does not work for all shapes. For example, it works for n_shape='2ddash' but does not work for n_shape='cylinder'.
              Default is 2.

    n_color_func : function or array of three float, optional
                   Color for the director glyphs.
                   If it is a function, the input should be an array of shape (N, M, L, 3) and output should be array of shape (N,M,L), which is the scalars to color the directors.
                   If it is an array, it specifies RGB color for all directors.
                   For example, if n_color_func=(1,0,0), all directors will be red.
                   Default is n_color_func_default, a function I personally used to distinguish directors with different orientation.
                   Warning: Two directors with the SAME color could have DIFFERENT orientation,
                   while two directors with different color always have different orientations.

    n_length_ratio_to_dist : float, optional
                             Ratio bwtween the length of director's glyph and the minimal distance between two glyphs.
                             Used to set the length of direcor's glyph.
                             Default is 0.8.

    n_is_colorbar : bool, optional
                    If True, display a colorbar for the director field when n_color_func is a function.
                    Default is True.

    n_colorbar_params : dict, optional
                        Parameters for the director field colorbar.
                        For example, {"orientation": "vertical"} will make the colorbar vertical.
                        Check the document of mlab.colorbar() to see all options.
                        Default is {}.

    n_colorbar_range : array of two float, or "default', or 'max', optional
                       Range for the director field colorbar.
                       If n_colorbar_range is an array of two floats (a,b), the range will be (a,b).
                       If n_colorbar_range='max', the range will cover the minimum and maximum value of scalars (read the introduction of n_color_func).
                       If n_colorbar_range='default', it only works for n_color_func=n_color_func_default. The range will be (0, 2.6) as the therotical range of the default function.
                       Default is 'default'.

    S_threshold : float, optional
                  Threshold for plotting scalar order parameter contours.
                  Default is 0.45.

    S_opacity : float, optional
                Opacity of the scalar order parameter field. Default is 1.

    S_plot_params : dict, optional
                    Parameters for plotting the scalar order parameter field, except opacity and threshold.
                    For example, {'color': (1,1,1)} will make the contours white.
                    Check the document of mlab.contour3D() to see all options.
                    Default is {}.

    S_is_colorbar : bool, optional
                    If True, display a colorbar for the scalar order parameter field based on the value of S.
                    If the color of contours is set, colorbar will NOT be presented.
                    Default is True.

    S_colorbar_params : dict, optional
                        Parameters for the scalar order parameter colorbar.
                        For example, {"orientation": "vertical"} will make the colorbar vertical.
                        Check the document of mlab.colorbar() to see all options.
                        Default is {}.

    S_colorbar_range : array of two float, optional
                       Range for the scalar order parameter colorbar.
                       Default is (0, 1).

    defect_opacity : float, optional
                     Opacity of the defect points.
                     Default is 0.8.

    defect_size_ratio_to_dist : float, optional
                                The ratio bwtween the size of defect points to the minimal distance between two glyphs.
                                Used to set the size of defect points.
                                Default is 2.

    is_boundary_periodic : bool, optional
                        If True, use periodic boundary conditions during defects detection.
                        Default is False.
                        Warning: If only part of the box is selected by sub_space so that it will not have the periodic boundary condition,
                        is_boundary_periodic will be reset to False automatically if it is initially True.

    defect_color : array of three floats, optional
                   Color of defect points in RGB.
                   Default is (0,0,0), black.

    defect_threshold : float, optional
                       Threshold for detecting defects.
                       When calculating the winding number, if the inner product of neighboring directors after one loop is smaller than defect_threshold,
                       the center of the loop is identified as one defect.
                       Default is 0.

    defect_print_time : bool, optional
                        If True, print the time taken to detect defects in each dimension.
                        Default is False.

    new_figure : bool, optional
                 If True, create a new figure for the plot.
                 Default is True.

    bgcolor : array of three floats, optional
              Background color of the plot in RGB.
              Default is (1, 1, 1), white.

    fgcolor : array of three floats, optional
              Foreground color of the plot in RBG.
              It is the color of all text annotation labels (axes, orientation axes, scalar bar labels).
              It should be sufficiently far from bgcolor to see the annotation texts.
              Default is (0,0,0), black

    figsize : tuple of two floats, optional,
              The size of the figure as (sizeX, sizeY).
              Default is (1920, 1360)

    is_axes : bool, optional
              If True, display axes in the plot in the unit of real space.
              Default is False.

    defect_n_opacity : float, optional
                       The opacity of directors around the defects.
                       Default is -1, which means not to distinguish the directors around the defects or not

    n_is_color_immerse : bool, optional
                         If true, the directors will colored by nematics_color_immerse(), which use the immersion from RP2 to R3,
                         where RP2 is the state space of 3D nematics and R3 is the state space of RGB.
                         The specific advantage and shortage with this color function is briefly introduced in #! a document
                         Here, to plot n with this method, the directors have to be plotted with a loop, rather than a parallel function,
                         so it will need much more time.
                         This flag will cover n_color_func and n_is_color_bar.
                         Default is False, using the parallel way to color n.


    Dependencies
    ------------
    - NumPy: 1.26.4
    - mayavi: 4.8.2

    """
    from .disclination import defect_detect, defect_vinicity_grid

    # examine the input data
    n = np.array(n)
    S = np.array(S)
    if S.size == 1 and plotS == True:
        raise NameError("No data of S input to plot contour plot of S.")
    if n.size == 1 and (plotn == True or plotdefects == True):
        raise NameError("no data of n input to plot director field or defects.")
    if S.size != 1 and n.size != 1:
        if np.linalg.norm(np.array(np.shape(n)[:3]) - np.array(np.shape(S))) != 0:
            raise NameError("The shape of n and S should be the same.")

    from mayavi import mlab

    # examine the input parameters
    space_index_ratio = array_from_single_or_list(space_index_ratio)
    n_interval = array_from_single_or_list(n_interval)
    expand_ratio = array_from_single_or_list(expand_ratio)

    if np.linalg.norm(expand_ratio - 1) != 0 and is_axes == True:
        print("\nWarning: The expand_ratio is not (1,1,1)")
        print("This means the distance between glyphs are changed")
        print(
            "The values on the coordinate system (axes of figure) does NOT present the real space."
        )
    if n_shape == "cylinder" and n_width != 2:
        print(
            "\nWarning: n_shape=cylinder, whose thickness can not be controlled by n_width."
        )

    # the basic axes for the plotting in real space of the entire box
    if np.size(n) > 1:
        Nx, Ny, Nz = np.array(np.shape(n)[:3])
    else:
        Nx, Ny, Nz = np.array(np.shape(S)[:3])
    x = np.linspace(0, Nx * space_index_ratio[0] - 1, Nx)
    y = np.linspace(0, Ny * space_index_ratio[1] - 1, Ny)
    z = np.linspace(0, Nz * space_index_ratio[2] - 1, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # create a new figure with the given background if needed
    if new_figure:
        mlab.figure(bgcolor=bgcolor, fgcolor=fgcolor, size=figsize)

    # select the sub box to be plotted
    if len(np.shape([sub_space])) == 1:
        indexx = np.arange(
            int(Nx / 2 - Nx / 2 / sub_space), int(Nx / 2 + Nx / 2 / sub_space)
        )
        indexy = np.arange(
            int(Ny / 2 - Ny / 2 / sub_space), int(Ny / 2 + Ny / 2 / sub_space)
        )
        indexz = np.arange(
            int(Nz / 2 - Nz / 2 / sub_space), int(Nz / 2 + Nz / 2 / sub_space)
        )
    else:
        indexx = np.arange(sub_space[0][0], sub_space[0][-1] + 1)
        indexy = np.arange(sub_space[1][0], sub_space[1][-1] + 1)
        indexz = np.arange(sub_space[2][0], sub_space[2][-1] + 1)

    inx, iny, inz = np.meshgrid(indexx, indexy, indexz, indexing="ij")
    ind = (inx, iny, inz)

    # when we select the subbox, we need to know the "origin" of the subbox in the unit of indices, as start_point
    # in the following, we might translate the indices by start_point
    if len(np.shape([sub_space])) == 1:
        start_point = [
            x[int(Nx / 2 - Nx / 2 / sub_space)],
            y[int(Ny / 2 - Ny / 2 / sub_space)],
            z[int(Nz / 2 - Nz / 2 / sub_space)],
        ]
    else:
        start_point = [x[sub_space[0][0]], y[sub_space[1][0]], z[sub_space[2][0]]]

    # plot defects
    if plotdefects:

        if isinstance(defect_indices, np.ndarray):
            print("\ndefecr_indices are manually input")
        else:
            print("\nNo defect_indices input.")
            print("Start to detect defect automatically.")

            if sub_space != 1 and is_boundary_periodic == True:
                print(
                    "\nWarning: The periodic boundary condition is only possible for the whole box"
                )
                print("Automatically deactivate the periodic boundary condition")
                print("Read the document for more information.")
                is_boundary_periodic = False

            # find defects
            print("\nDetecting defects")
            defect_indices = defect_detect(
                n[ind],
                is_boundary_periodic=is_boundary_periodic,
                threshold=defect_threshold,
                print_time=defect_print_time,
            )
            print("Finished!")

        # Move the defect points to have the correct location in real space
        defect_indices = np.einsum(
            "na, a, a -> na", defect_indices, space_index_ratio, expand_ratio
        )

        defect_indices = defect_indices + np.broadcast_to(
            start_point, (np.shape(defect_indices)[0], 3)
        )
        defect_indices = defect_indices + np.broadcast_to(
            origin, (np.shape(defect_indices)[0], 3)
        )

        # set the size of defect points
        distx = (x[indexx[1]] - x[indexx[0]]) * expand_ratio[0]
        disty = (y[indexy[1]] - y[indexy[0]]) * expand_ratio[1]
        distz = (z[indexz[1]] - z[indexz[0]]) * expand_ratio[2]
        dist = [distx, disty, distz]
        defect_size = np.min(dist) * defect_size_ratio_to_dist
        print(f"\nSize of defect points: {defect_size}")

        # make plot of defect points
        mlab.points3d(
            defect_indices[:, 0],
            defect_indices[:, 1],
            defect_indices[:, 2],
            scale_factor=defect_size,
            opacity=defect_opacity,
            color=defect_color,
        )

    # plot contours of S
    if plotS:

        if np.min(S[ind]) >= S_threshold or np.max(S[ind]) <= S_threshold:

            print(f"\nThe range of S is ({np.min(S[ind])}, {np.max(S[ind])})")
            print("The threshold of contour plot of S is out of range")
            print("No S region is plotted")

        else:
            # the grid in real space to plot S
            cord1 = X[ind] * expand_ratio[0] + origin[0]
            cord2 = Y[ind] * expand_ratio[1] + origin[1]
            cord3 = Z[ind] * expand_ratio[2] + origin[2]

            # check if the color of contours is set
            S_plot_color = S_plot_params.get("color")

            # make plot
            S_region = mlab.contour3d(
                cord1,
                cord2,
                cord3,
                S[ind],
                contours=[S_threshold],
                opacity=S_opacity,
                **S_plot_params,
            )
            # make colorbar of S
            if S_plot_color == None:
                if S_is_colorbar:
                    S_colorbar_label_fmt = S_colorbar_params.get("label_fmt", "%.2f")
                    S_colorbar_nb_labels = S_colorbar_params.get("nb_labels", 5)
                    S_colorbar_orientation = S_colorbar_params.get(
                        "orientation", "vertical"
                    )
                    S_lut_manager = mlab.colorbar(
                        object=S_region,
                        label_fmt=S_colorbar_label_fmt,
                        nb_labels=S_colorbar_nb_labels,
                        orientation=S_colorbar_orientation,
                        **S_colorbar_params,
                    )
                    S_lut_manager.data_range = S_colorbar_range
            elif S_is_colorbar == True:
                print("\nThe color of S is set. So there is no colorbar for S.")

    # plot directors
    if plotn:

        # the axes to plot n in real space is selected by n_interval
        indexx_n = indexx[:: n_interval[0]]
        indexy_n = indexy[:: n_interval[1]]
        indexz_n = indexz[:: n_interval[2]]
        indexall = (indexx, indexy, indexz)

        # set the length of directors' glyph
        distx = (x[indexx_n[1]] - x[indexx_n[0]]) * expand_ratio[0]
        disty = (y[indexy_n[1]] - y[indexy_n[0]]) * expand_ratio[1]
        distz = (z[indexz_n[1]] - z[indexz_n[0]]) * expand_ratio[2]
        dist = [distx, disty, distz]
        n_length = np.min(dist) * n_length_ratio_to_dist
        print(f"\nThe length of directors' glyph: {n_length}")

        if len(n_plane_index) == 0:
            ind_init = np.array(list(product(indexx_n, indexy_n, indexz_n)))
        else:
            ind_init = np.empty((0, 3), int)
            defect_n = np.empty((0, 3), int)
            for i, planes in enumerate(n_plane_index):
                if len(planes) != 0:
                    indexall_n = [indexx_n, indexy_n, indexz_n]
                    indexall_n[i] = indexall[i][planes]
                    ind_init = np.vstack(
                        [ind_init, np.array(list(product(*indexall_n)))]
                    )
                    inx, iny, inz = np.meshgrid(
                        indexall_n[0], indexall_n[1], indexall_n[2], indexing="ij"
                    )
                    ind_local = (inx, iny, inz)
                    planes_defect_detect = np.zeros(3)
                    planes_defect_detect[i] = 1
                    defect_local = defect_detect(
                        n[ind_local],
                        planes=planes_defect_detect,
                        threshold=defect_threshold,
                        print_time=defect_print_time,
                    )
                    # defect_n_local = find_defect_n(defect_local)
                    defect_n_local = (
                        defect_vinicity_grid(defect_local, num_shell=1)
                        .reshape(-1, 3)
                        .astype(int)
                    )
                    defect_n_local[:, 0] = indexall_n[0][defect_n_local[:, 0]]
                    defect_n_local[:, 1] = indexall_n[1][defect_n_local[:, 1]]
                    defect_n_local[:, 2] = indexall_n[2][defect_n_local[:, 2]]
                    defect_n_local = defect_n_local + np.broadcast_to(
                        origin, (np.shape(defect_n_local)[0], 3)
                    )
                    defect_n = np.concatenate([defect_n, defect_n_local])

        if defect_n_opacity > 0:
            # if not plotdefects: ###############################################! for bulk
            #     print('\nDetecting defects')
            #     defect_indices = defect_detect(n[ind_init], threshold=defect_threshold,
            #                                 print_time=defect_print_time)
            #     print('Finished!')
            #     defect_indices = defect_indices + np.broadcast_to(start_point, (np.shape(defect_indices)[0],3))
            # defect_n = find_defect_n(defect_indices_n, size=[np.max(indexx_n)+1, np.max(indexy_n)+1, np.max(indexz_n)+1])

            dtype = [("x", int), ("y", int), ("z", int)]
            tempA = ind_init.view(dtype)
            tempB = defect_n.view(dtype)
            n_order_ind = np.setdiff1d(tempA, tempB)
            n_defect_ind = np.setdiff1d(tempA, n_order_ind)
            n_order_ind = n_order_ind.view(int).reshape(-1, 3)
            n_defect_ind = n_defect_ind.view(int).reshape(-1, 3)

        else:
            n_order_ind = ind_init
            n_defect_ind = np.array([0])

        n_order_ind = tuple(n_order_ind.T)
        n_defect_ind = tuple(n_defect_ind.T)

        def make_plot_directors(ind, n_opacity, n_is_color_immerse=False):

            cord1 = (
                (X[ind]) * expand_ratio[0] - n[ind][..., 0] * n_length / 2 + origin[0]
            )
            cord2 = (
                (Y[ind]) * expand_ratio[1] - n[ind][..., 1] * n_length / 2 + origin[1]
            )
            cord3 = (
                (Z[ind]) * expand_ratio[2] - n[ind][..., 2] * n_length / 2 + origin[2]
            )

            nx = n[ind][..., 0]
            ny = n[ind][..., 1]
            nz = n[ind][..., 2]

            # examine if directors are colored by function or uniform color
            if n_is_color_immerse == False:
                try:
                    len(
                        n_color_func
                    )  # if n_color_func is a list, to plot the director with this list
                    scalars = X[ind] * 0
                    n_color_scalars = False
                    n_color = n_color_func
                except:
                    scalars = n_color_func(n[ind])
                    n_color_scalars = True
                    n_color = (1, 1, 1)
            else:
                n_color_scalars = False
                scalars = X[ind] * 0
                n_color = nematics_color_immerse(n[ind])
                print("\nWarning! n_is_color_immerse is true, it will be much slower")
                print("n_color_func and n_is_colorbar will be ignored ")

            # make plot
            if n_is_color_immerse == False:
                object = mlab.quiver3d(
                    cord1,
                    cord2,
                    cord3,
                    nx,
                    ny,
                    nz,
                    mode=n_shape,
                    scalars=scalars,
                    scale_factor=n_length,
                    opacity=n_opacity,
                    line_width=n_width,
                    color=tuple(n_color),
                )
                if n_color_scalars:
                    object.glyph.color_mode = "color_by_scalar"
            else:
                for i in range(len(nx)):
                    object = mlab.quiver3d(
                        cord1[i],
                        cord2[i],
                        cord3[i],
                        nx[i],
                        ny[i],
                        nz[i],
                        mode=n_shape,
                        scale_factor=n_length,
                        opacity=n_opacity,
                        line_width=n_width,
                        color=tuple(n_color[i]),
                    )

            return object, n_color_scalars, scalars

        vector, n_color_scalars, scalars = make_plot_directors(
            n_order_ind, n_opacity=n_opacity, n_is_color_immerse=n_is_color_immerse
        )

        if np.size(n_defect_ind) > 1:
            make_plot_directors(
                n_defect_ind,
                n_opacity=defect_n_opacity,
                n_is_color_immerse=n_is_color_immerse,
            )

        # make colorbar of n
        if n_color_scalars == True:
            if n_is_colorbar:
                n_colorbar_label_fmt = n_colorbar_params.get("label_fmt", "%.2f")
                n_colorbar_nb_labels = n_colorbar_params.get("nb_labels", 5)
                n_colorbar_orientation = n_colorbar_params.get(
                    "orientation", "vertical"
                )

                n_lut_manager = mlab.colorbar(
                    object=vector,
                    label_fmt=n_colorbar_label_fmt,
                    nb_labels=n_colorbar_nb_labels,
                    orientation=n_colorbar_orientation,
                    **n_colorbar_params,
                )

                if n_colorbar_range == "max":
                    n_colorbar_range = (np.min(scalars), np.max(scalars))
                elif n_colorbar_range == "default":
                    if n_color_func == n_color_func_default:
                        n_colorbar_range = (0, 2.6)
                    else:
                        print(
                            "\nWarning: When n_color_func is not the default function,"
                        )
                        print(
                            "n_colorbar_range should be 'max', or set by hand, but not 'default'."
                        )
                        print(
                            "Here n_colorbar_range is automatically turned to be 'max'."
                        )
                        print("Read the document for more information")

                n_lut_manager.data_range = n_colorbar_range
                if plotS and S_is_colorbar:
                    print("\nWarning: The colorbars for n and S both exist.")
                    print(
                        "These two colorbars might interfere with each other visually."
                    )

        elif n_is_colorbar == True:
            print("\nThe color of n is set. So there is no colorbar for n.")

    # add axes if neede
    if is_axes:
        mlab.axes()


def nematics_color_embed(n):

    n = np.array(n)

    CMYK = np.zeros((*(np.shape(n)[:-1]), 4))
    RGB = np.zeros((*(np.shape(n)[:-1]), 3))

    CMYK[..., 0] = n[..., 0] * n[..., 1]
    CMYK[..., 1] = n[..., 0] * n[..., 2]
    CMYK[..., 2] = n[..., 1] ** 2 - n[..., 2] ** 2
    CMYK[..., 3] = n[..., 1] * n[..., 2]

    CMYK = CMYK / 2 + 0.5

    RGB[..., 0] = (1 - CMYK[..., 0]) * (1 - CMYK[..., 3])
    RGB[..., 1] = (1 - CMYK[..., 1]) * (1 - CMYK[..., 3])
    RGB[..., 2] = (1 - CMYK[..., 2]) * (1 - CMYK[..., 3])

    return RGB


def nematics_color_immerse(n):

    n = np.array(n)

    RGB = np.zeros((*(np.shape(n)[:-1]), 3))

    x = n[..., 0]
    y = n[..., 1]
    z = n[..., 2]

    x2 = x**2
    y2 = y**2
    z2 = z**2

    RGB[..., 0] = (
        (2 * x2 - y2 - z2)
        + 2 * y * z * (y2 - z2)
        + z * x * (x2 - z2)
        + x * y * (y2 - x2)
    )
    RGB[..., 1] = (y2 - z2) + z * x * (z2 - x2) + x * y * (y2 - x2)
    RGB[..., 2] = (x + y + z) * ((x + y + z) ** 3 + 4 * (y - x) * (z - y) * (x - z))

    RGB[..., 0] = RGB[..., 0] / 2
    RGB[..., 1] = RGB[..., 1] * 7 / 8
    RGB[..., 2] = RGB[..., 2] / 8

    result = np.zeros((*(np.shape(n)[:-1]), 3))
    result[..., 0] = 1.01667 * RGB[..., 0] - 0.3 * RGB[..., 1] - 0.48333 * RGB[..., 2]
    result[..., 1] = -1.01667 * RGB[..., 0] - 1.5 * RGB[..., 1] - 1.31667 * RGB[..., 2]
    result[..., 2] = -0.18333 * RGB[..., 0] + 0.3 * RGB[..., 1] + 1.31667 * RGB[..., 2]

    result[..., 0] = result[..., 0] / 2.1 + 0.45
    result[..., 1] = result[..., 1] / 4.2 + 0.51
    result[..., 2] = result[..., 2] / 2.0 + 0.23

    return result
