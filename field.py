# ------------------------------------
# Analysis of Q field in 3D
# Yingyou Ma, Physics @ Brandeis, 2023
# ------------------------------------

import time
from typing import Tuple, Optional, Union, Sequence, List

import numpy as np

# from .general import *
from .datatypes import (
    Vect,
    as_Vect,
    QField5,
    QField9,
    as_QField9,
    nField,
    SField,
    check_Sn,
    GeneralField,
    DimensionFlagInput,
    DimensionPeriodicInput,
    as_dimension_info,
)
from .logging_decorator import logging_and_warning_decorator


@logging_and_warning_decorator()
def Q_diagonalize(
    qtensor: Union[QField5, QField9], logger=None
) -> Tuple[SField, nField]:
    """
    Analytically diagonalize a Q-tensor field to obtain the scalar order parameter (S)
    and the director field (n).

    This implementation uses tensor invariants to compute the largest eigenvalue
    and corresponding eigenvector without calling `np.linalg.eigh` on each grid point,
    which is significantly faster for large 3D fields.

    Parameters
    ----------
    qtensor : QField5 or QField9
        The Q-tensor field to be diagonalized.
        - QField5: shape (..., 5), 5 independent components
        - QField9: shape (..., 3, 3), full symmetric traceless tensor

    logger : logging.Logger, optional
        Logger instance for diagnostic messages.

    Returns
    -------
    S : SField
        Scalar order parameter, shape (...,). Defined as 1.5 × λ_max.

    n : nField
        Director field (unit vector), shape (..., 3).

    Notes
    -----
    - The sign of `n` is not unique: (n, -n) are equivalent.
    - Future work may extend this to biaxial order and negative S cases.

    Raises
    ------
    TypeError
        If `qtensor` is not a float-type NumPy array.

    ValueError
        If `qtensor` shape is not a valid QField5 or QField9.
    """
    Q: QField9 = as_QField9(qtensor)

    # Compute tensor invariants
    logger.debug("Computing tensor invariants (p, q, r).")
    start = time.time()
    p = 0.5 * np.einsum("...ab, ...ba -> ...", Q, Q)
    q = np.linalg.det(Q)
    r = 2 * np.sqrt(p / 3)
    logger.debug(f"Tensor invariants computed in {time.time() - start:.2f} seconds.")

    # Largest eigenvalue λ (before scaling)
    logger.debug("Computing largest eigenvalue λ_max.")
    start = time.time()
    cos_arg = 4 * q / r**3
    cos_arg = np.clip(cos_arg, -1.0, 1.0)  # ensure valid domain
    lambda_max = r * np.cos((1 / 3) * np.arccos(cos_arg))
    logger.debug(f"λ_max computed in {time.time() - start:.2f} seconds.")

    # Director corresponding to λ_max
    logger.debug("Computing director field n.")
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
    logger.debug(f"Director field computed in {time.time() - start:.2f} seconds.")

    # Scale eigenvalue to get S
    S: SField = 1.5 * lambda_max

    return S, n


@logging_and_warning_decorator()
def getQ(n: nField, S: SField = None, logger=None) -> QField9:
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

    n = check_Sn(n, "n", is_3d_strict=False)

    Q = np.einsum("...i, ...j -> ...ij", n, n) - np.eye(3) / 3
    if S is not None:
        S = check_Sn(S, "S", is_3d_strict=False)
        Q = np.einsum("..., ...ij -> ...ij", S, Q)
    else:
        logger.warning(">>> No S input. Set to be 1.")

    return Q


def add_periodic_boundary(
    data: GeneralField, is_boundary_periodic: DimensionFlagInput = 0
) -> GeneralField:
    #! loop
    """
    Extend a physical field with periodic boundary slices in specified dimensions.

    This function appends one extra grid slice along each of the periodic dimensions.
    The added slice is a copy of the first slice along that axis, ensuring periodic continuity.
    If a dimension is non-periodic, it is left unchanged.

    Parameters
    ----------
    data : GeneralField
        Input physical field of shape (Nx, Ny, Nz, ...), where (Nx, Ny, Nz) are spatial dimensions,
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
            (Nx + is_periodic[0], Ny + is_periodic[1], Nz + is_periodic[2], ...)
    """
    is_boundary_periodic = as_dimension_info(is_boundary_periodic)

    if np.any(is_boundary_periodic):
        Nx, Ny, Nz, *rest_shape = data.shape  # Extract the first three dimensions
        output = np.empty(
            (
                Nx + is_boundary_periodic[0],
                Ny + is_boundary_periodic[1],
                Nz + is_boundary_periodic[2],
                *rest_shape,
            ),
            dtype=data.dtype,
        )
        output[:Nx, :Ny, :Nz] = data  # Copy original data into the new array

        # Copy first slices to last.
        if is_boundary_periodic[0]:
            output[Nx] = output[0]
        if is_boundary_periodic[1]:
            output[:, Ny] = output[:, 0]
        if is_boundary_periodic[2]:
            output[:, :, Nz] = output[:, :, 0]
    else:
        output = data

    return output


def align_directors(n_reference: nField, n_target: nField) -> nField:
    """
    Align target director to have similar orientation as reference.
    This is used to handle the nematic symmetry of directors.
    """
    n_reference = check_Sn(n_reference, "n", is_3d_strict=False)
    n_target = check_Sn(n_target, "n", is_3d_strict=False)
    signs = np.sign(np.einsum("...i,...i->...", n_reference, n_target))
    return np.einsum("...,...i->...i", signs, n_target)

def align_stack(stack):
    
    dots = np.einsum('...i,...i->...', stack[:-1], stack[1:])
    
    flips = np.ones(stack.shape[:-1])
    flips[1:] = np.where(dots < 0, -1, 1)
    
    acc_flips = np.cumprod(flips, axis=0)

    return stack * acc_flips[..., np.newaxis]

def generate_coordinate_grid(
    shape_source: Tuple[int, ...], shape_target: Tuple[int, ...]
) -> np.ndarray:
    """
    Generate an N-dimensional coordinate grid over the source domain.

    Parameters
    ----------
    shape_source : tuple of int
        Shape of the original data in N dimensions.

    shape_target : tuple of int
        Desired shape of the resampled grid in N dimensions.

    Returns
    -------
    grid : np.ndarray
        Grid of shape (*shape_target, N), where each entry is a vector
        of coordinates in the original index space.

    Raises
    ------
    ValueError
        If shapes are inconsistent or invalid.
    """
    ndim = len(shape_source)
    if ndim != len(shape_target):
        raise ValueError(
            "shape_source and shape_target must have the same number of dimensions"
        )

    axes = [np.linspace(0, s - 1, t) for s, t in zip(shape_source, shape_target)]
    mesh = np.meshgrid(
        *axes, indexing="ij"
    )  # List of N arrays, each shape (*shape_target)
    grid = np.stack(mesh, axis=-1)  # Shape: (*shape_target, N)

    axes_int = [np.arange(t) for t in shape_target]
    mesh_int = np.meshgrid(*axes_int, indexing="ij")
    grid_int = np.stack(mesh_int, axis=-1)
    grid_int = np.asarray(grid_int)

    steps = np.array(
        [
            (s - 1) / (t - 1) if t > 1 else 0.0
            for s, t in zip(shape_source, shape_target)
        ],
        dtype=float,
    )

    return grid, grid_int, steps


def generate_fixed_step_grid(
    size1: float,
    size2: float,
    step1: float,
    step2: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """
    Generate a 2D coordinate grid with fixed step sizes.

    Parameters
    ----------
    size1, size2 : float
        Extent of the domain along axis-1 and axis-2.
        The grid starts at 0 and does not exceed the given size.

    step1, step2 : float
        Fixed step size along each axis.

    Returns
    -------
    grid : np.ndarray
        Continuous coordinate grid of shape (n1, n2, 2),
        where grid[i, j] = (x, y).

    grid_int : np.ndarray
        Integer index grid of shape (n1, n2, 2),
        where grid_int[i, j] = (i, j).

    size_eff : tuple of float
        The effective sizes (size1_eff, size2_eff) actually covered
        by the grid, computed as:
            size*_eff = (n* - 1) * step*
    """
    # number of grid points (including 0)
    n1 = int(np.floor(size1 / step1)) + 1
    n2 = int(np.floor(size2 / step2)) + 1
    
    # integer index grid
    axis1_int = np.arange(n1)
    axis2_int = np.arange(n2)
    
    mesh_int = np.meshgrid(axis1_int, axis2_int, indexing="ij")
    grid_int = np.stack(mesh_int, axis=-1)  # (n1, n2, 2)
    
    # continuous coordinate grid (mapped from integer indices)
    grid = grid_int.astype(float)
    grid[..., 0] *= step1
    grid[..., 1] *= step2
    
    # effective sizes actually covered
    size1_eff = (n1 - 1) * step1
    size2_eff = (n2 - 1) * step2

    return grid, grid_int, (size1_eff, size2_eff)



def apply_linear_transform(
    points: np.ndarray,
    transform: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply a linear transformation and optional offset to a point cloud.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (..., N), where N is the dimensionality.
        This can be a coordinate grid or any point set.

    transform : np.ndarray, optional
        Linear transformation matrix of shape (N, N). Defaults to identity.

    offset : np.ndarray, optional
        Offset vector (translation) of shape (N,). Defaults to zero.

    Returns
    -------
    transformed : np.ndarray
        Transformed array of shape (..., N).

    Raises
    ------
    ValueError
        If transform or offset shapes are invalid.
    """
    points = np.asarray(points)
    ndim = points.shape[-1]

    if transform is not None:
        transform = np.asarray(transform)
        if transform.shape != (ndim, ndim):
            raise ValueError(f"transform must have shape ({ndim}, {ndim})")
        points = np.einsum("...i,ij->...j", points, transform)

    if offset is not None:
        offset = np.asarray(offset)
        if offset.shape != (ndim,):
            raise ValueError(f"offset must have shape ({ndim},)")
        points = points + offset

    return points


def generate_mirror_point_periodic_boundary(
    point: Vect(3),
    box_size_periodic: DimensionPeriodicInput = np.inf,
    is_self: bool = True,
):
    """
    Find all mirror images of a given point across periodic boundaries, if the point lies
    within one index of a periodic edge.

    This function is used in periodic systems (e.g., simulations) to generate equivalent
    positions of a point that may straddle a periodic boundary. It applies only when the
    point lies near the boundary, i.e., between [-1, 0] or [N-1, N], where N is the box size
    in that dimension.

    For each periodic dimension, if the point lies within one unit of the edge, a mirrored
    version will be created by shifting by ±N. Non-periodic dimensions (with box size ∞)
    are ignored in this mirroring logic.

    Parameters
    ----------
    point : Vect(3)
        array-like of shape (3,)
        The coordinate index of the point to be mirrored. Can include negative or
        out-of-bound values near the edge.

    box_size_periodic : DimensionPeriodicInput, optional
        int or array-like of 3 ints,
        The periodic domain size along each axis. Can be a single int (broadcasted to 3D).
        A value of `np.inf` indicates non-periodic along that dimension.
        Example: [128, 128, np.inf] means periodic in x and y, open in z.

    is_self : bool, optional
        If True (default), the original point is included in the output.
        If False, only mirrored versions are returned.

    Returns
    -------
    mirror_points : ndarray of shape (N, 3)
        An array of all mirror images (including original if `is_self=True`).
        Each row is a 3D coordinate, possibly shifted by ±N in some dimensions.

    Examples
    --------
    >>> find_mirror_point_boundary([-1, 127, 127.5], [128, np.inf, 128])
    array([
        [127. , 127. , 127.5],
        [127. , 127. , -0.5],
        [-1.  , 127. , 127.5],
        [-1.  , 127. , -0.5],
    ])
    """

    from itertools import product

    box_size = as_dimension_info(box_size_periodic)
    point = as_Vect(
        point, name="The position of point which needs to find mirror image"
    )

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


def shift_to_box(points_unwrap, box_size_periodic, ref_index=10):
    """
    Shift the entire trajectory so that the first point is inside the periodic box.

    Parameters
    ----------
    points_unwrap : (N, 3) ndarray
        Already unwrapped trajectory points.

    box_size_periodic : (3,) array-like
        Box size in each dimension (np.inf for non-periodic).

    Returns
    -------
    shifted_points : (N, 3) ndarray
        Trajectory shifted so that the first point is inside [0, L) for periodic dimensions.
    """
    points_unwrap = np.asarray(points_unwrap, dtype=float)
    L = as_dimension_info(box_size_periodic)

    shifted = points_unwrap.copy()
    for dim in range(3):
        if np.isfinite(L[dim]):
            # Wrap the starting point into [0, L)
            shift_amount = -np.floor(shifted[ref_index, dim] / L[dim]) * L[dim]
            shifted[:, dim] += shift_amount

    return shifted


def unwrap_trajectory(
    points: Union[np.ndarray, Sequence[Sequence[float]]],
    box_size_periodic: DimensionPeriodicInput = np.inf,
    is_start_in_box=False,
    ref_index=0,
    is_reverse=False,
):
    """
    Unwrap a trajectory of points across periodic boundaries to produce a geometrically continuous path.

    In periodic systems, when a line crosses the periodic boundary, wrapped coordinates can create
    artificial discontinuities (large jumps between adjacent points). This function detects such jumps
    and corrects them by unwrapping the trajectory, making the path continuous in real space.

    Parameters
    ----------
    points : array-like of shape (N, 3)
        A sequence of points representing a path or line in 3D periodic space.
        Each point should be a length-3 vector.

    box_size_periodic : DimensionPeriodicInput, optional
        int or array-like of 3 ints
        The box size in each dimension (X, Y, Z). If a single int is provided,
        it is broadcasted to all three axes. Set an axis to `np.inf` if it is non-periodic.
        For example: [Lx, Ly, np.inf] means periodic in x and y, but open in z.
        Default is [np.inf, np.inf, np.inf], meaning no unwrapping is applied.

    Returns
    -------
    points_unwrap : np.ndarray of shape (N, 3)
        The unwrapped version of the input points, forming a continuous path.
    """

    box_size_periodic = as_dimension_info(box_size_periodic)
    points = np.array(points, dtype=float)

    if is_reverse:
        points = points[::-1]

    deltas = np.diff(points, axis=0)

    mask_periodic = np.isfinite(box_size_periodic)
    L = box_size_periodic

    # Apply minimum image convention with multi-box handling
    deltas[:, mask_periodic] -= (
        np.round(deltas[:, mask_periodic] / L[mask_periodic]) * L[mask_periodic]
    )

    points_unwrap = np.vstack([points[0], points[0] + np.cumsum(deltas, axis=0)])

    if is_start_in_box:
        points_unwrap = shift_to_box(
            points_unwrap, box_size_periodic, ref_index=ref_index
        )

    if is_reverse:
        points_unwrap = points_unwrap[::-1]

    return points_unwrap


def unfold_cluster(points: np.ndarray, box_size_periodic: np.ndarray = np.inf):
    """
    Unfolds a cluster of points that may cross periodic boundaries into a single continuous region.

    Parameters
    ----------
    points : (N, 3) ndarray
        Coordinates of the point cluster.
        Assumes coordinates are in the range [0, box_size) for periodic dimensions.

    box_size_periodic : float or array-like
        Periodic box size for each dimension.
        - Can be a scalar (same size in all periodic dimensions) or an array of shape (3,).
        - If a dimension size is np.inf, it is treated as non-periodic.

    Returns
    -------
    unfolded : (N, 3) ndarray
        Coordinates of the cluster after unfolding so that all points lie in the same contiguous region.

    Notes
    -----
    This function detects if points are separated across periodic boundaries and applies
    minimal ±box_size translations to bring them together.
    - A reference point (the first point) is chosen.
    - For each point and each periodic dimension:
        * If the distance to the reference point is greater than half the box size,
          the point is shifted by -box_size.
        * If the distance is less than negative half the box size,
          the point is shifted by +box_size.
    - Non-periodic dimensions (size = np.inf) are left unchanged.

    Example
    -------
    >>> points = np.array([[0.1, 0.2, 0.9],
    ...                    [0.15, 0.25, 0.05],  # Crosses z-boundary
    ...                    [0.12, 0.22, 0.95]])
    >>> box = np.array([1.0, 1.0, 1.0])
    >>> unfold_cluster(points, box)
    array([[0.1 , 0.2 , 0.9 ],
           [0.15, 0.25, 1.05],
           [0.12, 0.22, 0.95]])
    """

    points = np.asarray(points, dtype=float)
    if np.all(box_size_periodic == np.inf):
        return points

    box_size_periodic = as_dimension_info(box_size_periodic)

    unfolded = points.copy()
    ref = points[0]

    for i in range(len(points)):
        for dim, size in enumerate(box_size_periodic):
            if size != np.inf:
                delta = points[i, dim] - ref[dim]
                if delta > size / 2:
                    unfolded[i, dim] -= size
                elif delta < -size / 2:
                    unfolded[i, dim] += size

    return unfolded


def n_color_immerse(n: nField) -> List[Tuple]:
    """
    Map a nematic director field to RGB colors for visualization.

    This function encodes the orientation of a unit director vector `n`
    into RGB color values using a nonlinear polynomial mapping followed by
    a fixed linear transformation and scaling. The colormap is an immersion
    from RP^2 to R^3. This ensures that similar orientatioin of n refer to
    similar colors but different orientations might refer to the same color.

    The colormap is modified from boy's surface.

    The color is specifically desined to be distince on white background,
    and to get x, y, z direction closed to red, blue and green colors.

    x: [0.90535893, 0.22874911, 0.22062688]
    y: [0.05416607, 0.27934554, 0.48937438]
    z: [0.30416607, 0.90434554, 0.22687438]

    Parameters
    ----------
    n : array_like, shape (..., 3)
        Nematic director field.
        Can be of arbitrary leading dimensions.

    Returns
    -------
    colors : list of tuples. Each tuple has 3 elements
        RGB color with values typically in [0, 1], suitable for plotting.

    Examples
    --------
    >>> n = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    >>> colors = n_color_immerse(n)
    >>> colors
    [(0.39345357, 0.1364875 , 0.60187625),
     (0.09285714, 0.47428571, 0.605     ),
     (0.51845357, 0.4489875 , 0.47062625)]
    """

    n = check_Sn(n, "n", is_3d_strict=False, is_norm=True)

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

    M = np.array(
        [
            [1.01667, -0.3, -0.48333],
            [-1.01667, -1.5, -1.31667],
            [-0.18333, 0.3, 1.31667],
        ]
    )

    result = np.einsum("...i, ji -> ...j", RGB, M)

    scales = np.array([2.1, 4.2, 2.0])
    offsets = np.array([0.45, 0.51, 0.23])
    result = result / scales + offsets

    colors = []
    for color in result:
        colors.append(tuple(color))

    return colors

