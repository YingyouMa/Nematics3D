"""
Grid, coordinate-transform, and periodic-boundary utilities.
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .datatypes import (
    DimensionPeriodicInput,
    Vect,
    as_Tensor,
    as_Vect,
    as_dimension_info,
    as_points,
)


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
    alignment: str = "bottom-left",
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """
    Generate a 2D coordinate grid with fixed step sizes.

    Parameters
    ----------
    size1, size2 : float
        Extent of the domain along axis-1 and axis-2.

    step1, step2 : float
        Fixed step size along each axis.

    alignment : {"bottom-left", "center"}
        Grid generation mode. ``bottom-left`` starts the grid at 0 and grows
        toward positive directions only. ``center`` guarantees that 0 is a real
        grid point and expands symmetrically along positive/negative directions.

    Returns
    -------
    grid : np.ndarray
        Continuous coordinate grid of shape (n1, n2, 2),
        where grid[i, j] = (x, y).

    grid_int : np.ndarray
        Integer index grid of shape (n1, n2, 2).

    size_eff : tuple of float
        The effective sizes (size1_eff, size2_eff) actually covered.
    """
    alignment = str(alignment)
    if alignment == "bottom-left":
        n1 = int(np.floor(size1 / step1)) + 1
        n2 = int(np.floor(size2 / step2)) + 1

        axis1 = np.arange(n1, dtype=float) * step1
        axis2 = np.arange(n2, dtype=float) * step2
        axis1_int = np.arange(n1)
        axis2_int = np.arange(n2)

        size1_eff = (n1 - 1) * step1
        size2_eff = (n2 - 1) * step2

    elif alignment == "center":
        n1_half = int(np.floor(size1 / step1 / 2))
        n2_half = int(np.floor(size2 / step2 / 2))

        axis1 = np.arange(-n1_half, n1_half + 1, dtype=float) * step1
        axis2 = np.arange(-n2_half, n2_half + 1, dtype=float) * step2
        axis1_int = np.arange(axis1.shape[0])
        axis2_int = np.arange(axis2.shape[0])

        size1_eff = 2 * n1_half * step1
        size2_eff = 2 * n2_half * step2
    else:
        raise ValueError(
            f"alignment must be 'bottom-left' or 'center', got {alignment!r}"
        )

    mesh = np.meshgrid(axis1, axis2, indexing="ij")
    grid = np.stack(mesh, axis=-1)

    mesh_int = np.meshgrid(axis1_int, axis2_int, indexing="ij")
    grid_int = np.stack(mesh_int, axis=-1)

    return grid, grid_int, (size1_eff, size2_eff)


class _GridTransformIdentity:
    """
    Sentinel representing the canonical identity grid transform.

    This is intentionally identity-based, like ``UNSET`` in ``datatypes``.
    Do not make it array-like: callers that need a numeric matrix should handle
    this sentinel explicitly so the fast path is not lost through coercion.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "GRID_TRANSFORM_IDENTITY"

    def __deepcopy__(self, memo):
        del memo
        return self


GRID_TRANSFORM_IDENTITY = _GridTransformIdentity()
GridTransformIdentity = _GridTransformIdentity


def is_grid_transform_identity(transform) -> bool:
    """Return whether ``transform`` should be treated as the identity transform."""
    return transform is GRID_TRANSFORM_IDENTITY or transform is None


def as_grid_transform(transform, name="grid_transform"):
    """Validate a right-handed orthogonal grid transform.

    The transform columns are interpreted as lattice-basis vectors. They may
    carry scale, but shear, reflections, and degenerate axes are not supported.
    """
    if is_grid_transform_identity(transform):
        return transform

    transform = as_Tensor(transform, (3, 3), name=name)
    axis_lengths = np.linalg.norm(transform, axis=0)
    if np.any(axis_lengths <= 1e-12):
        raise ValueError(f"{name} must have three nonzero column vectors.")

    gram = transform.T @ transform
    off_diag = gram - np.diag(np.diag(gram))
    scale_sq = max(float(np.max(axis_lengths) ** 2), 1.0)
    if not np.allclose(off_diag, 0.0, atol=1e-8 * scale_sq):
        raise ValueError(
            f"{name} must define an orthogonal grid basis: its column vectors "
            "may be scaled, but must be pairwise orthogonal."
        )

    det_scale = max(float(np.prod(axis_lengths)), 1.0)
    if np.linalg.det(transform) <= 1e-12 * det_scale:
        raise ValueError(
            f"{name} must define a right-handed grid basis; reflections and "
            "degenerate transforms are not supported."
        )

    return transform


def apply_linear_transform(
    points: np.ndarray,
    transform=GRID_TRANSFORM_IDENTITY,
    offset: Optional[np.ndarray] = None,
    *,
    is_inv: bool = False,
) -> np.ndarray:
    """
    Apply the repository grid transform convention, or its inverse.

    Forward convention:
        ``points_physical = points_index @ transform + offset``

    Inverse convention:
        ``points_index = (points_physical - offset) @ inv(transform)``

    Parameters
    ----------
    points : np.ndarray
        Array of shape (..., N), where N is the dimensionality.
        This can be a coordinate grid or any point set.

    transform : np.ndarray or GRID_TRANSFORM_IDENTITY, optional
        Linear transformation matrix of shape (N, N), or the identity sentinel.

    offset : np.ndarray, optional
        Offset vector (translation) of shape (N,). Defaults to zero.

    is_inv : bool, optional
        Whether to apply the inverse convention instead of the forward one.

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

    if is_grid_transform_identity(transform):
        transform_use = transform
    else:
        transform_use = as_Tensor(transform, (ndim, ndim), name="grid transform")

    if offset is None:
        offset_use = None
    else:
        offset_use = np.asarray(offset)
        if offset_use.shape != (ndim,):
            raise ValueError(f"offset must have shape ({ndim},)")

    if is_inv:
        result = points if offset_use is None else points - offset_use
        if is_grid_transform_identity(transform_use):
            return result
        return np.einsum("...i,ij->...j", result, np.linalg.inv(transform_use))

    if is_grid_transform_identity(transform_use):
        result = points
    else:
        result = np.einsum("...i,ij->...j", points, transform_use)

    if offset_use is not None:
        result = result + offset_use
    return result


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
    version will be created by shifting by +/-N. Non-periodic dimensions (with box size inf)
    are ignored in this mirroring logic.
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


def wrap_points_to_box(
    points: Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]],
    box_size_periodic: DimensionPeriodicInput = np.inf,
    transform=GRID_TRANSFORM_IDENTITY,
    offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Wrap points into the principal periodic box.

    Periodicity is encoded in lattice/index coordinates. When a non-identity
    transform or offset is supplied, input points are interpreted as physical
    coordinates, mapped back to grid coordinates, wrapped there, and mapped
    back to physical coordinates.
    """
    box_size_periodic = as_dimension_info(box_size_periodic)
    points_input = np.asarray(points, dtype=float)
    is_single_point = points_input.ndim == 1
    points = as_points(points_input, name="points to wrap", dim=3)
    points_index = apply_linear_transform(
        points,
        transform=transform,
        offset=offset,
        is_inv=True,
    )

    points_index_wrapped = points_index.copy()
    mask_periodic = np.isfinite(box_size_periodic)
    points_index_wrapped[..., mask_periodic] = np.mod(
        points_index_wrapped[..., mask_periodic],
        box_size_periodic[mask_periodic],
    )
    wrapped = apply_linear_transform(
        points_index_wrapped,
        transform=transform,
        offset=offset,
    )
    return wrapped[0] if is_single_point else wrapped


def shift_to_box(points_unwrap, box_size_periodic, ref_index=10):
    """
    Shift the entire trajectory so that the first point is inside the periodic box.
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
