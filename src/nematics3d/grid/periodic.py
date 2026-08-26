"""Periodic point, trajectory, and cluster helpers."""

from itertools import product
from typing import Optional, Sequence, Union

import numpy as np

from ..datatypes import (
    BoxSizePeriodic,
    Vect,
    as_box_size_periodic,
    as_points,
    as_vector,
)
from .transform import GRID_TRANSFORM_IDENTITY, apply_linear_transform


def generate_mirror_point_periodic_boundary(
    point: Vect(3),
    box_size_periodic: BoxSizePeriodic = np.inf,
    is_self: bool = True,
):
    """Generate nearby mirror images of a point across periodic boundaries.

    Images are produced only when the wrapped point lies within one index unit
    of a periodic edge. Axes marked by positive infinity are ignored.
    """
    box_size = as_box_size_periodic(
        box_size_periodic,
        name="box_size_periodic",
    )
    point = as_vector(
        point,
        name="The position of point which needs to find mirror image",
    )

    point = np.where(box_size == np.inf, point, point % box_size)

    mirrors = [[value] for value in point]
    for i, mirror in enumerate(mirrors):
        size = box_size[i]
        value = point[i]
        if size != np.inf:
            if -1 <= value <= 0:
                mirror.append(value + size)
            elif size - 1 <= value <= size:
                mirror.append(value - size)

    mirror_points = np.array(list(product(*mirrors)))

    if not is_self:
        mirror_points = mirror_points[1:]

    return mirror_points


def wrap_points_to_box(
    points: Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]],
    box_size_periodic: BoxSizePeriodic = np.inf,
    transform=GRID_TRANSFORM_IDENTITY,
    offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Wrap points into the principal periodic box in lattice coordinates.

    With a non-identity transform or offset, physical input coordinates are
    mapped to lattice coordinates, wrapped there, and mapped back.
    """
    box_size_periodic = as_box_size_periodic(
        box_size_periodic,
        name="box_size_periodic",
    )
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


def shift_to_box(
    points_unwrap,
    box_size_periodic: BoxSizePeriodic,
    ref_index=10,
):
    """Shift a complete trajectory so its reference point lies in the box."""
    points_unwrap = np.asarray(points_unwrap, dtype=float)
    box_size = as_box_size_periodic(
        box_size_periodic,
        name="box_size_periodic",
    )

    shifted = points_unwrap.copy()
    for dim in range(3):
        if np.isfinite(box_size[dim]):
            shift_amount = (
                -np.floor(shifted[ref_index, dim] / box_size[dim]) * box_size[dim]
            )
            shifted[:, dim] += shift_amount

    return shifted


def unwrap_trajectory(
    points: Union[np.ndarray, Sequence[Sequence[float]]],
    box_size_periodic: BoxSizePeriodic = np.inf,
    is_start_in_box=False,
    ref_index=0,
    is_reverse=False,
):
    """Unwrap a trajectory across periodic boundaries into a continuous path.

    Consecutive displacements are reduced through the minimum-image convention
    before they are cumulatively reconstructed from the first point.
    """
    box_size_periodic = as_box_size_periodic(
        box_size_periodic,
        name="box_size_periodic",
    )
    points = np.array(points, dtype=float)

    if is_reverse:
        points = points[::-1]

    deltas = np.diff(points, axis=0)

    mask_periodic = np.isfinite(box_size_periodic)
    periods = box_size_periodic
    deltas[:, mask_periodic] -= (
        np.round(deltas[:, mask_periodic] / periods[mask_periodic])
        * periods[mask_periodic]
    )

    points_unwrap = np.vstack([points[0], points[0] + np.cumsum(deltas, axis=0)])

    if is_start_in_box:
        points_unwrap = shift_to_box(
            points_unwrap,
            box_size_periodic,
            ref_index=ref_index,
        )

    if is_reverse:
        points_unwrap = points_unwrap[::-1]

    return points_unwrap


def unfold_cluster(
    points: np.ndarray,
    box_size_periodic: BoxSizePeriodic = np.inf,
):
    """Unfold a periodic cluster into one continuous region."""
    points = np.asarray(points, dtype=float)
    box_size_periodic = as_box_size_periodic(
        box_size_periodic,
        name="box_size_periodic",
    )
    if np.all(np.isinf(box_size_periodic)):
        return points

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
