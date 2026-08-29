"""Periodic point, trajectory, and cluster helpers."""

import operator
from typing import Optional, Sequence, Union

import numpy as np

from ..logging_decorator import logging_and_warning_decorator
from ..datatypes import (
    BoxSizePeriodic,
    as_bool,
    as_box_size_periodic,
    as_points,
)
from .transform import GRID_TRANSFORM_IDENTITY, GridTransform, apply_linear_transform


def wrap_points_to_box(
    points: Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]],
    box_size_periodic: BoxSizePeriodic = np.inf,
    transform: GridTransform = GRID_TRANSFORM_IDENTITY,
    offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Wrap points into the principal periodic box in lattice coordinates.

    With a non-identity transform or offset, physical input coordinates are
    mapped to lattice coordinates, wrapped there, and mapped back. A single
    point keeps shape ``(3,)``; a point collection keeps shape ``(N, 3)``.
    Empty collections are returned with shape ``(0, 3)``.
    """
    box_size_periodic = as_box_size_periodic(
        box_size_periodic,
        name="box_size_periodic",
    )

    raw_points = np.asarray(points)
    is_single_point = raw_points.ndim == 1 and raw_points.size != 0
    points = as_points(points, name="points to wrap", d=3)
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
    ref_index: int = 0,
    *,
    is_validate: bool = True,
    is_inplace: bool = False,
) -> np.ndarray:
    """Shift a complete trajectory so its reference point lies in the box.

    By default, inputs are validated and a shifted copy is returned. Set
    ``is_inplace=True`` to modify and return the input array itself. The
    ``is_validate=False`` fast path is intended for already validated internal
    arrays; invalid inputs then have undefined behavior.
    """
    is_validate = as_bool(is_validate, name="is_validate")
    is_inplace = as_bool(is_inplace, name="is_inplace")

    if is_validate:
        if is_inplace:
            if not isinstance(points_unwrap, np.ndarray):
                raise TypeError(
                    "'points_unwrap' must be a NumPy array when " "'is_inplace=True'."
                )
            if not np.issubdtype(points_unwrap.dtype, np.floating):
                raise TypeError(
                    "'points_unwrap' must have a floating dtype when "
                    "'is_inplace=True'."
                )
            if not points_unwrap.flags.writeable:
                raise ValueError(
                    "'points_unwrap' must be writable when 'is_inplace=True'."
                )
            shifted = points_unwrap
        else:
            shifted = as_points(points_unwrap, name="points_unwrap", d=3)

        if len(shifted) == 0:
            raise ValueError("'points_unwrap' must contain at least one point.")
        if is_inplace and not np.all(np.isfinite(shifted)):
            raise ValueError("'points_unwrap' must contain only finite values.")
        if is_inplace and (shifted.ndim != 2 or shifted.shape[1] != 3):
            raise ValueError(
                "'points_unwrap' must have shape (N, 3). " f"Got shape={shifted.shape}."
            )

        box_size = as_box_size_periodic(box_size_periodic, name="box_size_periodic")
        if isinstance(ref_index, (bool, np.bool_)):
            raise TypeError("'ref_index' must be an integer, not a boolean.")
        try:
            ref_index = operator.index(ref_index)
        except TypeError as exc:
            raise TypeError("'ref_index' must be an integer.") from exc
        if not -len(shifted) <= ref_index < len(shifted):
            raise IndexError(
                f"'ref_index'={ref_index} is out of bounds for "
                f"{len(shifted)} points."
            )
    else:
        shifted = points_unwrap if is_inplace else np.array(points_unwrap, copy=True)
        box_size = np.asarray(box_size_periodic)

    for dim in range(3):
        if np.isfinite(box_size[dim]):
            shift_amount = (
                -np.floor(shifted[ref_index, dim] / box_size[dim]) * box_size[dim]
            )
            shifted[:, dim] += shift_amount

    return shifted


@logging_and_warning_decorator(start_finish_level=5)
def unwrap_trajectory(
    points: Union[np.ndarray, Sequence[Sequence[float]]],
    box_size_periodic: BoxSizePeriodic = np.inf,
    *,
    is_start_in_box: bool = False,
    ref_index: int = 0,
    is_reverse: bool = False,
    logger=None,
) -> np.ndarray:
    """Unwrap a trajectory across periodic boundaries into a continuous path.

    Consecutive displacements are reduced through the minimum-image convention
    before they are cumulatively reconstructed from the first point.
    """
    # Normalize the periodic lengths first. A finite entry marks a periodic
    # axis; positive infinity marks an axis that must remain unchanged.
    box_size = as_box_size_periodic(
        box_size_periodic,
        name="box_size_periodic",
    )
    is_start_in_box = as_bool(is_start_in_box, name="is_start_in_box")
    is_reverse = as_bool(is_reverse, name="is_reverse")

    # as_points() handles a single point, an empty collection, dimensionality,
    # finite-coordinate validation, conversion to float, and input isolation.
    # Consequently, every operation below can assume a writable (N, 3) array.
    points = as_points(points, name="points", d=3)
    if len(points) == 0:
        if is_start_in_box:
            raise ValueError(
                "'points' must contain at least one point when "
                "'is_start_in_box=True'."
            )
        logger.debug("Received an empty trajectory; returning shape (0, 3).")
        return points

    logger.debug(
        f"Unwrapping {len(points):,} point(s); "
        f"box_size_periodic={box_size.tolist()}, is_reverse={is_reverse}, "
        f"is_start_in_box={is_start_in_box}."
    )

    # ref_index affects only the optional final translation. Validate it here
    # before entering the trusted shift_to_box() fast path below. Negative
    # indices retain their normal NumPy meaning.
    if is_start_in_box:
        if isinstance(ref_index, (bool, np.bool_)):
            raise TypeError("'ref_index' must be an integer, not a boolean.")
        try:
            ref_index = operator.index(ref_index)
        except TypeError as exc:
            raise TypeError("'ref_index' must be an integer.") from exc
        if not -len(points) <= ref_index < len(points):
            raise IndexError(
                f"'ref_index'={ref_index} is out of bounds for "
                f"{len(points)} points."
            )

    # Reversing changes which endpoint anchors the continuous reconstruction.
    # This is useful when the last sample is the more reliable reference. The
    # result is reversed back to the caller's original ordering at the end.
    if is_reverse:
        logger.debug("Using the final input point as the unwrapping anchor.")
        points = points[::-1]

    # Work with consecutive wrapped displacements instead of absolute
    # positions. There are N - 1 displacements for N input points.
    deltas = np.diff(points, axis=0)

    # Apply the minimum-image convention only along periodic axes. Subtracting
    # the nearest integer multiple of each period maps every displacement into
    # the nearest periodic image. Exact half-period steps follow np.round()'s
    # tie-to-even convention and are therefore inherently direction-ambiguous.
    mask_periodic = np.isfinite(box_size)
    if np.any(mask_periodic):
        periods = box_size[mask_periodic]
        deltas[:, mask_periodic] -= (
            np.round(deltas[:, mask_periodic] / periods) * periods
        )

    # Reconstruct absolute positions from the corrected displacements. Writing
    # cumsum directly into the output avoids an additional full trajectory
    # allocation; the anchor is added afterward to restore absolute location.
    points_unwrap = np.empty_like(points)
    points_unwrap[0] = points[0]
    if len(points_unwrap) > 1:
        np.cumsum(deltas, axis=0, out=points_unwrap[1:])
        points_unwrap[1:] += points_unwrap[0]

    # Unwrapping guarantees continuity but does not guarantee that any point
    # lies inside the principal periodic box. When requested, translate the
    # entire trajectory by whole box lengths so the selected reference does.
    # The output is already validated, independent, floating, and writable, so
    # the internal call safely skips repeated validation and modifies in place.
    if is_start_in_box:
        logger.debug(
            f"Shifting the unwrapped trajectory so point {ref_index} lies in "
            "the principal periodic box."
        )
        shift_to_box(
            points_unwrap,
            box_size,
            ref_index=ref_index,
            is_validate=False,
            is_inplace=True,
        )

    if is_reverse:
        # Restore the caller's original sample ordering after reverse anchoring.
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
