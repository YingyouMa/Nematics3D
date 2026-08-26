"""Grid-transform validation and application helpers."""

from typing import Optional

import numpy as np

from ..datatypes import as_readonly_array, as_tensor


class _GridTransformIdentity:
    """Sentinel representing the canonical identity grid transform.

    This is intentionally identity-based, like ``UNSET`` in ``datatypes``.
    It must not become array-like because callers use identity checks to retain
    the no-transform fast path.
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

    Transform columns are lattice-basis vectors. They may carry scale, but
    shear, reflections, and degenerate axes are unsupported.
    """
    if is_grid_transform_identity(transform):
        return transform

    transform = as_tensor(transform, (3, 3), name=name)
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


def as_readonly_grid_offset(offset):
    """Return one read-only grid offset array, preserving ``None``."""
    if offset is None:
        return None
    return as_readonly_array(offset, dtype=float)


def as_readonly_grid_transform(transform):
    """Return one read-only grid transform array, preserving identity."""
    if is_grid_transform_identity(transform):
        return transform
    return as_readonly_array(transform, dtype=float)


def apply_linear_transform(
    points: np.ndarray,
    transform=GRID_TRANSFORM_IDENTITY,
    offset: Optional[np.ndarray] = None,
    *,
    is_inv: bool = False,
) -> np.ndarray:
    """Apply the repository grid-transform convention, or its inverse.

    Forward transformation uses ``points @ transform + offset``. The inverse
    uses ``(points - offset) @ inv(transform)``. Point arrays may have arbitrary
    leading dimensions and one trailing coordinate axis.
    """
    points = np.asarray(points)
    ndim = points.shape[-1]

    if is_grid_transform_identity(transform):
        transform_use = transform
    else:
        transform_use = as_tensor(transform, (ndim, ndim), name="grid transform")

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
