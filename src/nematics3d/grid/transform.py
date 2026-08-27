"""Grid-transform validation, storage, and application helpers."""

import numpy as np

from ..datatypes import (
    Tensor,
    as_bool,
    as_points,
    as_readonly_array,
    as_tensor,
    as_vector,
)


# GridTransform is a reader-facing semantic annotation. Runtime validation is
# performed by as_grid_transform().
GridTransform = Tensor((3, 3))

_GRID_ORTHOGONAL_RTOL = 1e-8
_GRID_DEGENERATE_RTOL = 1e-12


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


def is_grid_transform_identity(transform) -> bool:
    """Return whether ``transform`` should be treated as the identity transform."""
    return transform is GRID_TRANSFORM_IDENTITY or transform is None


def as_grid_transform(
    transform: GridTransform,
    name="grid_transform",
    *,
    is_readonly: bool = False,
) -> GridTransform:
    """Validate a right-handed orthogonal grid transform.

    Transform columns are lattice-basis vectors. They may carry scale, but
    shear, reflections, and degenerate axes are unsupported.
    """
    is_readonly = as_bool(is_readonly, name="is_readonly")
    if is_grid_transform_identity(transform):
        return GRID_TRANSFORM_IDENTITY

    transform = as_tensor(transform, (3, 3), name=name)
    axis_lengths = np.linalg.norm(transform, axis=0)
    if np.any(axis_lengths <= _GRID_DEGENERATE_RTOL):
        raise ValueError(f"{name} must have three nonzero column vectors.")

    gram = transform.T @ transform
    off_diag = gram - np.diag(np.diag(gram))
    scale_sq = max(float(np.max(axis_lengths) ** 2), 1.0)
    if not np.allclose(
        off_diag,
        0.0,
        rtol=0.0,
        atol=_GRID_ORTHOGONAL_RTOL * scale_sq,
    ):
        raise ValueError(
            f"{name} must define an orthogonal grid basis: its column vectors "
            "may be scaled, but must be pairwise orthogonal."
        )

    det_scale = max(float(np.prod(axis_lengths)), 1.0)
    if np.linalg.det(transform) <= _GRID_DEGENERATE_RTOL * det_scale:
        raise ValueError(
            f"{name} must define a right-handed grid basis; reflections and "
            "degenerate transforms are not supported."
        )

    if is_readonly:
        return as_readonly_array(transform)
    return transform


def as_grid_offset(offset, name="grid_offset", *, is_readonly: bool = False):
    """Validate a three-dimensional grid offset, preserving ``None``."""
    is_readonly = as_bool(is_readonly, name="is_readonly")
    if offset is None:
        return None
    offset = as_vector(offset, d=3, name=name)
    if is_readonly:
        return as_readonly_array(offset)
    return offset


def apply_linear_transform(
    points: np.ndarray,
    transform: GridTransform = GRID_TRANSFORM_IDENTITY,
    offset=None,
    *,
    is_inv: bool = False,
) -> np.ndarray:
    """Apply the repository grid-transform convention, or its inverse.

    Forward transformation uses ``points @ transform + offset``. The inverse
    uses ``(points - offset) @ inv(transform)``. Point arrays may have arbitrary
    leading dimensions and one trailing coordinate axis.
    """
    is_inv = as_bool(is_inv, name="is_inv")
    raw_points = np.asarray(points)
    if raw_points.ndim == 0 or raw_points.shape[-1] != 3:
        raise ValueError(
            "'points' must have a trailing coordinate axis of length 3. "
            f"Got shape={raw_points.shape}."
        )
    points_shape = raw_points.shape
    points = as_points(raw_points.reshape(-1, 3), d=3, name="points").reshape(
        points_shape
    )
    transform_use = as_grid_transform(transform, name="transform")
    offset_use = as_grid_offset(offset, name="offset")

    if is_inv:
        result = points if offset_use is None else points - offset_use
        if is_grid_transform_identity(transform_use):
            return result
        result_flat = result.reshape(-1, 3)
        transformed = np.linalg.solve(transform_use.T, result_flat.T).T
        return transformed.reshape(points_shape)

    if is_grid_transform_identity(transform_use):
        result = points
    else:
        result = np.einsum("...i,ij->...j", points, transform_use)

    if offset_use is not None:
        result = result + offset_use
    return result
