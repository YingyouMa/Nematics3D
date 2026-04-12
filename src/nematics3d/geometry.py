"""
Geometry helpers for vector parameterization, angle wrapping, and local frame conversions.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from .datatypes import Tensor, Vect, as_Tensor, as_Vect


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
    """Return whether ``transform`` is the canonical identity-transform sentinel."""
    return transform is GRID_TRANSFORM_IDENTITY


def as_grid_transform(transform, name="grid_transform"):
    """Validate a grid transform while preserving the identity sentinel."""
    if is_grid_transform_identity(transform):
        return transform
    return as_Tensor(transform, (3, 3), name=name)


def apply_grid_transform(
    points,
    transform=GRID_TRANSFORM_IDENTITY,
    offset=None,
    *,
    is_inv: bool = False,
):
    """
    Apply the repository grid transform convention, or its inverse.

    Forward convention:
        ``points_physical = points_index @ transform + offset``

    Inverse convention:
        ``points_index = (points_physical - offset) @ inv(transform)``

    The identity sentinel skips matrix multiplication while still respecting
    the translation offset.
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


def calc_vec_from_azimuth_polar(azimuth, polar_angle):
    x = np.sin(polar_angle) * np.cos(azimuth)
    y = np.sin(polar_angle) * np.sin(azimuth)
    z = np.cos(polar_angle)
    return np.array((x, y, z), dtype=float)


def get_azimuth(vec):
    vec = np.asarray(vec, dtype=float)
    az_rad = np.arctan2(vec[1], vec[0])
    return np.degrees(az_rad) % 360


def get_polar_angle(vec):
    vec = np.asarray(vec, dtype=float).copy()
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)
    polar = np.arccos(vec[2])
    return np.degrees(polar)


def get_axis1_azimuth(axis1, normal):
    """
    Measure the azimuth of `axis1` in the local plane orthogonal to `normal`.

    The local reference frame is built by rotating the global z-axis onto
    `normal`, then taking the rotated global x/y axes as the in-plane basis.
    The returned angle is in degrees within ``[0, 360)``.
    """
    axis1 = np.asarray(axis1, dtype=float).copy()
    axis1 /= np.linalg.norm(axis1, axis=-1, keepdims=True)
    rotation = rotation_matrix_from_vectors((0, 0, 1), normal)
    axisx = rotation @ np.array([1.0, 0.0, 0.0])
    axisy = rotation @ np.array([0.0, 1.0, 0.0])
    az_rad = np.arctan2(axis1 @ axisy, axis1 @ axisx)
    return np.degrees(az_rad) % 360


def wrap_to_pi(angle):
    """Wrap angles in radians into the half-open interval ``[-pi, pi)``."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def rotation_matrix_from_vectors(
    source_vector: Vect(3), target_vector: Vect(3)
) -> Tensor((3, 3)):
    """
    Construct a rotation matrix that rotates one vector to another.

    This function computes a 3x3 rotation matrix `R` such that:
        R @ source_vector ~= target_vector

    It internally uses SciPy's `Rotation.align_vectors` to find the
    minimal rotation that maps the source direction to the target
    direction.
    """
    source_vector = as_Vect(
        source_vector,
        name="The vector used as the starting source when constructing the rotation matrix",
        is_norm=True,
    )
    target_vector = as_Vect(
        target_vector,
        name="The vector used as the ending target when constructing the rotation matrix",
        is_norm=True,
    )

    rot, _ = R.align_vectors([target_vector], [source_vector])
    return rot.as_matrix()


def find_rotation_axis(directors, is_return_metric=False):
    """
    Find the common rotation axis for a sequence of normalized vectors.
    """
    M = np.dot(directors.T, directors)
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    axis = eigenvectors[:, 0]

    cross_prods = np.cross(directors[:-1], directors[1:])
    avg_cross = np.sum(cross_prods, axis=0)

    if np.dot(axis, avg_cross) < 0:
        axis = -axis

    if not is_return_metric:
        return axis

    total_var = np.sum(eigenvalues)
    orthogonality_score = 1.0 - (eigenvalues[0] / total_var)
    rms_sin_theta = np.sqrt(eigenvalues[0] / len(directors))
    tilt_angle_deg = np.degrees(np.arcsin(np.clip(rms_sin_theta, -1.0, 1.0)))

    signed_rotation_steps = np.dot(cross_prods, axis)
    total_signed_rotation = np.sum(signed_rotation_steps)
    total_rotation_magnitude = np.sum(np.abs(signed_rotation_steps))
    rotation_consistency = (
        np.abs(total_signed_rotation) / total_rotation_magnitude
        if total_rotation_magnitude > 1e-12
        else 0.0
    )

    metric = {
        "orthogonality_score": orthogonality_score,
        "rms_sin_theta": rms_sin_theta,
        "tilt_angle_degrees": tilt_angle_deg,
        "rotation_consistency": rotation_consistency,
        "eigenvalues": eigenvalues,
    }
    return axis, metric


def find_plane_normal(points, is_return_metric=False):
    """
    Estimate the normal vector of a point cloud and evaluate its planarity.
    """
    if len(points) < 3:
        raise ValueError("At least 3 points are required to define a plane.")

    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    M = np.dot(centered_points.T, centered_points)
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    normal = eigenvectors[:, 0]

    if not is_return_metric:
        return normal

    total_variance = np.sum(eigenvalues)
    planarity = (
        1.0 - (3.0 * eigenvalues[0] / total_variance) if total_variance > 0 else 1.0
    )
    thickness_rms = np.sqrt(eigenvalues[0] / len(points))
    linearity_risk = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] > 1e-9 else 1.0

    metric = {
        "centroid": centroid,
        "planarity_score": np.clip(planarity, 0, 1),
        "thickness_rms": thickness_rms,
        "eigenvalues": eigenvalues,
        "linearity_risk": linearity_risk,
    }
    return normal, metric
