"""Conversions and wrapping utilities for three-dimensional angles."""

import numpy as np

from ..datatypes import as_vector
from .rotation import rotation_matrix_from_vectors


def _as_finite_angles(values, *, name: str) -> np.ndarray:
    """Return finite real angle values as a floating-point array."""
    raw_values = np.asarray(values)
    if raw_values.dtype.kind not in "iuf":
        raise TypeError(f"`{name}` must contain only real numbers.")
    try:
        angles = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"`{name}` must contain only real numbers.") from exc
    if not np.isfinite(angles).all():
        raise ValueError(f"`{name}` must contain only finite values.")
    return angles


def _as_nonzero_vectors(vectors, *, name: str) -> np.ndarray:
    """Return finite non-zero 3D vectors with coordinates on the final axis."""
    raw_vectors = np.asarray(vectors)
    if raw_vectors.ndim == 0 or raw_vectors.shape[-1] != 3:
        raise ValueError(
            f"`{name}` must have shape (3,) or (..., 3). Got {raw_vectors.shape}."
        )
    if np.issubdtype(raw_vectors.dtype, np.bool_) or np.iscomplexobj(raw_vectors):
        raise TypeError(f"`{name}` must contain only real numbers.")
    try:
        vectors = np.asarray(vectors, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"`{name}` must contain only real numbers.") from exc
    if not np.isfinite(vectors).all():
        raise ValueError(f"`{name}` must contain only finite values.")
    if np.any(np.linalg.norm(vectors, axis=-1) <= 1e-12):
        raise ValueError(f"`{name}` must not contain zero vectors.")
    return vectors


def vector_from_spherical_angles(azimuth, polar_angle) -> np.ndarray:
    """Return unit vectors from spherical angles in radians.

    ``azimuth`` is measured from positive x toward positive y. ``polar_angle``
    is measured from positive z. The inputs may be scalars or broadcastable
    arrays; the returned coordinate axis is always last, with shape ``(..., 3)``.

    Both inputs must contain finite real numeric data. Scalar inputs return an
    array of shape ``(3,)``; broadcast array inputs return a floating-point
    array whose leading shape is the broadcast shape. The inputs are never
    modified.

    Raises
    ------
    TypeError
        If either input does not contain real numeric values.
    ValueError
        If either input contains a non-finite value or the two inputs cannot be
        broadcast together.
    """
    azimuth = _as_finite_angles(azimuth, name="azimuth")
    polar_angle = _as_finite_angles(polar_angle, name="polar_angle")
    try:
        azimuth, polar_angle = np.broadcast_arrays(azimuth, polar_angle)
    except ValueError as exc:
        raise ValueError("`azimuth` and `polar_angle` must be broadcastable.") from exc

    sin_polar = np.sin(polar_angle)
    return np.stack(
        (
            sin_polar * np.cos(azimuth),
            sin_polar * np.sin(azimuth),
            np.cos(polar_angle),
        ),
        axis=-1,
    )


def azimuth_from_vector(vector):
    """Return vector azimuths in radians within ``[0, 2*pi)``.

    ``vector`` may have shape ``(3,)`` or ``(..., 3)``. At either pole the
    azimuth is geometrically undefined and this function returns zero by
    convention. Magnitude does not affect the result. A single vector returns
    a Python ``float``; batched vectors return a floating-point ``ndarray`` with
    the leading input shape. The input is never modified.

    Raises
    ------
    TypeError
        If ``vector`` does not contain real numeric values.
    ValueError
        If the final axis does not have length three, or if any vector is zero
        or contains a non-finite value.
    """
    vector = _as_nonzero_vectors(vector, name="vector")
    is_single_vector = vector.ndim == 1
    azimuth = np.mod(np.arctan2(vector[..., 1], vector[..., 0]), 2.0 * np.pi)
    is_pole = np.hypot(vector[..., 0], vector[..., 1]) <= 1e-12
    azimuth = np.where(is_pole, 0.0, azimuth)
    return float(azimuth) if is_single_vector else azimuth


def polar_angle_from_vector(vector):
    """Return vector polar angles in radians within ``[0, pi]``.

    ``vector`` may have shape ``(3,)`` or ``(..., 3)``. Magnitude does not
    affect the returned angle. A single vector returns a Python ``float``;
    batched vectors return a floating-point ``ndarray`` with the leading input
    shape. The input is never modified.

    Raises
    ------
    TypeError
        If ``vector`` does not contain real numeric values.
    ValueError
        If the final axis does not have length three, or if any vector is zero
        or contains a non-finite value.
    """
    vector = _as_nonzero_vectors(vector, name="vector")
    is_single_vector = vector.ndim == 1
    norm = np.linalg.norm(vector, axis=-1)
    cosine = np.clip(vector[..., 2] / norm, -1.0, 1.0)
    polar_angle = np.arccos(cosine)
    return float(polar_angle) if is_single_vector else polar_angle


def plane_azimuth_from_direction(direction, normal) -> float:
    """Return one direction's azimuth in a plane, in radians within ``[0, 2*pi)``.

    The local reference frame is built by rotating the global z-axis onto the
    plane normal. The direction is projected into that plane before its angle
    is measured. A direction parallel to the normal has no in-plane azimuth
    and is rejected. Both inputs are three-component finite real vectors;
    ``normal`` must already be normalized. The returned value is a Python
    ``float`` and neither input is modified.

    Raises
    ------
    TypeError
        If an input does not contain real numeric values.
    ValueError
        If an input has the wrong shape, is invalid under the vector contract,
        or ``direction`` is parallel to ``normal``.
    """
    direction = as_vector(
        direction,
        name="in-plane direction",
        d=3,
        is_zero_allowed=False,
    )
    normal = as_vector(normal, name="plane normal", d=3, is_normalized=True)

    direction = direction - np.dot(direction, normal) * normal
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        raise ValueError("`direction` must not be parallel to `normal`.")
    direction /= direction_norm

    rotation = rotation_matrix_from_vectors((0.0, 0.0, 1.0), normal)
    axis_x = rotation @ np.array([1.0, 0.0, 0.0])
    axis_y = rotation @ np.array([0.0, 1.0, 0.0])
    angle = np.arctan2(np.dot(direction, axis_y), np.dot(direction, axis_x))
    return float(angle % (2.0 * np.pi))


def wrap_angle_to_pi(angle):
    """Wrap finite real angles in radians into ``[-pi, pi)``.

    A scalar input returns a Python ``float``. An array-like input returns a
    floating-point ``ndarray`` with the same shape. The input is never modified.

    Raises
    ------
    TypeError
        If ``angle`` does not contain real numeric values.
    ValueError
        If ``angle`` contains a non-finite value.
    """
    angle = _as_finite_angles(angle, name="angle")
    is_scalar = angle.ndim == 0
    wrapped = (angle + np.pi) % (2.0 * np.pi) - np.pi
    return float(wrapped) if is_scalar else wrapped
