"""Pure camera-pose conversions used by Nematics3D visual scenes."""

import numpy as np


_CAMERA_EPS = 1e-12


def _base_up_from_view_direction(view_direction):
    """Return the zero-roll up vector for one normalized viewing direction."""
    world_up = np.array([0.0, 0.0, 1.0])
    up = world_up - np.dot(world_up, view_direction) * view_direction
    norm = np.linalg.norm(up)
    if norm <= _CAMERA_EPS:
        # At either z pole azimuth is undefined.  Fix +y as the zero-roll up
        # direction so both conversion directions use the same convention.
        return np.array([0.0, 1.0, 0.0])
    return up / norm


def camera_pose_from_vectors(position, focal_point, view_up):
    """Return spherical camera pose from position, focal point, and up vector.

    Angles are returned in degrees as ``(azimuth, elevation, roll, distance)``.
    Azimuth is measured from +x toward +y, elevation from the xy plane, and
    roll about the viewing direction.  At either z pole azimuth is undefined
    and is returned as zero by convention.

    The camera position must differ from the focal point.  ``view_up`` need not
    be normalized, but it must have a non-zero component perpendicular to the
    viewing direction.
    """
    position = np.asarray(position, dtype=float)
    focal_point = np.asarray(focal_point, dtype=float)
    view_up = np.asarray(view_up, dtype=float)
    for value, name in (
        (position, "position"),
        (focal_point, "focal_point"),
        (view_up, "view_up"),
    ):
        if value.shape != (3,):
            raise ValueError(f"`{name}` must have shape (3,). Got {value.shape}.")
        if not np.isfinite(value).all():
            raise ValueError(f"`{name}` must contain only finite values.")

    offset = position - focal_point
    distance = float(np.linalg.norm(offset))
    if distance <= _CAMERA_EPS:
        raise ValueError("Camera `position` must differ from `focal_point`.")

    elevation = float(np.degrees(np.arcsin(np.clip(offset[2] / distance, -1, 1))))
    horizontal_norm = float(np.hypot(offset[0], offset[1]))
    if horizontal_norm <= _CAMERA_EPS * distance:
        azimuth = 0.0
    else:
        azimuth = float(np.degrees(np.arctan2(offset[1], offset[0])) % 360.0)

    view_direction = -offset / distance
    base_up = _base_up_from_view_direction(view_direction)
    right = np.cross(view_direction, base_up)

    projected_up = view_up - np.dot(view_up, view_direction) * view_direction
    up_norm = float(np.linalg.norm(projected_up))
    if up_norm <= _CAMERA_EPS:
        raise ValueError("`view_up` must not be parallel to the viewing direction.")
    projected_up /= up_norm
    roll = float(
        np.degrees(
            np.arctan2(
                np.dot(projected_up, right),
                np.dot(projected_up, base_up),
            )
        )
    )

    return azimuth, elevation, roll, distance


def camera_vectors_from_pose(azimuth, elevation, roll, distance, focal_point):
    """Return camera position, focal point, and up vector from a spherical pose.

    Angles are interpreted in degrees.  ``distance`` must be finite and
    strictly positive.  At elevation +/-90 degrees azimuth has no geometric
    effect; the zero-roll up direction is +y by convention.
    """
    azimuth = float(azimuth)
    elevation = float(elevation)
    roll = float(roll)
    distance = float(distance)
    if not np.isfinite((azimuth, elevation, roll, distance)).all():
        raise ValueError("Camera pose values must be finite.")
    if distance <= _CAMERA_EPS:
        raise ValueError("Camera `distance` must be strictly positive.")

    focal_point = np.asarray(focal_point, dtype=float)
    if focal_point.shape != (3,):
        raise ValueError(
            f"`focal_point` must have shape (3,). Got {focal_point.shape}."
        )
    if not np.isfinite(focal_point).all():
        raise ValueError("`focal_point` must contain only finite values.")

    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)
    roll_rad = np.radians(roll)
    offset = distance * np.array(
        [
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.cos(elevation_rad) * np.sin(azimuth_rad),
            np.sin(elevation_rad),
        ]
    )
    position = focal_point + offset
    view_direction = -offset / distance
    up = _base_up_from_view_direction(view_direction)

    if abs(roll_rad) > _CAMERA_EPS:
        cosine = np.cos(roll_rad)
        sine = np.sin(roll_rad)
        up = (
            up * cosine
            + np.cross(view_direction, up) * sine
            + view_direction * np.dot(view_direction, up) * (1.0 - cosine)
        )

    return position, focal_point.copy(), up
