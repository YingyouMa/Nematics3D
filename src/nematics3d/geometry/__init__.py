"""Geometry helpers."""

from .angles import (
    azimuth_from_vector,
    plane_azimuth_from_direction,
    polar_angle_from_vector,
    vector_from_spherical_angles,
    wrap_angle_to_pi,
)
from .box import get_box_corners, select_points_in_box
from .misc import *
from .nearest import closest_point_on_polyline, find_nearest_point
from .plane import PlaneNormalResult, find_plane_normal
from .points import points_membership_mask
from .polydata import (
    as_polydata_input,
    copy_polydata_geometry,
)
from .rotation import (
    RotationAxisResult,
    find_rotation_axis,
    rotation_matrix_from_vectors,
)
from .smoothing import *
from .triangulation import *

__all__ = [name for name in globals() if not name.startswith("_")]
