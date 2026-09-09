"""Geometry helpers."""

from .angles import (
    azimuth_from_vector,
    plane_azimuth_from_direction,
    polar_angle_from_vector,
    vector_from_spherical_angles,
    wrap_angle_to_pi,
)
from .box import get_box_corners, select_points_in_box
from .frame import align_axes_to_reference, axes_angle_changes_deg
from .nearest import closest_point_on_polyline, find_nearest_point
from .obb import (
    OBBFit,
    box_corners_from_center_axes_radii,
    canonicalize_axes,
    compute_convex_hull_points,
    obb_fit_approx,
    obb_fit_pca,
    obb_refine_random_search,
)
from .plane import PlaneNormalResult, find_plane_normal
from .points import points_membership_mask
from .polydata import (
    as_polydata_input,
    copy_polydata_geometry,
)
from .rotation import (
    RotationAxisResult,
    find_rotation_axis,
    frame_from_spherical_roll,
    roll_angle_from_frame,
    rotation_matrix_from_vectors,
    rotate_vector_about_axis,
)
from .smoothing import *
from .triangulation import *

__all__ = [name for name in globals() if not name.startswith("_")]
