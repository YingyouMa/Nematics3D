"""Grid, coordinate-transform, and periodic-boundary utilities."""

from .coordinate import generate_coordinate_grid, generate_fixed_step_grid
from .periodic import shift_to_box, unwrap_trajectory, wrap_points_to_box
from .plane import resolve_plane_physical_axes
from .transform import (
    GRID_TRANSFORM_IDENTITY,
    GridTransform,
    apply_linear_transform,
    as_grid_offset,
    as_grid_transform,
    is_grid_transform_identity,
)

# Reserved name of the per-voxel validity mask field on a GridFieldDataset.
VALIDITY_FIELD_NAME = "mask"

__all__ = [
    "GRID_TRANSFORM_IDENTITY",
    "GridTransform",
    "VALIDITY_FIELD_NAME",
    "apply_linear_transform",
    "as_grid_offset",
    "as_grid_transform",
    "generate_coordinate_grid",
    "generate_fixed_step_grid",
    "is_grid_transform_identity",
    "resolve_plane_physical_axes",
    "shift_to_box",
    "unwrap_trajectory",
    "wrap_points_to_box",
]
