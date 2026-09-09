"""Compatibility imports for the migrated Bounds host utilities."""

from nematics3d.analysis.bounds import (
    Bounds,
    BoundsData,
    OptsBounds,
    as_bounds,
    bounds_expanded,
    bounds_minimal_wrapping_points,
    bounds_sample_points,
    obb_bounds_from_fit,
)

__all__ = [
    "Bounds",
    "BoundsData",
    "OptsBounds",
    "as_bounds",
    "bounds_expanded",
    "bounds_minimal_wrapping_points",
    "bounds_sample_points",
    "obb_bounds_from_fit",
]
