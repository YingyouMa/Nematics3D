"""Structured domain classes exposed by the ``nematics3d.classes`` package."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "ContourSurface",
    "ContourSurfaceSet",
    "OptsSurfaceSampling",
    "SurfaceSampling",
]


if TYPE_CHECKING:
    from .contour_surface import ContourSurface, ContourSurfaceSet
    from .surface_sampling import OptsSurfaceSampling, SurfaceSampling


def __getattr__(name: str):
    if name in {"ContourSurface", "ContourSurfaceSet"}:
        from .contour_surface import ContourSurface, ContourSurfaceSet

        exports = {
            "ContourSurface": ContourSurface,
            "ContourSurfaceSet": ContourSurfaceSet,
        }
        return exports[name]
    if name in {"OptsSurfaceSampling", "SurfaceSampling"}:
        from .surface_sampling import OptsSurfaceSampling, SurfaceSampling

        exports = {
            "OptsSurfaceSampling": OptsSurfaceSampling,
            "SurfaceSampling": SurfaceSampling,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
