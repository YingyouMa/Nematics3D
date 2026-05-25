"""Structured domain classes exposed by the ``nematics3d.classes`` package."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "ContourSurface",
    "ContourSurfaceSet",
]


if TYPE_CHECKING:
    from .contour_surface import ContourSurface, ContourSurfaceSet


def __getattr__(name: str):
    if name in {"ContourSurface", "ContourSurfaceSet"}:
        from .contour_surface import ContourSurface, ContourSurfaceSet

        exports = {
            "ContourSurface": ContourSurface,
            "ContourSurfaceSet": ContourSurfaceSet,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
