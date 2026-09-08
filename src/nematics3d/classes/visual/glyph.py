"""Compatibility import for the migrated glyph base classes."""

from nematics3d.visual.glyph import (
    ColorMode,
    OpacityMode,
    OptsGlyph,
    PlotGlyph,
    RadiusMode,
    ScalarsMode,
    _as_resolver_source_or_none as _as_resolver_source_or_none,
)

__all__ = [
    "ColorMode",
    "OpacityMode",
    "OptsGlyph",
    "PlotGlyph",
    "RadiusMode",
    "ScalarsMode",
]
