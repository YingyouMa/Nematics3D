"""Visualization subpackage for nematics3d."""

from .plot_contour_surface import OptsContourSurface, PlotContourSurface
from .plot_vector import OptsVector, PlotVector
from .scalar_bar import OptsScalarBar, ScalarBar
from .scalar_bar_registry import ScalarBarRegistry

__all__ = [
    "OptsContourSurface",
    "PlotContourSurface",
    "OptsVector",
    "PlotVector",
    "OptsScalarBar",
    "ScalarBar",
    "ScalarBarRegistry",
]
