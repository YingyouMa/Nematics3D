"""Visualization subpackage for nematics3d."""

from .plot_contour_surface import OptsContourSurface, PlotContourSurface
from .plot_delaunay import OptsDelaunay, PlotDelaunay
from .plot_polydata import OptsPolyData, PlotPolyData
from .plot_vector import OptsVector, PlotVector
from .scalar_bar import OptsScalarBar, ScalarBar
from .scalar_bar_registry import ScalarBarRegistry

__all__ = [
    "OptsContourSurface",
    "PlotContourSurface",
    "OptsDelaunay",
    "PlotDelaunay",
    "OptsPolyData",
    "PlotPolyData",
    "OptsVector",
    "PlotVector",
    "OptsScalarBar",
    "ScalarBar",
    "ScalarBarRegistry",
]
