"""Visualization subpackage for nematics3d."""

from .color import (
    blue_red_in_white_bg,
    director_color_pareto_034,
    director_color_pareto_oklab_043,
    plot_director_color_sphere,
)
from .plot_contour_surface import OptsContourSurface, PlotContourSurface
from .plot_delaunay import OptsDelaunay, PlotDelaunay
from .plot_polydata import (
    OptsPolyData,
    PlotPolyData,
    as_polydata_input,
)
from .plot_vector import OptsVector, PlotVector
from .scalar_bar import OptsScalarBar, ScalarBar
from .scalar_bar_registry import ScalarBarRegistry

__all__ = [
    "blue_red_in_white_bg",
    "director_color_pareto_034",
    "director_color_pareto_oklab_043",
    "plot_director_color_sphere",
    "OptsContourSurface",
    "PlotContourSurface",
    "OptsDelaunay",
    "PlotDelaunay",
    "OptsPolyData",
    "PlotPolyData",
    "as_polydata_input",
    "OptsVector",
    "PlotVector",
    "OptsScalarBar",
    "ScalarBar",
    "ScalarBarRegistry",
]
