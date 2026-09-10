from importlib import import_module

from .misc import *
from .q_field import QDiagonalizationResult, get_q, q_diagonalize
from .analysis.sampling import *
from .grid import *
from .analysis.principal_plane import *
from .analysis.disclination import *
from .core import *

# from .elastic import *
# from .coarse import *
from .geometry.smoothing import *
from .surface.contour import *
from .sample import *
from .sample.plane_grid import *
from .sample.plane_grid_polar import *
from .sample.q_plane import *
from .sample.vector_plane import *
from .sample.q_surface import *
from .geometry import *
from .logging_decorator import logging_and_warning_decorator


_LAZY_EXPORTS = {
    "InputQ": (".q_field", "InputQ"),
    "QFieldObject": (".q_field", "QFieldObject"),
    "quick_visualize_q": (".quick", "quick_visualize_q"),
    "qt": (".visual", "qt"),
    "FigureData": (".visual", "FigureData"),
    "OptsFigure": (".visual", "OptsFigure"),
    "PlotFigure": (".visual", "PlotFigure"),
    "as_PlotFigure": (".visual", "as_PlotFigure"),
    "as_plotfigure": (".visual", "as_plotfigure"),
    "OptsDelaunay": (".visual", "OptsDelaunay"),
    "PlotDelaunay": (".visual", "PlotDelaunay"),
    "OptsContourSurface": (".visual", "OptsContourSurface"),
    "PlotContourSurface": (".visual", "PlotContourSurface"),
    "PlotExtent": (".visual", "PlotExtent"),
    "OptsPolyData": (".visual", "OptsPolyData"),
    "PlotPolyData": (".visual", "PlotPolyData"),
    "OptsRod": (".visual", "OptsRod"),
    "PlotRod": (".visual", "PlotRod"),
    "OptsSphere": (".visual", "OptsSphere"),
    "PlotSphere": (".visual", "PlotSphere"),
    "OptsTube": (".visual", "OptsTube"),
    "PlotTube": (".visual", "PlotTube"),
    "OptsVector": (".visual", "OptsVector"),
    "PlotVector": (".visual", "PlotVector"),
}


def __getattr__(name):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value

__version__ = "0.9.0b1"
