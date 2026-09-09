"""Canonical visualization package for Nematics3D.

Public names are imported lazily so low-level helpers can be used without
loading the complete Plot/Qt dependency graph.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "camera_pose_from_vectors": (".camera", "camera_pose_from_vectors"),
    "camera_vectors_from_pose": (".camera", "camera_vectors_from_pose"),
    "n_color_immerse": (".color", "n_color_immerse"),
    "FigureManager": (".figure_manager", "FigureManager"),
    "OptsGlyph": (".glyph", "OptsGlyph"),
    "PlotGlyph": (".glyph", "PlotGlyph"),
    "OptsPickManager": (".pick_manager", "OptsPickManager"),
    "PickManager": (".pick_manager", "PickManager"),
    "OptsDefectLinePlot": (".disclination_line", "OptsDefectLinePlot"),
    "DisclinationLineSmoothPlot": (
        ".disclination_line",
        "DisclinationLineSmoothPlot",
    ),
    "FigureData": (".plot_figure", "FigureData"),
    "OptsFigure": (".plot_figure", "OptsFigure"),
    "PlotFigure": (".plot_figure", "PlotFigure"),
    "as_PlotFigure": (".plot_figure", "as_PlotFigure"),
    "as_plotfigure": (".plot_figure", "as_plotfigure"),
    "OptsDelaunay": (".plot_delaunay", "OptsDelaunay"),
    "PlotDelaunay": (".plot_delaunay", "PlotDelaunay"),
    "OptsContourSurface": (".plot_contour_surface", "OptsContourSurface"),
    "PlotContourSurface": (".plot_contour_surface", "PlotContourSurface"),
    "PlotExtent": (".plot_extent", "PlotExtent"),
    "OptsPolyData": (".plot_polydata", "OptsPolyData"),
    "PlotPolyData": (".plot_polydata", "PlotPolyData"),
    "OptsRod": (".plot_rod", "OptsRod"),
    "PlotRod": (".plot_rod", "PlotRod"),
    "OptsSphere": (".plot_sphere", "OptsSphere"),
    "PlotSphere": (".plot_sphere", "PlotSphere"),
    "OptsTube": (".plot_tube", "OptsTube"),
    "PlotTube": (".plot_tube", "PlotTube"),
    "OptsVector": (".plot_vector", "OptsVector"),
    "PlotVector": (".plot_vector", "PlotVector"),
    "OptsScalarBar": (".scalar_bar", "OptsScalarBar"),
    "ScalarBar": (".scalar_bar", "ScalarBar"),
    "ScalarBarRegistry": (".scalar_bar_registry", "ScalarBarRegistry"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
