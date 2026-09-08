"""Compatibility import for the migrated PolyData visual module."""

from nematics3d.geometry.polydata import as_polydata_input
from nematics3d.visual.plot_polydata import OptsPolyData, PlotPolyData

__all__ = ["OptsPolyData", "PlotPolyData", "as_polydata_input"]
