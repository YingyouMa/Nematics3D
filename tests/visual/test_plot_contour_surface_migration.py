from nematics3d.classes.visual.plot_contour_surface import (
    OptsContourSurface as LegacyOptsContourSurface,
)
from nematics3d.classes.visual.plot_contour_surface import (
    PlotContourSurface as LegacyPlotContourSurface,
)
from nematics3d.classes.visual.qt.interact_contour_surface import (
    InteractContourSurface as LegacyInteractContourSurface,
)
from nematics3d.visual.plot_contour_surface import (
    OptsContourSurface,
    PlotContourSurface,
)
from nematics3d.visual.qt.interact_contour_surface import InteractContourSurface


def test_contour_visual_legacy_imports_alias_canonical_classes():
    assert LegacyOptsContourSurface is OptsContourSurface
    assert LegacyPlotContourSurface is PlotContourSurface
    assert LegacyInteractContourSurface is InteractContourSurface
