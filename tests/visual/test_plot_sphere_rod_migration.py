from nematics3d.classes.visual.plot_rod import OptsRod as LegacyOptsRod
from nematics3d.classes.visual.plot_rod import PlotRod as LegacyPlotRod
from nematics3d.classes.visual.plot_sphere import OptsSphere as LegacyOptsSphere
from nematics3d.classes.visual.plot_sphere import PlotSphere as LegacyPlotSphere
from nematics3d.visual.plot_rod import OptsRod, PlotRod
from nematics3d.visual.plot_sphere import OptsSphere, PlotSphere


def test_sphere_legacy_imports_alias_canonical_classes():
    assert LegacyOptsSphere is OptsSphere
    assert LegacyPlotSphere is PlotSphere


def test_rod_legacy_imports_alias_canonical_classes():
    assert LegacyOptsRod is OptsRod
    assert LegacyPlotRod is PlotRod
