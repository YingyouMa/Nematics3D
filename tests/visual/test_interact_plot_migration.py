import numpy as np

from nematics3d.classes.visual.qt.interact_delaunay import (
    InteractDelaunay as LegacyInteractDelaunay,
)
from nematics3d.classes.visual.qt.interact_polydata import (
    InteractPolyData as LegacyInteractPolyData,
)
from nematics3d.classes.visual.qt.interact_rod import InteractRod as LegacyInteractRod
from nematics3d.classes.visual.qt.interact_sphere import (
    InteractSphere as LegacyInteractSphere,
)
from nematics3d.classes.visual.qt.interact_tube import (
    InteractTube as LegacyInteractTube,
)
from nematics3d.classes.visual.qt.interact_vector import (
    InteractVector as LegacyInteractVector,
)
from nematics3d.visual.qt.interact_delaunay import InteractDelaunay
from nematics3d.visual.qt.interact_glyph_base import InteractGlyphBase
from nematics3d.visual.qt.interact_polydata import InteractPolyData
from nematics3d.visual.qt.interact_rod import InteractRod
from nematics3d.visual.qt.interact_sphere import InteractSphere
from nematics3d.visual.qt.interact_tube import InteractTube
from nematics3d.visual.qt.interact_vector import InteractVector


def test_processed_plot_interaction_panels_use_canonical_classes():
    assert LegacyInteractDelaunay is InteractDelaunay
    assert LegacyInteractPolyData is InteractPolyData
    assert LegacyInteractRod is InteractRod
    assert LegacyInteractSphere is InteractSphere
    assert LegacyInteractTube is InteractTube
    assert LegacyInteractVector is InteractVector


def test_scale_resolver_value_preserves_resolver_forms():
    assert InteractGlyphBase._helper_scale_resolver_value(2.0, 3.0) == 6.0

    array = np.array([1.0, 2.0])
    np.testing.assert_allclose(
        InteractGlyphBase._helper_scale_resolver_value(array, 0.5),
        [0.5, 1.0],
    )

    def resolver(x):
        return np.asarray(x) + 1.0

    scaled = InteractGlyphBase._helper_scale_resolver_value(resolver, 2.0)
    np.testing.assert_allclose(scaled(np.array([1.0, 3.0])), [4.0, 8.0])
