import sys
from pathlib import Path
import types
import unittest

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
PKG_DIR = SRC_DIR / "nematics3d"

sys.path.insert(0, str(SRC_DIR))

if "nematics3d" not in sys.modules:
    pkg = types.ModuleType("nematics3d")
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules["nematics3d"] = pkg

from nematics3d.classes.visual.plot_figure import PlotFigure
from nematics3d.classes.visual.plot_sphere import PlotSphere


class TestGlyphEmptyCoords(unittest.TestCase):
    def test_plot_sphere_empty_coords_skips_resolvers_and_stays_empty(self):
        def forbidden_resolver(_coords):
            raise AssertionError("empty glyph initialization should not call resolvers")

        figure = PlotFigure(is_off_screen=True, name="empty_sphere_test")
        sphere = PlotSphere(
            coords=np.empty((0, 3), dtype=float),
            figure=figure,
            color=forbidden_resolver,
            radius=forbidden_resolver,
        )

        self.assertTrue(sphere.calc_is_empty)
        self.assertIsNone(sphere.entity_actor)
        self.assertEqual(sphere.calc_coords.shape, (0, 3))
        self.assertEqual(sphere.calc_color.shape, (0, 3))
        self.assertEqual(sphere.calc_radius.shape, (0,))
        self.assertEqual(sphere.calc_keep_index.shape, (0,))
        self.assertIs(sphere.opts.color, forbidden_resolver)
        self.assertIs(sphere.opts.radius, forbidden_resolver)

    def test_empty_glyph_pick_reports_clear_error(self):
        figure = PlotFigure(is_off_screen=True, name="empty_pick_test")
        sphere = PlotSphere(coords=np.empty((0, 3), dtype=float), figure=figure)

        with self.assertRaisesRegex(RuntimeError, "empty glyph"):
            sphere.act_resolve_pick(np.zeros(3, dtype=float))


if __name__ == "__main__":
    unittest.main()
