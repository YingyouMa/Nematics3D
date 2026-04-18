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

from nematics3d.classes.visual.glyph import OptsGlyph, PlotGlyph


class MinimalGlyph(PlotGlyph):
    def __init__(self, coords, resolver_source="u_percent"):
        object.__setattr__(self, "raw_coords", np.asarray(coords, dtype=float))
        object.__setattr__(self, "opts", OptsGlyph(resolver_source=resolver_source))
        self.opts.act_finalize()


class TestGlyphResolverSource(unittest.TestCase):
    def test_u_percent_source_includes_both_endpoints(self):
        glyph = MinimalGlyph(np.zeros((5, 3)))

        np.testing.assert_allclose(
            glyph._helper_get_resolver_source(),
            np.array([0.0, 25.0, 50.0, 75.0, 100.0], dtype=np.float32),
        )

    def test_u_percent_source_handles_empty_and_single_point_inputs(self):
        empty = MinimalGlyph(np.empty((0, 3)))
        single = MinimalGlyph(np.zeros((1, 3)))

        self.assertEqual(empty._helper_get_resolver_source().shape, (0,))
        np.testing.assert_allclose(
            single._helper_get_resolver_source(),
            np.array([0.0], dtype=np.float32),
        )

    def test_coords_source_still_returns_raw_coords(self):
        coords = np.arange(6, dtype=float).reshape(2, 3)
        glyph = MinimalGlyph(coords, resolver_source="coords")

        np.testing.assert_allclose(glyph._helper_get_resolver_source(), coords)


if __name__ == "__main__":
    unittest.main()
