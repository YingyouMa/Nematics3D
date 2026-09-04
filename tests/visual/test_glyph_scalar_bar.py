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


class TestGlyphScalarBar(unittest.TestCase):
    def test_scalar_painting_creates_registered_and_backend_scalar_bar(self):
        figure = PlotFigure(is_off_screen=True, name="scalar_bar_sync_test")
        try:
            sphere = PlotSphere(
                coords=np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ]
                ),
                figure=figure,
                paint_by="scalars",
                scalars=np.array([0.0, 0.5, 1.0]),
                scalar_bar_title="test scalars",
            )

            self.assertEqual(len(figure.scalar_bars), 1)
            scalar_bar = figure.scalar_bars[0]
            self.assertIs(scalar_bar.source, sphere)
            self.assertIsNotNone(scalar_bar.backend)
            self.assertIn(scalar_bar.impl_name_pv, figure.pl.scalar_bars)
            self.assertIs(
                figure.pl.scalar_bars[scalar_bar.impl_name_pv],
                scalar_bar.backend,
            )
            self.assertEqual(len(figure.pl.scalar_bars), 1)
        finally:
            figure.act_close()


if __name__ == "__main__":
    unittest.main()
