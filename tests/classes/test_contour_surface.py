import sys
from pathlib import Path
import types
import unittest
from unittest import mock

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
PKG_DIR = SRC_DIR / "nematics3d"

sys.path.insert(0, str(SRC_DIR))

if "nematics3d" not in sys.modules:
    pkg = types.ModuleType("nematics3d")
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules["nematics3d"] = pkg

from nematics3d.classes.contour_surface import ContourSurface, ContourSurfaceSet


class TestContourSurface(unittest.TestCase):
    def test_act_add_surface_appends_unique_level(self):
        contour = ContourSurfaceSet(np.zeros((2, 2, 2), dtype=float), levels=(0.1, 0.2))

        surface = contour.act_add_surface(0.3)

        self.assertEqual(len(contour), 3)
        self.assertEqual(contour.calc_levels, (0.1, 0.2, 0.3))
        self.assertIs(surface, contour.act_get_surface_by_level(0.3))

    def test_act_add_surface_rejects_duplicate_level(self):
        contour = ContourSurfaceSet(np.zeros((2, 2, 2), dtype=float), levels=(0.1,))

        with self.assertRaises(ValueError):
            contour.act_add_surface(0.1)

    def test_plot_surface_uses_stored_visual_defaults_when_opts_missing(self):
        contour = ContourSurfaceSet(
            np.zeros((2, 2, 2), dtype=float),
            levels=(0.1,),
            visual_default={"opacity": 0.4, "color": (0.1, 0.2, 0.3)},
        )

        with mock.patch.object(
            ContourSurface,
            "act_plot",
            autospec=True,
            side_effect=lambda self, **kwargs: kwargs,
        ):
            kwargs = contour.act_plot_surface(0, line_width=2.0)

        self.assertEqual(kwargs["opacity"], 0.4)
        self.assertEqual(kwargs["color"], (0.1, 0.2, 0.3))
        self.assertEqual(kwargs["line_width"], 2.0)

    def test_init_is_plot_triggers_plot_all(self):
        with mock.patch.object(
            ContourSurfaceSet,
            "act_plot_all",
            autospec=True,
            return_value=(),
        ) as plot_all:
            contour = ContourSurfaceSet(
                np.zeros((2, 2, 2), dtype=float),
                levels=(0.1,),
                figure="dummy-figure",
                is_plot=True,
            )

        plot_all.assert_called_once_with(contour, figure="dummy-figure")

    def test_surface_plot_binds_single_visual_relation(self):
        contour = ContourSurfaceSet(np.zeros((2, 2, 2), dtype=float), levels=(0.1,))
        surface = contour[0]
        visual = mock.Mock(name="visual")

        with mock.patch(
            "nematics3d.classes.visual.plot_contour_surface.PlotContourSurface",
            return_value=visual,
        ):
            returned = surface.act_plot()

        self.assertIs(returned, visual)
        self.assertIs(surface.visual, visual)

    def test_surface_plot_replaces_existing_visual(self):
        contour = ContourSurfaceSet(np.zeros((2, 2, 2), dtype=float), levels=(0.1,))
        surface = contour[0]
        visual_old = mock.Mock(name="visual_old")
        visual_new = mock.Mock(name="visual_new")
        surface.act_bind_relation_base("visual", visual_old, is_weak=False)

        with mock.patch(
            "nematics3d.classes.visual.plot_contour_surface.PlotContourSurface",
            return_value=visual_new,
        ):
            returned = surface.act_plot()

        visual_old.act_remove.assert_called_once_with()
        self.assertIs(returned, visual_new)
        self.assertIs(surface.visual, visual_new)
