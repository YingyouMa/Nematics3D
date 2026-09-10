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

from nematics3d.surface.contour import ContourSurface, ContourSurfaceSet  # noqa: E402


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

    def test_init_is_extract_triggers_extract_all(self):
        with mock.patch.object(
            ContourSurfaceSet,
            "act_extract_all",
            autospec=True,
            return_value=(),
        ) as extract_all:
            contour = ContourSurfaceSet(
                np.zeros((2, 2, 2), dtype=float),
                levels=(0.1,),
                is_extract=True,
            )
        extract_all.assert_called_once_with(contour)

    def test_extract_surface_builds_real_polydata_and_records_level(self):
        x = np.arange(3, dtype=float)[:, None, None]
        values = np.broadcast_to(x, (3, 3, 3)).copy()
        contour = ContourSurfaceSet(values, levels=(1.0,))

        mesh = contour.act_extract_surface(0)

        self.assertGreater(mesh.n_points, 0)
        np.testing.assert_allclose(mesh.points[:, 0], 1.0)
        np.testing.assert_allclose(mesh.field_data["contour_level"], [1.0])

    def test_periodic_x_adds_isosurface_across_high_low_seam(self):
        values = np.empty((2, 3, 3), dtype=float)
        values[0, :, :] = 1.0
        values[1, :, :] = -1.0

        nonperiodic = ContourSurfaceSet(
            values,
            levels=(0.0,),
            box_periodic_flag=(False, False, False),
        )
        periodic_x = ContourSurfaceSet(
            values,
            levels=(0.0,),
            box_periodic_flag=(True, False, False),
        )

        mesh_nonperiodic = nonperiodic.act_extract_surface(0)
        mesh_periodic = periodic_x.act_extract_surface(0)

        x_nonperiodic = np.unique(np.round(mesh_nonperiodic.points[:, 0], 12))
        x_periodic = np.unique(np.round(mesh_periodic.points[:, 0], 12))

        np.testing.assert_allclose(x_nonperiodic, [0.5])
        np.testing.assert_allclose(x_periodic, [0.5, 1.5])

    def test_periodic_extension_respects_selected_axes_only(self):
        values = np.empty((3, 2, 3), dtype=float)
        values[:, 0, :] = 1.0
        values[:, 1, :] = -1.0

        contour = ContourSurfaceSet(
            values,
            levels=(0.0,),
            box_periodic_flag=(False, True, False),
        )
        mesh = contour.act_extract_surface(0)

        y_coords = np.unique(np.round(mesh.points[:, 1], 12))
        x_coords = np.unique(np.round(mesh.points[:, 0], 12))

        np.testing.assert_allclose(y_coords, [0.5, 1.5])
        self.assertLessEqual(float(np.max(x_coords)), 2.0)

    def test_surface_plot_binds_single_visual_relation(self):
        contour = ContourSurfaceSet(np.zeros((2, 2, 2), dtype=float), levels=(0.1,))
        surface = contour[0]
        visual = mock.Mock(name="visual")
        with mock.patch(
            "nematics3d.visual.plot_contour_surface.PlotContourSurface",
            return_value=visual,
        ):
            returned = surface.act_plot()
        self.assertIs(returned, visual)
        self.assertIs(surface.visual, visual)

    def test_surface_plot_replaces_existing_visual(self):
        contour = ContourSurfaceSet(np.zeros((2, 2, 2), dtype=float), levels=(0.1,))
        surface = contour[0]
        visual_old = mock.Mock(name="visual_old")
        visual_old.fig = mock.Mock(is_alive=True)
        visual_new = mock.Mock(name="visual_new")
        surface.act_bind_relation_base("visual", visual_old, is_weak=False)
        with mock.patch(
            "nematics3d.visual.plot_contour_surface.PlotContourSurface",
            return_value=visual_new,
        ):
            returned = surface.act_plot(is_replace=True)
        visual_old.act_remove.assert_called_once_with()
        self.assertIs(returned, visual_new)
        self.assertIs(surface.visual, visual_new)

    def test_surface_plot_rejects_existing_live_visual_without_replace(self):
        contour = ContourSurfaceSet(np.zeros((2, 2, 2), dtype=float), levels=(0.1,))
        surface = contour[0]
        visual_old = mock.Mock(name="visual_old")
        visual_old.fig = mock.Mock(is_alive=True)
        surface.act_bind_relation_base("visual", visual_old, is_weak=False)
        with self.assertRaises(RuntimeError):
            surface.act_plot()

    def test_surface_plot_clears_stale_visual_without_replace(self):
        contour = ContourSurfaceSet(np.zeros((2, 2, 2), dtype=float), levels=(0.1,))
        surface = contour[0]
        visual_old = mock.Mock(name="visual_old")
        visual_old.fig = None
        visual_new = mock.Mock(name="visual_new")
        surface.act_bind_relation_base("visual", visual_old, is_weak=False)
        with mock.patch(
            "nematics3d.visual.plot_contour_surface.PlotContourSurface",
            return_value=visual_new,
        ):
            returned = surface.act_plot()
        visual_old.act_remove.assert_not_called()
        self.assertIs(returned, visual_new)
        self.assertIs(surface.visual, visual_new)

    def test_remove_surface_unbinds_owner_relation(self):
        contour = ContourSurfaceSet(np.zeros((2, 2, 2), dtype=float), levels=(0.1, 0.2))
        surface = contour[0]
        removed = contour.act_remove_surface(0)
        self.assertIs(removed, surface)
        self.assertIsNone(surface.owner)
        self.assertEqual(len(contour), 1)

    def test_remove_surface_removes_live_visual(self):
        contour = ContourSurfaceSet(np.zeros((2, 2, 2), dtype=float), levels=(0.1,))
        surface = contour[0]
        visual = mock.Mock(name="visual")
        visual.fig = mock.Mock(is_alive=True)
        surface.act_bind_relation_base("visual", visual, is_weak=False)
        contour.act_remove_surface(0)
        visual.act_remove.assert_called_once_with()
        self.assertIsNone(surface.visual)
        self.assertIsNone(surface.owner)
