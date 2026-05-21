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

from nematics3d.classes.visual import OptsVector as ExportedOptsVector
from nematics3d.classes.visual import PlotVector as ExportedPlotVector
from nematics3d.classes.visual.plot_figure import PlotFigure
from nematics3d.classes.visual.plot_vector import OptsVector, PlotVector


def build_vector_debug_figure(*, is_off_screen=False):
    """Build and return a PlotVector figure for manual debugging."""
    figure = PlotFigure(is_off_screen=is_off_screen)
    PlotVector(
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        figure=figure,
        length=lambda orient_length: orient_length,
        radius=0.05,
        tip_length_fraction=0.25,
        tip_radius_ratio=3.0,
        sides=8,
    )
    return figure


def test_plot_vector_debug_figure():
    """Return a foreground PlotFigure for interactive vector debugging."""
    figure = build_vector_debug_figure(is_off_screen=False)
    return figure


class TestPlotVector(unittest.TestCase):
    foreground_figures = []

    def _make_figure(self, *, is_off_screen=True):
        return PlotFigure(is_off_screen=is_off_screen)

    def _close_if_off_screen(self, figure, *, is_off_screen):
        if is_off_screen:
            figure.act_close()
        else:
            self.foreground_figures.append(figure)

    def test_visual_subpackage_exports_vector_classes(self):
        self.assertIs(ExportedOptsVector, OptsVector)
        self.assertIs(ExportedPlotVector, PlotVector)

    def test_opts_vector_defaults_use_orient_length(self):
        opts = OptsVector()
        opts.act_finalize()

        self.assertEqual(opts.resolver_source, "orient_length")
        self.assertIsNone(opts.resolver_source_color)
        self.assertIsNone(opts.resolver_source_opacity)
        self.assertIsNone(opts.resolver_source_radius)
        self.assertIsNone(opts.resolver_source_scalars)
        self.assertEqual(opts.anchor, "center")
        self.assertAlmostEqual(opts.tip_length_fraction, 0.2)

    def test_vector_attr_specific_resolver_source_accepts_orient_length(self):
        opts = OptsVector(resolver_source_radius="orient_length")
        opts.act_finalize()

        self.assertEqual(opts.resolver_source_radius, "orient_length")

    def test_length_and_radius_resolve_from_orient_length(self):
        fig = self._make_figure()
        try:
            vectors = PlotVector(
                np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
                figure=fig,
                length=lambda orient_length: orient_length,
                radius=lambda orient_length: 0.1 * orient_length,
                tip_length_fraction=0.25,
                tip_radius_ratio=3.0,
                anchor="tail",
                sides=8,
            )

            np.testing.assert_allclose(vectors.calc_length, [1.0, 2.0])
            np.testing.assert_allclose(vectors.calc_radius, [0.1, 0.2])
            np.testing.assert_allclose(vectors.calc_shaft_length, [0.75, 1.5])
            np.testing.assert_allclose(vectors.calc_tip_length, [0.25, 0.5])
            np.testing.assert_allclose(vectors.calc_tip_radius, [0.3, 0.6])
        finally:
            fig.act_close()

    def test_anchor_center_and_tail_key_points(self):
        fig_center = self._make_figure()
        fig_default = self._make_figure()
        fig_tail = self._make_figure()
        try:
            center = PlotVector(
                np.array([[0.0, 0.0, 0.0]]),
                np.array([[2.0, 0.0, 0.0]]),
                figure=fig_center,
                length=lambda orient_length: orient_length,
                tip_length_fraction=0.25,
                anchor="center",
                sides=8,
            )
            default_center = PlotVector(
                np.array([[0.0, 0.0, 0.0]]),
                np.array([[2.0, 0.0, 0.0]]),
                figure=fig_default,
                length=lambda orient_length: orient_length,
                tip_length_fraction=0.25,
                sides=8,
            )
            tail = PlotVector(
                np.array([[1.0, 0.0, 0.0]]),
                np.array([[2.0, 0.0, 0.0]]),
                figure=fig_tail,
                length=lambda orient_length: orient_length,
                tip_length_fraction=0.25,
                anchor="tail",
                sides=8,
            )

            np.testing.assert_allclose(center.calc_tail, [[-1.0, 0.0, 0.0]])
            np.testing.assert_allclose(center.calc_shaft_end, [[0.5, 0.0, 0.0]])
            np.testing.assert_allclose(center.calc_tip_end, [[1.0, 0.0, 0.0]])

            np.testing.assert_allclose(default_center.calc_tail, center.calc_tail)
            np.testing.assert_allclose(
                default_center.calc_shaft_end,
                center.calc_shaft_end,
            )
            np.testing.assert_allclose(default_center.calc_tip_end, center.calc_tip_end)

            np.testing.assert_allclose(tail.calc_tail, [[1.0, 0.0, 0.0]])
            np.testing.assert_allclose(tail.calc_shaft_end, [[2.5, 0.0, 0.0]])
            np.testing.assert_allclose(tail.calc_tip_end, [[3.0, 0.0, 0.0]])
        finally:
            fig_center.act_close()
            fig_default.act_close()
            fig_tail.act_close()

    def test_vector_mesh_is_nonempty_and_has_visual_arrays(self):
        for label, is_off_screen in (
            ("background", True),
            ("foreground", False),
        ):
            with self.subTest(label=label):
                if is_off_screen:
                    fig = self._make_figure(is_off_screen=True)
                    vectors = PlotVector(
                        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                        np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
                        figure=fig,
                        length=lambda orient_length: orient_length,
                        radius=0.05,
                        tip_length_fraction=0.25,
                        tip_radius_ratio=3.0,
                        sides=8,
                    )
                else:
                    fig = test_plot_vector_debug_figure()
                    vectors = next(iter(fig.glyphs))
                try:
                    mesh = vectors.entity_actor.mapper.dataset
                    if is_off_screen:
                        self.assertTrue(fig.pl.off_screen)
                    else:
                        self.assertFalse(fig.pl.off_screen)
                    self.assertGreater(mesh.n_points, 0)
                    self.assertGreater(mesh.n_cells, 0)
                    self.assertIn("rgba", mesh.point_data)
                    self.assertIn("opacity", mesh.point_data)
                    self.assertIn("scalars", mesh.point_data)
                finally:
                    self._close_if_off_screen(fig, is_off_screen=is_off_screen)

    def test_vector_pick_report_includes_vector_lengths(self):
        fig = self._make_figure()
        try:
            vectors = PlotVector(
                np.array([[0.0, 0.0, 0.0]]),
                np.array([[2.0, 0.0, 0.0]]),
                figure=fig,
                length=lambda orient_length: orient_length,
                tip_length_fraction=0.25,
                anchor="center",
                sides=8,
            )

            _, message, idx = vectors._helper_resolve_pick(np.array([0.0, 0.0, 0.0]))

            self.assertEqual(idx, 0)
            self.assertIn("Local orientation:", message)
            self.assertIn("Local orientation length:", message)
            self.assertIn("Local display length:", message)
            self.assertIn("Local shaft length:", message)
            self.assertIn("Local tip length:", message)
        finally:
            fig.act_close()


if __name__ == "__main__":
    figure = test_plot_vector_debug_figure()
    print("Created foreground PlotVector debug figure: figure")
