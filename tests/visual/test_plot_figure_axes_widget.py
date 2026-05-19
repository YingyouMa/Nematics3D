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
from nematics3d.field import n_color_immerse


class FakeOrientationWidget:
    def __init__(self):
        self.enabled_off_count = 0

    def EnabledOff(self):
        self.enabled_off_count += 1


class FakePlotter:
    def __init__(self):
        self.calls = []

    def add_orientation_widget(self, actor, interactive=None, viewport=None):
        self.calls.append(
            {
                "actor": actor,
                "interactive": interactive,
                "viewport": viewport,
            }
        )
        return FakeOrientationWidget()


class TestPlotFigureAxesWidget(unittest.TestCase):
    def test_axes_actor_defaults_to_director_axis_colors(self):
        actor = PlotFigure._helper_build_axes_widget_actor()
        expected = n_color_immerse(np.eye(3))

        np.testing.assert_allclose(actor.GetXAxisShaftProperty().GetColor(), expected[0])
        np.testing.assert_allclose(actor.GetYAxisShaftProperty().GetColor(), expected[1])
        np.testing.assert_allclose(actor.GetZAxisShaftProperty().GetColor(), expected[2])
        np.testing.assert_allclose(actor.GetXAxisTipProperty().GetColor(), expected[0])
        np.testing.assert_allclose(actor.GetYAxisTipProperty().GetColor(), expected[1])
        np.testing.assert_allclose(actor.GetZAxisTipProperty().GetColor(), expected[2])
        np.testing.assert_allclose(
            actor.GetXAxisCaptionActor2D().GetCaptionTextProperty().GetColor(),
            expected[0],
        )
        np.testing.assert_allclose(
            actor.GetYAxisCaptionActor2D().GetCaptionTextProperty().GetColor(),
            expected[1],
        )
        np.testing.assert_allclose(
            actor.GetZAxisCaptionActor2D().GetCaptionTextProperty().GetColor(),
            expected[2],
        )
        self.assertEqual(actor.GetXAxisLabelText(), "x")
        self.assertEqual(actor.GetYAxisLabelText(), "y")
        self.assertEqual(actor.GetZAxisLabelText(), "z")

    def test_act_add_axes_widget_uses_plotter_orientation_widget_api(self):
        figure = object.__new__(PlotFigure)
        plotter = FakePlotter()
        object.__setattr__(figure, "entity_plotter", plotter)
        object.__setattr__(figure, "entity_axes_actor", None)
        object.__setattr__(figure, "entity_axes_widget", None)

        widget = figure.act_add_axes_widget(
            interactive=True,
            viewport=(0.1, 0.2, 0.3, 0.4),
        )

        self.assertIs(widget, figure.entity_axes_widget)
        self.assertIs(figure.entity_axes_actor, plotter.calls[0]["actor"])
        self.assertTrue(plotter.calls[0]["interactive"])
        self.assertEqual(plotter.calls[0]["viewport"], (0.1, 0.2, 0.3, 0.4))

    def test_act_add_axes_widget_replaces_previous_widget(self):
        figure = object.__new__(PlotFigure)
        plotter = FakePlotter()
        previous_widget = FakeOrientationWidget()
        object.__setattr__(figure, "entity_plotter", plotter)
        object.__setattr__(figure, "entity_axes_actor", object())
        object.__setattr__(figure, "entity_axes_widget", previous_widget)

        figure.act_add_axes_widget()

        self.assertEqual(previous_widget.enabled_off_count, 1)
        self.assertIsNotNone(figure.entity_axes_widget)
        self.assertEqual(len(plotter.calls), 1)


if __name__ == "__main__":
    unittest.main()
