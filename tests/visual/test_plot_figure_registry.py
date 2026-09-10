import numpy as np

from nematics3d.visual.plot_figure import PlotFigure
from nematics3d.visual.plot_sphere import PlotSphere


def _sphere(figure, *, name="sphere"):
    return PlotSphere(
        np.array([[0.0, 0.0, 0.0]]),
        figure=figure,
        radius=0.2,
        sides=8,
        name=name,
    )


def test_figure_register_returns_registry_result_for_existing_glyph():
    figure = PlotFigure(is_off_screen=True)
    try:
        sphere = _sphere(figure)

        result = figure.act_register(sphere, is_contain_ok=True)

        assert result is sphere
        assert len(figure) == 1
    finally:
        figure.act_close()


def test_figure_collection_protocol_delegates_to_glyph_registry():
    figure = PlotFigure(is_off_screen=True)
    try:
        first = _sphere(figure, name="first")
        second = _sphere(figure, name="second")

        assert figure() == (first, second)
        assert len(figure) == 2
        assert list(figure) == [first, second]
        assert first in figure
        assert figure[0] is first
        assert figure[1] is second
        assert figure["first"] is first
        assert figure["second"] is second
        assert figure[None] is None
    finally:
        figure.act_close()


def test_figure_unregister_removes_registry_and_figure_relations():
    figure = PlotFigure(is_off_screen=True)
    try:
        sphere = _sphere(figure)
        assert sphere.fig is figure
        assert sphere.registry is figure.glyphs

        result = figure.act_unregister(sphere)

        assert result is None
        assert sphere not in figure
        assert sphere.fig is None
        assert sphere.registry is None
    finally:
        figure.act_close()


def test_clear_category_removes_only_matching_glyphs_and_returns_them():
    figure = PlotFigure(is_off_screen=True)
    try:
        first = _sphere(figure, name="first")
        second = _sphere(figure, name="second")
        category = first.category

        removed = figure.act_clear_category(category)

        assert removed == [first, second]
        assert len(figure) == 0
        assert first.fig is None
        assert second.fig is None
    finally:
        figure.act_close()


def test_clear_missing_category_follows_explicit_missing_policy():
    figure = PlotFigure(is_off_screen=True)
    try:
        assert figure.act_clear_category("missing", is_missing_ok=True) == []

        try:
            figure.act_clear_category("missing", is_missing_ok=False)
        except KeyError:
            pass
        else:
            raise AssertionError("Expected a missing category to raise KeyError")
    finally:
        figure.act_close()
