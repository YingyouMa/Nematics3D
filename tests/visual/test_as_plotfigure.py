import pyvista as pv
import pytest

from nematics3d.visual.plot_figure import PlotFigure, as_plotfigure


def test_none_creates_new_plotfigure():
    figure = as_plotfigure(None)
    try:
        assert isinstance(figure, PlotFigure)
        assert figure.is_alive
    finally:
        figure.act_close()


def test_live_plotfigure_is_returned_unchanged():
    figure = PlotFigure(is_off_screen=True)
    try:
        assert as_plotfigure(figure) is figure
    finally:
        figure.act_close()


def test_pyvista_plotter_is_wrapped_without_replacement():
    plotter = pv.Plotter(off_screen=True)
    figure = as_plotfigure(plotter)
    try:
        assert figure.pl is plotter
    finally:
        figure.act_close()


def test_invalid_input_keeps_legacy_recovery_to_new_figure():
    figure = as_plotfigure(object())
    try:
        assert isinstance(figure, PlotFigure)
    finally:
        figure.act_close()


def test_internal_plotfigure_commit_error_is_not_swallowed(monkeypatch):
    figure = PlotFigure(is_off_screen=True)

    def fail_commit(*args, **kwargs):
        raise RuntimeError("internal commit bug")

    monkeypatch.setattr(PlotFigure, "act_commit", fail_commit)
    try:
        with pytest.raises(RuntimeError, match="internal commit bug"):
            as_plotfigure(figure)
    finally:
        figure.act_close()


def test_internal_plotter_wrapping_error_is_not_swallowed(monkeypatch):
    plotter = pv.Plotter(off_screen=True)
    original_init = PlotFigure.__init__

    def fail_init(self, *args, **kwargs):
        if kwargs.get("plotter") is plotter:
            raise RuntimeError("internal wrapping bug")
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(PlotFigure, "__init__", fail_init)
    try:
        with pytest.raises(RuntimeError, match="internal wrapping bug"):
            as_plotfigure(plotter)
    finally:
        plotter.close()
