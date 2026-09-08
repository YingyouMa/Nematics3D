import pyvista as pv

from nematics3d.classes.visual.plot_figure import PlotFigure


class _Panel:
    def __init__(self):
        self.name = "panel"
        self.close_count = 0

    def close(self):
        self.close_count += 1


def test_close_is_idempotent_and_marks_figure_dead():
    figure = PlotFigure(is_off_screen=True)

    assert figure.is_alive
    figure.act_close()
    figure.act_close()

    assert not figure.is_alive


def test_close_adopts_and_closes_wrapped_plotter():
    plotter = pv.Plotter(off_screen=True)
    figure = PlotFigure(plotter=plotter)

    figure.act_close()

    assert plotter._closed
    assert not figure.is_alive


def test_close_interacts_closes_and_unregisters_panels():
    figure = PlotFigure(is_off_screen=True)
    panel = _Panel()
    figure.interacts.act_register(
        panel,
        is_bind_registry_relation=False,
    )

    figure.act_close()

    assert panel.close_count == 1
    assert len(figure.interacts) == 0


def test_close_removes_glyphs_by_default():
    from nematics3d.classes.visual.plot_sphere import PlotSphere

    figure = PlotFigure(is_off_screen=True)
    sphere = PlotSphere([[0.0, 0.0, 0.0]], figure=figure)

    figure.act_close()

    assert len(figure.glyphs) == 0
    assert sphere.fig is None
    assert sphere.registry is None


def test_close_can_preserve_glyph_registry_when_requested():
    from nematics3d.classes.visual.plot_sphere import PlotSphere

    figure = PlotFigure(is_off_screen=True)
    sphere = PlotSphere([[0.0, 0.0, 0.0]], figure=figure)

    figure.act_close(is_remove_glyphs=False)

    assert not figure.is_alive
    assert sphere in figure.glyphs
    assert sphere.fig is figure
    assert sphere.registry is figure.glyphs
