import pyvista as pv

from nematics3d.visual.plot_figure import PlotFigure


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
    from nematics3d.visual.plot_sphere import PlotSphere

    figure = PlotFigure(is_off_screen=True)
    sphere = PlotSphere([[0.0, 0.0, 0.0]], figure=figure)

    figure.act_close()

    assert len(figure.glyphs) == 0
    assert sphere.fig is None
    assert sphere.registry is None


def test_close_can_preserve_glyph_registry_when_requested():
    from nematics3d.visual.plot_sphere import PlotSphere

    figure = PlotFigure(is_off_screen=True)
    sphere = PlotSphere([[0.0, 0.0, 0.0]], figure=figure)

    figure.act_close(is_remove_glyphs=False)

    assert not figure.is_alive
    assert sphere in figure.glyphs
    assert sphere.fig is figure
    assert sphere.registry is figure.glyphs


def test_close_picking_services_removes_owned_observers_and_is_idempotent():
    removed = []

    class Iren:
        def remove_observer(self, observer_id):
            removed.append(observer_id)

    class Plotter:
        def __init__(self):
            self.iren = Iren()
            self.disable_count = 0

        def disable_picking(self):
            self.disable_count += 1

    figure = object.__new__(PlotFigure)
    plotter = Plotter()
    object.__setattr__(figure, "entity_plotter", plotter)
    object.__setattr__(figure, "impl_interaction_observer_ids", [11, 12])

    figure._helper_close_picking_services()
    figure._helper_close_picking_services()

    assert removed == [11, 12]
    assert plotter.disable_count == 2
    assert figure.impl_interaction_observer_ids == []
