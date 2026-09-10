import numpy as np
import pytest

from nematics3d.datatypes import UNSET
from nematics3d.visual.plot_figure import PlotFigure
from nematics3d.visual.plot_sphere import PlotSphere
from nematics3d.visual.scalar_bar import OptsScalarBar, ScalarBar


def _make_scalar_sphere(figure):
    return PlotSphere(
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


def test_scalar_bar_position_validator_accepts_viewport_position():
    opts = OptsScalarBar(position=(0.2, 0.3))
    opts.act_finalize()

    assert opts.position == pytest.approx((0.2, 0.3))


@pytest.mark.parametrize("position", [(-0.1, 0.2), (0.2, 1.1)])
def test_scalar_bar_position_validator_rejects_outside_viewport(position):
    opts = OptsScalarBar(position=position)
    assert opts.position is UNSET


def test_scalar_bar_builds_pyvista_kwargs_from_opts():
    scalar_bar = ScalarBar(
        position=(0.2, 0.3),
        width=0.25,
        height=0.4,
        is_vertical=False,
        is_interactive=False,
    )

    assert scalar_bar.calc_pyvista_kwargs["position_x"] == pytest.approx(0.2)
    assert scalar_bar.calc_pyvista_kwargs["position_y"] == pytest.approx(0.3)
    assert scalar_bar.calc_pyvista_kwargs["width"] == pytest.approx(0.25)
    assert scalar_bar.calc_pyvista_kwargs["height"] == pytest.approx(0.4)
    assert scalar_bar.calc_pyvista_kwargs["vertical"] is False
    assert scalar_bar.calc_pyvista_kwargs["interactive"] is False


def test_scalar_bar_opts_update_syncs_existing_backend_in_place():
    figure = PlotFigure(is_off_screen=True, name="scalar_bar_update_test")
    try:
        _make_scalar_sphere(figure)
        scalar_bar = figure.scalar_bars[0]
        backend_before = scalar_bar.backend

        scalar_bar.act_commit(
            width=0.22,
            height=0.31,
            position=(0.12, 0.18),
            is_visible=False,
        )

        assert scalar_bar.backend is backend_before
        assert scalar_bar.backend.GetWidth() == pytest.approx(0.22)
        assert scalar_bar.backend.GetHeight() == pytest.approx(0.31)
        assert scalar_bar.backend.GetPosition()[:2] == pytest.approx((0.12, 0.18))
        assert scalar_bar.backend.GetVisibility() == 0
    finally:
        figure.act_close()


def test_scalar_bar_interactive_change_rebuilds_backend():
    figure = PlotFigure(is_off_screen=True, name="scalar_bar_rebuild_test")
    try:
        _make_scalar_sphere(figure)
        scalar_bar = figure.scalar_bars[0]
        backend_before = scalar_bar.backend

        scalar_bar.act_commit(is_interactive=True)

        assert scalar_bar.backend is not None
        assert scalar_bar.backend is not backend_before
        assert scalar_bar.impl_backend_rebuild_signature == {"is_interactive": True}
    finally:
        figure.act_close()


def test_unregister_scalar_bar_removes_backend_and_owner_relation():
    figure = PlotFigure(is_off_screen=True, name="scalar_bar_unregister_test")
    try:
        _make_scalar_sphere(figure)
        scalar_bar = figure.scalar_bars[0]
        backend_name = scalar_bar.impl_name_pv

        figure.act_unregister_scalar_bar(scalar_bar)

        assert scalar_bar.backend is None
        assert scalar_bar.backend_widget is None
        assert scalar_bar.owner is None
        assert backend_name not in figure.pl.scalar_bars
        assert len(figure.scalar_bars) == 0
    finally:
        figure.act_close()


def test_scalar_bar_interactive_widget_geometry_writes_back_to_opts():
    figure = PlotFigure(is_off_screen=True, name="scalar_bar_widget_writeback_test")
    try:
        _make_scalar_sphere(figure)
        scalar_bar = figure.scalar_bars[0]

        scalar_bar.act_commit(is_interactive=True)
        widget = scalar_bar.backend_widget
        assert widget is not None

        rep = widget.GetRepresentation()
        rep.SetPosition(0.23, 0.27)
        rep.SetPosition2(0.19, 0.41)
        position_effective = tuple(float(x) for x in rep.GetPosition()[:2])
        size_effective = tuple(float(x) for x in rep.GetPosition2()[:2])

        payload = figure.scalar_bars._helper_pull_scalar_bar_widget_geometry(scalar_bar)

        assert payload["position"] == pytest.approx(position_effective)
        assert payload["width"] == pytest.approx(size_effective[0])
        assert payload["height"] == pytest.approx(size_effective[1])
        assert scalar_bar.opts.position == pytest.approx(position_effective)
        assert scalar_bar.opts.width == pytest.approx(size_effective[0])
        assert scalar_bar.opts.height == pytest.approx(size_effective[1])
    finally:
        figure.act_close()


def test_scalar_bar_rebuild_replaces_widget_observer():
    figure = PlotFigure(is_off_screen=True, name="scalar_bar_widget_rebuild_test")
    try:
        _make_scalar_sphere(figure)
        scalar_bar = figure.scalar_bars[0]

        scalar_bar.act_commit(is_interactive=True)
        widget_before = scalar_bar.backend_widget
        observer_before = scalar_bar.impl_backend_widget_observer_tag
        assert widget_before is not None
        assert observer_before is not None

        scalar_bar.act_commit(is_interactive=False)
        assert scalar_bar.backend_widget is None
        assert scalar_bar.impl_backend_widget_observer_tag is None

        scalar_bar.act_commit(is_interactive=True)
        assert scalar_bar.backend_widget is not None
        assert scalar_bar.backend_widget is not widget_before
        assert scalar_bar.impl_backend_widget_observer_tag is not None
    finally:
        figure.act_close()


def test_scalar_bar_visibility_updates_widget_and_backend():
    figure = PlotFigure(is_off_screen=True, name="scalar_bar_visibility_test")
    try:
        _make_scalar_sphere(figure)
        scalar_bar = figure.scalar_bars[0]
        scalar_bar.act_commit(is_interactive=True)
        widget = scalar_bar.backend_widget
        assert widget is not None

        scalar_bar.act_commit(is_visible=False)
        assert scalar_bar.backend.GetVisibility() == 0
        assert widget.GetEnabled() == 0

        scalar_bar.act_commit(is_visible=True)
        assert scalar_bar.backend.GetVisibility() == 1
        assert widget.GetEnabled() == 1
    finally:
        figure.act_close()


def test_scalar_bar_registry_clear_cleans_backends_relations_and_registry():
    figure = PlotFigure(is_off_screen=True, name="scalar_bar_clear_test")
    try:
        _make_scalar_sphere(figure)
        scalar_bar = figure.scalar_bars[0]
        backend_name = scalar_bar.impl_name_pv

        removed = figure.scalar_bars.act_clear(
            is_return_removed=True,
            is_show_existing=False,
        )

        assert removed == (scalar_bar,)
        assert scalar_bar.backend is None
        assert scalar_bar.backend_widget is None
        assert scalar_bar.owner is None
        assert scalar_bar.registry is None
        assert backend_name not in figure.pl.scalar_bars
        assert len(figure.scalar_bars) == 0
    finally:
        figure.act_close()


def test_scalar_bar_old_widget_no_longer_writes_back_after_unregister():
    figure = PlotFigure(is_off_screen=True, name="scalar_bar_widget_detach_test")
    try:
        _make_scalar_sphere(figure)
        scalar_bar = figure.scalar_bars[0]
        scalar_bar.act_commit(is_interactive=True)
        widget = scalar_bar.backend_widget
        assert widget is not None

        position_before = tuple(scalar_bar.opts.position)
        figure.act_unregister_scalar_bar(scalar_bar)

        rep = widget.GetRepresentation()
        rep.SetPosition(0.61, 0.17)
        widget.InvokeEvent("EndInteractionEvent")

        assert tuple(scalar_bar.opts.position) == pytest.approx(position_before)
        assert scalar_bar.impl_backend_widget_observer_tag is None
    finally:
        figure.act_close()


def test_scalar_bar_source_display_updates_title_cmap_and_clim():
    figure = PlotFigure(is_off_screen=True, name="scalar_bar_source_display_test")
    try:
        sphere = _make_scalar_sphere(figure)
        scalar_bar = figure.scalar_bars[0]

        sphere.act_commit(
            scalar_bar_title="updated scalars",
            scalars_cmap="plasma",
            scalars_clim=(0.2, 0.8),
        )

        assert scalar_bar.backend.GetTitle() == "updated scalars"
        assert scalar_bar.backend.GetLookupTable().GetRange() == pytest.approx(
            (0.2, 0.8)
        )
    finally:
        figure.act_close()


def test_scalar_bar_zero_labels_hides_tick_labels():
    figure = PlotFigure(is_off_screen=True, name="scalar_bar_zero_labels_test")
    try:
        _make_scalar_sphere(figure)
        scalar_bar = figure.scalar_bars[0]

        scalar_bar.act_commit(n_labels=0)

        assert scalar_bar.backend.GetDrawTickLabels() == 0
    finally:
        figure.act_close()


def test_glyph_disabling_scalar_bar_unregisters_it():
    figure = PlotFigure(is_off_screen=True, name="scalar_bar_disable_test")
    try:
        sphere = _make_scalar_sphere(figure)
        scalar_bar = figure.scalar_bars[0]

        sphere.act_commit(is_scalar_bar=False)

        assert len(figure.scalar_bars) == 0
        assert scalar_bar.backend is None
        assert scalar_bar.owner is None
        assert scalar_bar.registry is None
    finally:
        figure.act_close()
