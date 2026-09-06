import numpy as np
import pytest

from nematics3d.classes.bounds import Bounds, OptsBounds
from nematics3d.classes.visual.plot_figure import PlotFigure
from nematics3d.classes.visual.plot_sphere import PlotSphere


@pytest.fixture
def figure():
    fig = PlotFigure(is_off_screen=True, name="glyph_contract")
    try:
        yield fig
    finally:
        fig.act_close()


def _coords():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )


def test_opts_assignment_updates_resolved_radius_and_live_mesh(figure):
    sphere = PlotSphere(_coords(), figure=figure, radius=0.2, sides=8)
    mesh_before = sphere.entity_actor.mapper.dataset

    sphere.opts.radius = 0.4

    np.testing.assert_allclose(sphere.calc_radius, 0.4)
    assert sphere.opts.radius == pytest.approx(0.4)
    mesh_after = sphere.entity_actor.mapper.dataset
    assert mesh_after.n_points > 0
    assert mesh_after is not mesh_before


def test_batch_commit_updates_multiple_visual_inputs(figure):
    sphere = PlotSphere(_coords(), figure=figure, sides=8)

    sphere.act_commit(
        radius=np.array([0.1, 0.2, 0.3]),
        opacity=np.array([0.25, 0.5, 0.75]),
        color=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )

    np.testing.assert_allclose(sphere.calc_radius, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(sphere.calc_opacity, [0.25, 0.5, 0.75])
    np.testing.assert_allclose(
        sphere.calc_color,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    assert sphere.opts.paint_by == "color"


def test_raw_coords_replacement_reapplies_callable_opts(figure):
    sphere = PlotSphere(
        _coords(),
        figure=figure,
        resolver_source="coords",
        radius=lambda pts: pts[:, 0] + 1.0,
        sides=8,
    )
    np.testing.assert_allclose(sphere.calc_radius, [1.0, 2.0, 3.0])

    sphere.coords = np.array([[4.0, 0.0, 0.0], [5.0, 0.0, 0.0]])

    np.testing.assert_allclose(sphere.raw_coords[:, 0], [4.0, 5.0])
    np.testing.assert_allclose(sphere.calc_radius, [5.0, 6.0])
    assert sphere.calc_color.shape == (2, 3)
    assert sphere.entity_actor is not None


def test_resolver_source_change_re_resolves_existing_callable(figure):
    sphere = PlotSphere(
        _coords(),
        figure=figure,
        resolver_source="coords",
        radius=lambda source: np.asarray(source)[:, 0] + 1.0,
        sides=8,
    )

    sphere.act_commit(
        radius=lambda source: np.asarray(source, dtype=float) / 100.0 + 1.0,
        resolver_source="u_percent",
    )

    np.testing.assert_allclose(sphere.calc_radius, [1.0, 1.5, 2.0])
    assert sphere.opts.resolver_source == "u_percent"


def test_attr_specific_resolver_source_only_reresolves_target_attr(figure):
    calls = {"radius": 0, "opacity": 0}

    def radius(source):
        calls["radius"] += 1
        return np.full(len(source), 0.2)

    def opacity(source):
        calls["opacity"] += 1
        return np.full(len(source), 0.8)

    sphere = PlotSphere(
        _coords(),
        figure=figure,
        resolver_source="coords",
        radius=radius,
        opacity=opacity,
        sides=8,
    )
    calls_before = dict(calls)

    sphere.act_commit(resolver_source_opacity="u_percent")

    assert calls["opacity"] == calls_before["opacity"] + 1
    assert calls["radius"] == calls_before["radius"]


def test_color_scalar_pipeline_switch_adds_and_removes_scalar_bar(figure):
    sphere = PlotSphere(_coords(), figure=figure, color=(0.2, 0.3, 0.4), sides=8)
    assert sphere.opts.paint_by == "color"
    assert len(figure.scalar_bars) == 0

    sphere.act_commit(scalars=np.array([0.0, 1.0, 2.0]))
    assert sphere.opts.paint_by == "scalars"
    assert len(figure.scalar_bars) == 1

    sphere.act_commit(color=(1.0, 0.0, 0.0))
    assert sphere.opts.paint_by == "color"
    assert len(figure.scalar_bars) == 0


def test_scalar_bar_toggle_updates_registry_without_recreating_glyph(figure):
    sphere = PlotSphere(
        _coords(),
        figure=figure,
        paint_by="scalars",
        scalars=np.array([0.0, 1.0, 2.0]),
        is_scalar_bar=True,
        sides=8,
    )
    actor = sphere.entity_actor
    assert len(figure.scalar_bars) == 1

    sphere.opts.is_scalar_bar = False
    assert sphere.entity_actor is actor
    assert len(figure.scalar_bars) == 0

    sphere.opts.is_scalar_bar = True
    assert sphere.entity_actor is actor
    assert len(figure.scalar_bars) == 1


def test_center_bounds_bind_disable_enable_and_unbind(figure):
    bounds = Bounds(
        opts=OptsBounds(
            origin=(-0.25, -0.5, -0.5),
            axis1=(1.0, 0.0, 0.0),
            axis2=(0.0, 1.0, 0.0),
            length1=1.5,
            length2=1.0,
            length3=1.0,
            alignment="min_corner",
        )
    )
    sphere = PlotSphere(_coords(), figure=figure, bounds=bounds, sides=8)

    np.testing.assert_array_equal(sphere.calc_keep_index, [0, 1])
    np.testing.assert_allclose(sphere.calc_coords[:, 0], [0.0, 1.0])

    sphere.act_bounds_disable()
    np.testing.assert_array_equal(sphere.calc_keep_index, [0, 1, 2])

    sphere.act_bounds_enable()
    np.testing.assert_array_equal(sphere.calc_keep_index, [0, 1])

    sphere.act_unbind_bounds()
    assert sphere.bounds is None
    np.testing.assert_array_equal(sphere.calc_keep_index, [0, 1, 2])


def test_empty_to_nonempty_and_back_transition(figure):
    sphere = PlotSphere(np.empty((0, 3)), figure=figure, radius=0.2, sides=8)
    assert sphere.calc_is_empty
    assert sphere.entity_actor is None

    sphere.coords = _coords()
    assert not sphere.calc_is_empty
    assert sphere.entity_actor is not None
    assert sphere.calc_radius.shape == (3,)

    sphere.coords = np.empty((0, 3))
    assert sphere.calc_is_empty
    assert sphere.entity_actor is None
    assert sphere.calc_radius.shape == (0,)


def test_protected_opts_reject_live_updates(figure):
    sphere = PlotSphere(_coords(), figure=figure, radius=0.2, sides=8)
    sphere.act_register_protected_attr("radius")

    sphere.opts.radius = 0.8

    assert sphere.opts.radius == pytest.approx(0.2)
    np.testing.assert_allclose(sphere.calc_radius, 0.2)

    sphere.act_unregister_protected_attr("radius")
    sphere.opts.radius = 0.8
    assert sphere.opts.radius == pytest.approx(0.8)
    np.testing.assert_allclose(sphere.calc_radius, 0.8)


def test_highlight_and_dehighlight_update_silhouette(figure):
    sphere = PlotSphere(_coords(), figure=figure, sides=8)
    silhouette = sphere.entity_silhouette
    assert silhouette is not None
    assert not silhouette.visibility

    sphere.act_highlight(color=(1.0, 0.0, 0.0), opacity=0.5, width=3.0)
    assert silhouette.visibility
    assert silhouette.prop.opacity == pytest.approx(0.5)
    assert silhouette.prop.line_width == pytest.approx(3.0)

    sphere.act_dehighlight()
    assert not silhouette.visibility


def test_pick_reports_nearest_point_visual_values(figure):
    sphere = PlotSphere(
        _coords(),
        figure=figure,
        radius=np.array([0.1, 0.2, 0.3]),
        opacity=np.array([0.4, 0.5, 0.6]),
        sides=8,
    )

    pos, message, idx = sphere.act_resolve_pick(np.array([1.05, 0.0, 0.0]))

    assert idx == 1
    np.testing.assert_allclose(pos, [1.0, 0.0, 0.0])
    assert "Local radius" in message
    assert "Local opacity" in message


def test_remove_cleans_figure_registration_relations_and_actor(figure):
    sphere = PlotSphere(_coords(), figure=figure, sides=8)
    assert sphere in list(figure.glyphs)
    assert sphere.fig is figure
    assert sphere.entity_actor is not None

    sphere.act_remove()

    assert sphere not in list(figure.glyphs)
    assert sphere.fig is None
    assert sphere.bounds is None
    assert sphere.entity_actor is None
