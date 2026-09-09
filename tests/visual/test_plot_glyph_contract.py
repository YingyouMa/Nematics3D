import numpy as np
import pytest

from nematics3d.analysis.bounds import Bounds, OptsBounds
from nematics3d.classes.visual.plot_figure import PlotFigure
from nematics3d.classes.visual.plot_rod import PlotRod
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


def test_opts_assignment_updates_resolved_radius_and_live_instance_data(figure):
    sphere = PlotSphere(_coords(), figure=figure, radius=0.2, sides=8)
    mapper_before = sphere.entity_actor.mapper
    poly_before = mapper_before.GetInput()
    assert poly_before.GetNumberOfPoints() == 3

    sphere.opts.radius = 0.4

    np.testing.assert_allclose(sphere.calc_radius, 0.4)
    assert sphere.opts.radius == pytest.approx(0.4)
    mapper_after = sphere.entity_actor.mapper
    poly_after = mapper_after.GetInput()
    assert mapper_after is mapper_before
    assert poly_after.GetNumberOfPoints() == 3
    assert poly_after is not poly_before
    np.testing.assert_allclose(poly_after.point_data["radius"], 0.4)


def test_instanced_sphere_reuses_mapper_until_actor_is_recreated(figure):
    sphere = PlotSphere(_coords(), figure=figure, sides=8)
    mapper = sphere.entity_actor.mapper

    sphere.act_commit(
        radius=np.array([0.1, 0.2, 0.3]),
        opacity=np.array([0.4, 0.6, 0.8]),
        color=(0.2, 0.4, 0.8),
    )

    assert sphere.entity_actor.mapper is mapper
    np.testing.assert_allclose(mapper.GetInput().point_data["radius"], [0.1, 0.2, 0.3])


def test_instanced_sphere_sides_update_reuses_mapper_and_replaces_source(figure):
    sphere = PlotSphere(_coords(), figure=figure, sides=8)
    mapper = sphere.entity_actor.mapper
    source_before = mapper.GetSource(0)

    sphere.opts.sides = 14

    assert sphere.entity_actor.mapper is mapper
    assert mapper.GetSource(0) is not source_before


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


def test_instanced_sphere_mesh_clip_is_explicitly_deferred(figure):
    with pytest.raises(NotImplementedError, match="mesh clip_mode"):
        PlotSphere(_coords(), figure=figure, clip_mode="mesh", sides=8)


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
    assert sphere.entity_silhouette is None

    sphere.act_highlight(color=(1.0, 0.0, 0.0), opacity=0.5, width=3.0)
    silhouette = sphere.entity_silhouette
    assert silhouette is not None
    assert silhouette.visibility
    assert silhouette.prop.opacity == pytest.approx(0.5)
    assert silhouette.prop.line_width == pytest.approx(3.0)

    sphere.act_dehighlight()
    assert not silhouette.visibility


def test_lazy_silhouette_is_not_built_until_first_highlight(figure):
    sphere = PlotSphere(_coords(), figure=figure, sides=8)

    assert sphere.entity_silhouette is None

    sphere.act_highlight()

    assert sphere.entity_silhouette is not None
    assert sphere.entity_silhouette.visibility


def test_existing_instanced_silhouette_is_updated_in_place_and_keeps_visibility(figure):
    sphere = PlotSphere(_coords(), figure=figure, radius=0.2, sides=8)
    sphere.act_highlight()
    silhouette_before = sphere.entity_silhouette
    mapper_before = silhouette_before.mapper
    poly_before = mapper_before.GetInput()

    sphere.opts.radius = 0.4

    assert sphere.entity_silhouette is silhouette_before
    assert sphere.entity_silhouette.mapper is mapper_before
    assert mapper_before.GetInput() is not poly_before
    assert sphere.entity_silhouette.visibility

    sphere.act_dehighlight()
    silhouette_hidden = sphere.entity_silhouette
    sphere.opts.radius = 0.3
    assert sphere.entity_silhouette is silhouette_hidden
    assert sphere.entity_silhouette.mapper is mapper_before
    assert not sphere.entity_silhouette.visibility


def test_instanced_rod_uses_one_center_per_instance_and_component_scaling(figure):
    orient = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, -3.0]], dtype=float)
    rod = PlotRod(
        _coords(),
        orient,
        figure=figure,
        length=np.array([1.0, 2.0, 3.0]),
        radius=np.array([0.1, 0.2, 0.3]),
        sides=8,
    )

    mapper = rod.entity_actor.mapper
    poly = mapper.GetInput()

    assert poly.GetNumberOfPoints() == 3
    np.testing.assert_allclose(poly.points, _coords())
    np.testing.assert_allclose(
        poly.point_data["orient"],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
    )
    np.testing.assert_allclose(
        poly.point_data["scale"],
        [[1.0, 0.1, 0.1], [2.0, 0.2, 0.2], [3.0, 0.3, 0.3]],
    )
    assert mapper.GetOrient()
    assert mapper.GetScaleMode() == mapper.SCALE_BY_COMPONENTS


def test_instanced_rod_reuses_mapper_for_geometry_and_color_updates(figure):
    orient = np.eye(3, dtype=float)
    rod = PlotRod(_coords(), orient, figure=figure, sides=8)
    mapper = rod.entity_actor.mapper

    rod.act_commit(
        length=np.array([2.0, 3.0, 4.0]),
        radius=np.array([0.2, 0.3, 0.4]),
        color=(0.2, 0.4, 0.8),
    )

    assert rod.entity_actor.mapper is mapper
    np.testing.assert_allclose(
        mapper.GetInput().point_data["scale"],
        [[2.0, 0.2, 0.2], [3.0, 0.3, 0.3], [4.0, 0.4, 0.4]],
    )


def test_instanced_rod_sides_update_reuses_mapper_and_replaces_source(figure):
    rod = PlotRod(_coords(), np.eye(3), figure=figure, sides=8)
    mapper = rod.entity_actor.mapper
    source_before = mapper.GetSource(0)

    rod.opts.sides = 14

    assert rod.entity_actor.mapper is mapper
    assert mapper.GetSource(0) is not source_before


def test_instanced_rod_center_clipping_keeps_instance_arrays_aligned(figure):
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
    orient = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    rod = PlotRod(
        _coords(),
        orient,
        figure=figure,
        bounds=bounds,
        length=np.array([1.0, 2.0, 3.0]),
        radius=np.array([0.1, 0.2, 0.3]),
        sides=8,
    )

    np.testing.assert_array_equal(rod.calc_keep_index, [0, 1])
    poly = rod.entity_actor.mapper.GetInput()
    np.testing.assert_allclose(poly.points[:, 0], [0.0, 1.0])
    np.testing.assert_allclose(poly.point_data["orient"], [[1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(poly.point_data["scale"], [[1, 0.1, 0.1], [2, 0.2, 0.2]])


def test_instanced_rod_scalar_coloring_and_scalar_bar(figure):
    rod = PlotRod(
        _coords(),
        np.eye(3),
        figure=figure,
        paint_by="scalars",
        scalars=np.array([0.0, 1.0, 2.0]),
        is_scalar_bar=True,
        sides=8,
    )

    mapper = rod.entity_actor.mapper
    np.testing.assert_allclose(mapper.GetInput().point_data["scalars"], [0.0, 1.0, 2.0])
    assert mapper.GetScalarVisibility()
    assert len(figure.scalar_bars) == 1


def test_instanced_rod_silhouette_uses_same_instance_geometry_configuration(figure):
    rod = PlotRod(_coords(), np.eye(3), figure=figure, sides=8)
    rod.act_highlight()

    mapper = rod.entity_silhouette.mapper
    assert mapper.GetOrient()
    assert mapper.GetScaleMode() == mapper.SCALE_BY_COMPONENTS
    np.testing.assert_allclose(
        mapper.GetInput().point_data["scale"],
        rod.entity_actor.mapper.GetInput().point_data["scale"],
    )


def test_instanced_rod_near_zero_orientation_remains_finite(figure):
    orient = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1e-8, 0.0]])
    rod = PlotRod(_coords(), orient, figure=figure, sides=8)

    resolved = rod.entity_actor.mapper.GetInput().point_data["orient"]
    assert np.all(np.isfinite(resolved))
    np.testing.assert_allclose(resolved[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(resolved[1], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(resolved[2], [0.0, 1e-8, 0.0])


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
