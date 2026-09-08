import numpy as np

from nematics3d.classes.bounds import Bounds
from nematics3d.classes.visual.plot_contour_surface import (
    OptsContourSurface as LegacyOptsContourSurface,
)
from nematics3d.classes.visual.plot_contour_surface import (
    PlotContourSurface as LegacyPlotContourSurface,
)
from nematics3d.classes.visual.qt.interact_contour_surface import (
    InteractContourSurface as LegacyInteractContourSurface,
)
from nematics3d.visual.plot_contour_surface import (
    OptsContourSurface,
    PlotContourSurface,
)
from nematics3d.surface.contour import ContourSurfaceSet
from nematics3d.visual.plot_figure import PlotFigure
from nematics3d.visual.qt.interact_contour_surface import InteractContourSurface


def test_contour_visual_legacy_imports_alias_canonical_classes():
    assert LegacyOptsContourSurface is OptsContourSurface
    assert LegacyPlotContourSurface is PlotContourSurface
    assert LegacyInteractContourSurface is InteractContourSurface


def _radial_values(shape=(9, 9, 9)):
    grid = np.indices(shape, dtype=float)
    center = (np.asarray(shape, dtype=float) - 1.0) / 2.0
    delta = grid - center[:, None, None, None]
    return np.sqrt(np.sum(delta**2, axis=0))


def test_contour_level_refresh_updates_topology_and_preserves_visual_opts():
    figure = PlotFigure(is_off_screen=True)
    try:
        contour = ContourSurfaceSet(_radial_values(), levels=(2.0,))
        surface = contour[0]
        visual = surface.act_plot(
            figure=figure,
            color=(0.2, 0.4, 0.8),
            opacity=0.65,
            is_show_edges=True,
            edge_color=(0.1, 0.2, 0.3),
            line_width=2.5,
        )

        n_points_before = visual.entity_actor.mapper.dataset.n_points
        surface.act_set_level(3.0)

        assert visual.calc_level == 3.0
        np.testing.assert_allclose(visual.raw_coords, surface.mesh.points)
        assert visual.entity_actor.mapper.dataset.n_points == surface.mesh.n_points
        assert visual.entity_actor.mapper.dataset.n_points != n_points_before
        assert visual.opts.color == (0.2, 0.4, 0.8)
        assert visual.opts.opacity == 0.65
        assert visual.opts.is_show_edges is True
        assert visual.opts.edge_color == (0.1, 0.2, 0.3)
        assert visual.opts.line_width == 2.5
    finally:
        figure.act_close()


def test_contour_level_refresh_reapplies_mesh_bounds_clipping():
    figure = PlotFigure(is_off_screen=True)
    try:
        contour = ContourSurfaceSet(_radial_values(), levels=(3.0,))
        bounds = Bounds(
            length1=4.0,
            length2=4.0,
            length3=4.0,
            origin=(2.0, 2.0, 2.0),
            alignment="min_corner",
        )
        surface = contour[0]
        visual = surface.act_plot(figure=figure, bounds=bounds)

        mesh_rendered_before = visual.entity_actor.mapper.dataset
        assert mesh_rendered_before.n_points < surface.mesh.n_points

        surface.act_set_level(2.5)

        mesh_rendered_after = visual.entity_actor.mapper.dataset
        assert mesh_rendered_after.n_points < surface.mesh.n_points
        assert mesh_rendered_after.n_points > 0
        assert visual.state_clip_mode == "mesh"
    finally:
        figure.act_close()
