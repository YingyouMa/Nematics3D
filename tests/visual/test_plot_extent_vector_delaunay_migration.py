import numpy as np

from nematics3d.analysis.bounds import Bounds
from nematics3d.classes.visual.plot_delaunay import (
    OptsDelaunay as LegacyOptsDelaunay,
)
from nematics3d.classes.visual.plot_delaunay import PlotDelaunay as LegacyPlotDelaunay
from nematics3d.classes.visual.plot_extent import PlotExtent as LegacyPlotExtent
from nematics3d.classes.visual.plot_vector import OptsVector as LegacyOptsVector
from nematics3d.classes.visual.plot_vector import PlotVector as LegacyPlotVector
from nematics3d.visual.plot_delaunay import OptsDelaunay, PlotDelaunay
from nematics3d.visual.plot_extent import PlotExtent
from nematics3d.visual.plot_figure import PlotFigure
from nematics3d.visual.plot_vector import OptsVector, PlotVector


def test_migrated_classes_preserve_legacy_identity():
    assert LegacyPlotExtent is PlotExtent
    assert LegacyOptsVector is OptsVector
    assert LegacyPlotVector is PlotVector
    assert LegacyOptsDelaunay is OptsDelaunay
    assert LegacyPlotDelaunay is PlotDelaunay


def test_extent_builds_twelve_disconnected_edges_from_corners():
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    coords, line_index = PlotExtent._helper_build_edges_from_corners(corners)
    assert coords.shape == (24, 3)
    np.testing.assert_array_equal(line_index, np.repeat(np.arange(12), 2))
    for edge_idx, (a, b) in enumerate(PlotExtent._EDGES):
        np.testing.assert_allclose(coords[2 * edge_idx], corners[a])
        np.testing.assert_allclose(coords[2 * edge_idx + 1], corners[b])


def _bounds():
    return Bounds(
        length1=2.0,
        length2=2.0,
        length3=2.0,
        origin=(0.0, 0.0, 0.0),
        alignment="min_corner",
    )


def test_vector_center_clipping_matches_public_bounds_contract():
    figure = PlotFigure(is_off_screen=True)
    try:
        coords = np.array([[-1.0, 0.5, 0.5], [0.5, 0.5, 0.5], [3.0, 0.5, 0.5]])
        vector = PlotVector(
            coords,
            np.tile([1.0, 0.0, 0.0], (3, 1)),
            figure=figure,
            bounds=_bounds(),
            sides=6,
        )
        np.testing.assert_array_equal(vector.calc_keep_index, np.array([1]))
    finally:
        figure.act_close()


def test_delaunay_center_clipping_matches_public_bounds_contract():
    figure = PlotFigure(is_off_screen=True)
    try:
        coords = np.array(
            [
                [-1.0, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [1.0, 0.5, 0.5],
                [0.5, 1.0, 0.5],
                [3.0, 0.5, 0.5],
            ]
        )
        surface = PlotDelaunay(coords, figure=figure, bounds=_bounds())
        np.testing.assert_array_equal(surface.calc_keep_index, np.array([1, 2, 3]))
    finally:
        figure.act_close()
