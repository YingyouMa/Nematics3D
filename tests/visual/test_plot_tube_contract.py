import numpy as np

from nematics3d.analysis.bounds import Bounds
from nematics3d.classes.visual.plot_tube import OptsTube as LegacyOptsTube
from nematics3d.classes.visual.plot_tube import PlotTube as LegacyPlotTube
from nematics3d.visual.plot_figure import PlotFigure
from nematics3d.visual.plot_tube import OptsTube, PlotTube


def test_tube_legacy_imports_alias_canonical_classes():
    assert LegacyOptsTube is OptsTube
    assert LegacyPlotTube is PlotTube


def _figure():
    return PlotFigure(is_off_screen=True)


def test_tube_pick_does_not_bridge_disconnected_paths():
    figure = _figure()
    try:
        tube = PlotTube(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [10.0, 1.0, 0.0],
                ]
            ),
            line_index=np.array([0, 0, 1, 1]),
            figure=figure,
            radius=0.1,
        )

        pos, _msg, _idx = tube._helper_resolve_pick(np.array([5.0, 0.5, 0.0]))
        assert pos[0] in (0.0, 10.0)
        assert pos[1] == 0.5
    finally:
        figure.act_close()


def test_tube_pick_preserves_legacy_u_percent_sampling_convention():
    figure = _figure()
    try:
        tube = PlotTube(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ]
            ),
            figure=figure,
            radius=0.1,
        )
        _pos, msg, idx = tube._helper_resolve_pick(np.array([2.0, 0.0, 0.0]))
        assert idx == 2
        assert "66.667" in msg
    finally:
        figure.act_close()


def test_tube_render_topology_does_not_bridge_disconnected_paths():
    figure = _figure()
    try:
        tube = PlotTube(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [10.0, 1.0, 0.0],
                ]
            ),
            line_index=np.array([0, 0, 1, 1]),
            figure=figure,
            radius=0.1,
        )

        np.testing.assert_array_equal(
            tube.calc_poly.lines,
            np.array([2, 0, 1, 2, 2, 3]),
        )
    finally:
        figure.act_close()


def test_tube_center_clipping_uses_public_bounds_containment_contract():
    figure = _figure()
    try:
        bounds = Bounds(
            length1=2.0,
            length2=2.0,
            length3=2.0,
            origin=(0.0, 0.0, 0.0),
            alignment="min_corner",
        )
        tube = PlotTube(
            np.array(
                [
                    [-1.0, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [1.0, 0.5, 0.5],
                    [3.0, 0.5, 0.5],
                ]
            ),
            figure=figure,
            bounds=bounds,
            radius=0.1,
        )
        np.testing.assert_array_equal(tube.calc_keep_index, np.array([1, 2]))
        np.testing.assert_allclose(
            tube.calc_coords,
            np.array([[0.5, 0.5, 0.5], [1.0, 0.5, 0.5]]),
        )
    finally:
        figure.act_close()
