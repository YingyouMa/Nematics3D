import numpy as np
from unittest.mock import patch

from nematics3d.analysis.disclination.plane import PlaneDefectResult
from nematics3d.bounds import Bounds, OptsBounds
from nematics3d.classes.grid_field import GridFieldDataset, InputGridField
from nematics3d.classes.q_plane import QPlane as LegacyQPlane
from nematics3d.q_field import get_q
from nematics3d.sample.plane_grid import OptsPlaneGrid
from nematics3d.sample.q_plane import QPlane
from nematics3d.visual.plot_figure import PlotFigure
from nematics3d.visual.qt.interact_defect_section import InteractDefectSection
from nematics3d.visual.qt.interact_plane import InteractPlane


def _q_plane():
    dataset = GridFieldDataset(inputValue=InputGridField(shape=(4, 4, 4)))
    director = np.zeros((4, 4, 4, 3), dtype=float)
    director[..., 0] = 1.0
    q_values = get_q(director, S=np.ones((4, 4, 4), dtype=float))
    field = dataset.act_add_field("Q", q_values)
    plane = QPlane(
        interpolator=field.act_add_interpolator(),
        opts=OptsPlaneGrid(
            normal=(0.0, 0.0, 1.0),
            axis1=(1.0, 0.0, 0.0),
            origin=(1.5, 1.5, 1.5),
            spacing=1.0,
            size=2.0,
        ),
    )
    return plane, field, dataset


def test_q_plane_and_interactions_preserve_legacy_identity():
    assert LegacyQPlane is QPlane


def test_q_plane_exposes_full_and_selected_sampling_results():
    plane, _field, _dataset = _q_plane()
    assert (
        plane.calc_result_all.shape[0]
        == plane.grid.entity_grid_all.reshape(-1, 3).shape[0]
    )
    assert plane.calc_n_all.shape[0] == plane.calc_result_all.shape[0]
    assert plane.calc_S_all.shape[0] == plane.calc_result_all.shape[0]
    assert plane.calc_is_near_defect_all.shape[0] == plane.calc_result_all.shape[0]
    np.testing.assert_allclose(
        plane.calc_result, plane.calc_result_all[plane.grid.calc_box_mask]
    )
    np.testing.assert_allclose(plane.calc_n, plane.calc_n_all[plane.grid.calc_box_mask])
    np.testing.assert_allclose(plane.calc_S, plane.calc_S_all[plane.grid.calc_box_mask])


def test_q_plane_visuals_are_created_from_visual_adapter_and_refresh():
    plane, _field, _dataset = _q_plane()
    figure = PlotFigure(is_off_screen=True)
    try:
        assert plane.act_visualize_n(figure=figure) is None
        assert plane.visual_nb is not None
        assert plane.visual_nd is not None
        assert plane.visual_defect is not None

        assert plane.act_visualize_S(figure=figure) is None
        assert plane.visual_S is not None

        plane.grid.act_commit(origin=(1.0, 1.0, 1.0))
        np.testing.assert_allclose(plane.visual_nb.raw_coords, plane.grid())
        np.testing.assert_allclose(plane.visual_S.raw_coords, plane.grid())
        np.testing.assert_allclose(plane.visual_S.calc_scalars, plane.calc_S)
    finally:
        figure.act_close()


def test_q_plane_detects_on_full_plane_before_bounds_selection():
    plane, _field, _dataset = _q_plane()
    positions_all = np.array(
        [
            [0.5, 0.5, 1.5],
            [2.5, 2.5, 1.5],
        ]
    )
    result = PlaneDefectResult(
        positions_all=positions_all,
        adjacent_mask_all=np.zeros(len(plane.calc_result_all), dtype=bool),
    )
    bounds = Bounds(
        opts=OptsBounds(
            origin=(0.0, 0.0, 0.0),
            axis1=(1.0, 0.0, 0.0),
            axis2=(0.0, 1.0, 0.0),
            length1=2.0,
            length2=2.0,
            length3=2.0,
            alignment="min_corner",
        )
    )

    with patch.object(QPlane, "_helper_detect_defects_all", return_value=result):
        plane.grid.act_bind_bounds(bounds)

    np.testing.assert_allclose(plane.calc_defect_pos_all, positions_all)
    np.testing.assert_allclose(plane.calc_defect_pos, positions_all[:1])
