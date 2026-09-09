import numpy as np

from nematics3d.classes.grid_field import GridFieldDataset, InputGridField
from nematics3d.classes.vector_plane import VectorPlane as LegacyVectorPlane
from nematics3d.sample.plane_grid import OptsPlaneGrid
from nematics3d.sample.vector_plane import VectorPlane
from nematics3d.visual.plot_figure import PlotFigure


def _vector_plane():
    dataset = GridFieldDataset(inputValue=InputGridField(shape=(4, 4, 4)))
    values = dataset.act_generate_grid()
    field = dataset.act_add_field("v", values)
    plane = VectorPlane(
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


def test_vector_plane_preserves_legacy_identity():
    assert LegacyVectorPlane is VectorPlane


def test_vector_plane_exposes_full_and_selected_magnitudes():
    plane, _field, _dataset = _vector_plane()
    np.testing.assert_allclose(
        plane.calc_magnitude_all,
        np.linalg.norm(plane.calc_result_all, axis=1),
    )
    np.testing.assert_allclose(
        plane.calc_magnitude,
        plane.calc_magnitude_all[plane.grid.calc_box_mask],
    )


def test_vector_plane_visual_refreshes_through_visual_adapter():
    plane, _field, _dataset = _vector_plane()
    figure = PlotFigure(is_off_screen=True)
    try:
        visual = plane.act_visualize_vector(figure=figure)
        assert visual is plane.visual
        plane.grid.act_commit(origin=(1.0, 1.0, 1.0))
        np.testing.assert_allclose(visual.raw_coords, plane.grid())
        np.testing.assert_allclose(visual.raw_orient, plane.result)
    finally:
        figure.act_close()
