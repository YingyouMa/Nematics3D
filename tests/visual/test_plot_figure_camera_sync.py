import numpy as np
import pyvista as pv

from nematics3d.classes.visual.plot_figure import PlotFigure


def test_plotter_camera_state_is_imported_when_wrapping_existing_plotter():
    plotter = pv.Plotter(off_screen=True)
    plotter.camera.position = (8.0, 3.0, 5.0)
    plotter.camera.focal_point = (1.0, -2.0, 0.5)
    plotter.camera.up = (0.0, 0.0, 1.0)

    figure = PlotFigure(plotter=plotter, is_off_screen=True)

    np.testing.assert_allclose(figure.opts.focal_point, (1.0, -2.0, 0.5))
    assert figure.opts.distance == np.linalg.norm(
        np.array((8.0, 3.0, 5.0)) - np.array((1.0, -2.0, 0.5))
    )


def test_explicit_camera_opts_override_wrapped_plotter_camera():
    plotter = pv.Plotter(off_screen=True)
    plotter.camera.position = (20.0, 0.0, 0.0)
    plotter.camera.focal_point = (0.0, 0.0, 0.0)

    figure = PlotFigure(
        plotter=plotter,
        is_off_screen=True,
        azimuth=35.0,
        elevation=20.0,
        roll=-15.0,
        distance=7.0,
        focal_point=(1.0, 2.0, 3.0),
    )

    assert figure.opts.azimuth == 35.0
    assert figure.opts.elevation == 20.0
    assert figure.opts.roll == -15.0
    assert figure.opts.distance == 7.0
    np.testing.assert_allclose(figure.opts.focal_point, (1.0, 2.0, 3.0))
    np.testing.assert_allclose(figure.pl.camera.focal_point, (1.0, 2.0, 3.0))
    np.testing.assert_allclose(
        np.linalg.norm(
            np.asarray(figure.pl.camera.position)
            - np.asarray(figure.pl.camera.focal_point)
        ),
        7.0,
    )


def test_commit_camera_opts_updates_plotter_and_round_trips_back_to_opts():
    figure = PlotFigure(is_off_screen=True)

    figure.act_commit(
        azimuth=123.0,
        elevation=-31.0,
        roll=47.0,
        distance=9.5,
        focal_point=(-2.0, 4.0, 1.5),
    )

    position_before = np.asarray(figure.pl.camera.position).copy()
    up_before = np.asarray(figure.pl.camera.up).copy()
    figure._helper_sync_from_plotter(is_only_camera=True)

    np.testing.assert_allclose(figure.opts.azimuth, 123.0, atol=1e-10)
    np.testing.assert_allclose(figure.opts.elevation, -31.0, atol=1e-10)
    np.testing.assert_allclose(figure.opts.roll, 47.0, atol=1e-10)
    np.testing.assert_allclose(figure.opts.distance, 9.5, atol=1e-10)
    np.testing.assert_allclose(figure.opts.focal_point, (-2.0, 4.0, 1.5))
    np.testing.assert_allclose(figure.pl.camera.position, position_before)
    np.testing.assert_allclose(figure.pl.camera.up, up_before)


def test_sync_from_plotter_can_leave_window_opts_unchanged():
    figure = PlotFigure(is_off_screen=True)
    original_size = np.asarray(figure.opts.size).copy()
    original_background = np.asarray(figure.opts.bg_color).copy()
    figure.pl.window_size = (640, 480)
    figure.pl.set_background((0.2, 0.3, 0.4))
    figure.pl.camera.position = (4.0, 5.0, 6.0)

    figure._helper_sync_from_plotter(is_only_camera=True)

    np.testing.assert_array_equal(figure.opts.size, original_size)
    np.testing.assert_array_equal(figure.opts.bg_color, original_background)
    assert figure.opts.distance > 0


def test_sync_from_plotter_imports_window_state_by_default():
    figure = PlotFigure(is_off_screen=True)
    figure.pl.window_size = (640, 480)
    figure.pl.set_background((0.2, 0.3, 0.4))

    figure._helper_sync_from_plotter()

    np.testing.assert_array_equal(figure.opts.size, (640, 480))
    np.testing.assert_allclose(figure.opts.bg_color, (0.2, 0.3, 0.4))


def test_standard_view_action_syncs_camera_back_to_opts():
    figure = PlotFigure(is_off_screen=True)

    figure.act_view_xy()

    camera = figure.pl.camera
    np.testing.assert_allclose(figure.opts.focal_point, camera.focal_point)
    np.testing.assert_allclose(
        figure.opts.distance,
        np.linalg.norm(np.asarray(camera.position) - np.asarray(camera.focal_point)),
    )
