import numpy as np
import pyvista as pv

from nematics3d.visual.plot_figure import PlotFigure


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
    actual_background = figure.pl.background_color.float_rgb

    figure._helper_sync_from_plotter()

    np.testing.assert_array_equal(figure.opts.size, (640, 480))
    np.testing.assert_allclose(figure.opts.bg_color, actual_background)


def test_figure_size_is_stored_as_positive_integer_pixels():
    figure = PlotFigure(is_off_screen=True, size=(640.0, 480.0))

    np.testing.assert_array_equal(figure.opts.size, (640, 480))
    assert np.issubdtype(figure.opts.size.dtype, np.integer)
    assert tuple(figure.pl.window_size) == (640, 480)


def test_invalid_figure_size_is_rejected_without_truncating_pixels():
    figure = PlotFigure(is_off_screen=True)
    original_size = np.asarray(figure.opts.size).copy()

    figure.opts.size = (640.5, 480)
    np.testing.assert_array_equal(figure.opts.size, original_size)

    figure.opts.size = (0, 480)
    np.testing.assert_array_equal(figure.opts.size, original_size)

    figure.opts.size = (-640, 480)
    np.testing.assert_array_equal(figure.opts.size, original_size)


def test_committing_window_state_updates_plotter():
    figure = PlotFigure(is_off_screen=True)

    figure.act_commit(size=(720, 540), bg_color=(0.1, 0.2, 0.3))

    assert tuple(figure.pl.window_size) == (720, 540)
    np.testing.assert_allclose(
        figure.pl.background_color.float_rgb,
        figure.opts.bg_color,
        atol=1.0 / 255.0,
    )


def test_standard_view_action_syncs_camera_back_to_opts():
    figure = PlotFigure(is_off_screen=True)

    figure.act_view_xy()

    camera = figure.pl.camera
    np.testing.assert_allclose(figure.opts.focal_point, camera.focal_point)
    np.testing.assert_allclose(
        figure.opts.distance,
        np.linalg.norm(np.asarray(camera.position) - np.asarray(camera.focal_point)),
    )


def test_standard_views_follow_pyvista_axis_convention():
    figure = PlotFigure(is_off_screen=True)

    cases = [
        (figure.act_view_xy, np.array([0.0, 0.0, 1.0])),
        (figure.act_view_xz, np.array([0.0, -1.0, 0.0])),
        (figure.act_view_yz, np.array([1.0, 0.0, 0.0])),
    ]

    for action, expected_position_direction in cases:
        action()
        camera = figure.pl.camera
        position_direction = np.asarray(camera.position) - np.asarray(
            camera.focal_point
        )
        position_direction /= np.linalg.norm(position_direction)
        np.testing.assert_allclose(
            position_direction,
            expected_position_direction,
            atol=1e-12,
        )


def test_standard_view_actions_are_idempotent_in_camera_pose():
    figure = PlotFigure(is_off_screen=True)

    for action in (
        figure.act_view_xy,
        figure.act_view_xz,
        figure.act_view_yz,
        figure.act_view_isometric,
    ):
        action()
        first_camera = (
            np.asarray(figure.pl.camera.position).copy(),
            np.asarray(figure.pl.camera.focal_point).copy(),
            np.asarray(figure.pl.camera.up).copy(),
        )
        first_opts = (
            figure.opts.azimuth,
            figure.opts.elevation,
            figure.opts.roll,
            figure.opts.distance,
            np.asarray(figure.opts.focal_point).copy(),
        )

        action()
        second_camera = (
            np.asarray(figure.pl.camera.position),
            np.asarray(figure.pl.camera.focal_point),
            np.asarray(figure.pl.camera.up),
        )
        second_opts = (
            figure.opts.azimuth,
            figure.opts.elevation,
            figure.opts.roll,
            figure.opts.distance,
            np.asarray(figure.opts.focal_point),
        )

        for lhs, rhs in zip(first_camera, second_camera):
            np.testing.assert_allclose(lhs, rhs, atol=1e-12)
        for lhs, rhs in zip(first_opts[:4], second_opts[:4]):
            np.testing.assert_allclose(lhs, rhs, atol=1e-12)
        np.testing.assert_allclose(first_opts[4], second_opts[4], atol=1e-12)


def test_reset_camera_preserves_valid_synced_pose():
    figure = PlotFigure(is_off_screen=True)
    figure.pl.add_mesh(pv.Sphere(radius=2.0, center=(3.0, -1.0, 4.0)))

    figure.act_reset_camera()

    camera = figure.pl.camera
    assert figure.opts.distance > 0
    np.testing.assert_allclose(figure.opts.focal_point, camera.focal_point)
    np.testing.assert_allclose(
        figure.opts.distance,
        np.linalg.norm(np.asarray(camera.position) - np.asarray(camera.focal_point)),
    )
