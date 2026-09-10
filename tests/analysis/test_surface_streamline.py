import numpy as np
import pyvista as pv

from nematics3d.analysis import integrate_surface_streamline


def test_constant_planar_field_traces_both_directions_to_boundary():
    surface = pv.Plane(i_resolution=8, j_resolution=8).triangulate()
    directors = np.tile([1.0, 0.0, 0.0], (surface.n_points, 1))

    result = integrate_surface_streamline(
        surface,
        directors,
        [0.0, 0.0, 0.0],
        step_size=0.05,
        max_length=4.0,
    )

    np.testing.assert_allclose(result.positions[:, 1:], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.positions[0, 0], -0.5, atol=1.0e-12)
    np.testing.assert_allclose(result.positions[-1, 0], 0.5, atol=1.0e-12)
    assert result.forward_status == "stagnated at surface constraint"
    assert result.backward_status == "stagnated at surface constraint"
    assert not result.positions.flags.writeable


def test_alternating_nematic_vertex_signs_still_trace_a_line():
    surface = pv.Plane(i_resolution=4, j_resolution=4).triangulate()
    directors = np.tile([1.0, 0.0, 0.0], (surface.n_points, 1))
    directors[::2] *= -1.0

    result = integrate_surface_streamline(
        surface,
        directors,
        [0.0, 0.0, 0.0],
        step_size=0.1,
        max_length=0.6,
    )

    assert len(result.positions) >= 5
    np.testing.assert_allclose(result.positions[:, 1:], 0.0, atol=1.0e-12)


def test_existing_streamline_stops_a_new_line_at_minimum_separation():
    surface = pv.Plane(i_resolution=8, j_resolution=8).triangulate()
    directors = np.tile([1.0, 0.0, 0.0], (surface.n_points, 1))
    stop_positions = np.column_stack(
        (np.linspace(-0.5, 0.5, 21), np.full(21, 0.1), np.zeros(21))
    )

    result = integrate_surface_streamline(
        surface,
        directors,
        [0.0, 0.0, 0.0],
        step_size=0.05,
        max_length=2.0,
        stop_positions=stop_positions,
        minimum_separation=0.2,
    )

    assert result.forward_status == "minimum separation"
    assert result.backward_status == "minimum separation"


def test_azimuthal_sphere_field_closes_once_and_stays_on_surface():
    surface = pv.Sphere(radius=1.0, theta_resolution=64, phi_resolution=32).triangulate()
    points = np.asarray(surface.points, dtype=float)
    directors = np.column_stack((-points[:, 1], points[:, 0], np.zeros(len(points))))

    result = integrate_surface_streamline(
        surface,
        directors,
        [1.0, 0.0, 0.0],
        step_size=0.05,
        max_length=10.0,
        closure_tolerance=0.06,
        minimum_closure_length=5.0,
    )

    assert result.forward_status == "closed loop"
    assert result.backward_status == "not traced: forward branch closed loop"
    np.testing.assert_allclose(result.positions[0], result.positions[-1], atol=1.0e-12)
    np.testing.assert_allclose(
        np.linalg.norm(result.positions, axis=1),
        1.0,
        atol=3.0e-3,
    )
    np.testing.assert_allclose(result.positions[:, 2], 0.0, atol=3.0e-3)
    assert 5.5 < result.length < 7.0


def test_normal_vertex_field_is_rejected_at_seed_after_tangent_projection():
    surface = pv.Plane(i_resolution=4, j_resolution=4).triangulate()
    directors = np.tile([0.0, 0.0, 1.0], (surface.n_points, 1))

    try:
        integrate_surface_streamline(surface, directors, [0.0, 0.0, 0.0])
    except ValueError as error:
        assert "nonzero tangent director" in str(error)
    else:
        raise AssertionError("Expected a normal director field to be rejected.")
