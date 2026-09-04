"""Focused tests for the experimental surface-streamline integrator."""

import numpy as np
import pyvista as pv

from surface_streamline import integrate_surface_streamline


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
