"""Focused tests for the experimental surface-director interpolator."""

import numpy as np
import pyvista as pv

from director_interpolation import interpolate_surface_directors


def _square_surface():
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    faces = np.array([[3, 0, 1, 2], [3, 0, 2, 3]], dtype=np.int64).ravel()
    return pv.PolyData(points, faces)


def test_nematic_signs_do_not_cancel_during_interpolation():
    surface = _square_surface()
    vertex_directors = np.array(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    positions = np.array([[0.75, 0.25, 0.0], [0.25, 0.75, 0.0]])

    result = interpolate_surface_directors(surface, vertex_directors, positions)

    np.testing.assert_allclose(np.abs(result.directors[:, 0]), 1.0)
    np.testing.assert_allclose(result.directors[:, 1:], 0.0)
    np.testing.assert_allclose(np.sum(result.barycentric_coordinates, axis=1), 1.0)
    np.testing.assert_allclose(result.surface_distances, 0.0)
    assert np.all(result.is_interpolable)


def test_reference_director_selects_continuous_output_sign():
    surface = _square_surface()
    vertex_directors = np.tile([1.0, 0.0, 0.0], (4, 1))
    positions = np.array([[0.75, 0.25, 0.0], [0.25, 0.75, 0.0]])

    result = interpolate_surface_directors(
        surface,
        vertex_directors,
        positions,
        reference_directors=np.array([-1.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(result.directors, [[-1.0, 0.0, 0.0]] * 2)


def test_off_surface_queries_report_closest_points_and_distances():
    surface = _square_surface()
    vertex_directors = np.tile([1.0, 0.0, 0.0], (4, 1))
    positions = np.array([[0.25, 0.25, 0.5]])

    result = interpolate_surface_directors(surface, vertex_directors, positions)

    np.testing.assert_allclose(result.surface_positions, [[0.25, 0.25, 0.0]])
    np.testing.assert_allclose(result.surface_distances, [0.5])
    assert result.cell_indices.shape == (1,)


def test_results_are_read_only():
    result = interpolate_surface_directors(
        _square_surface(),
        np.tile([1.0, 0.0, 0.0], (4, 1)),
        [[0.5, 0.5, 0.0]],
    )

    assert not result.directors.flags.writeable
    assert not result.barycentric_coordinates.flags.writeable
