import numpy as np
import pyvista as pv
import pytest

from nematics3d.analysis import (
    SurfaceDirectorProjectionResult,
    project_surface_directors,
)


def _triangle_surface():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    return pv.PolyData(points, faces=np.array([3, 0, 1, 2]))


def test_project_surface_directors_on_plane():
    surface = _triangle_surface()
    directors = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    result = project_surface_directors(surface, directors)

    assert isinstance(result, SurfaceDirectorProjectionResult)
    np.testing.assert_allclose(result.projected_directors[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(result.projected_directors[1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(result.projected_directors[2], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(result.tilt_angles_degrees, [0.0, 45.0, 90.0])
    np.testing.assert_array_equal(result.is_projectable, [True, True, False])


def test_project_surface_directors_is_nematic_sign_invariant():
    surface = _triangle_surface()
    directors = np.array([[1.0, 0.0, 1.0]] * surface.n_points)

    positive = project_surface_directors(surface, directors)
    negative = project_surface_directors(surface, -directors)

    np.testing.assert_allclose(
        positive.tilt_angles_degrees, negative.tilt_angles_degrees
    )
    np.testing.assert_allclose(
        positive.projected_directors, -negative.projected_directors
    )


def test_project_surface_directors_warns_above_tilt_threshold():
    surface = _triangle_surface()
    directors = np.array([[1.0, 0.0, 1.0]] * surface.n_points)

    with pytest.warns(RuntimeWarning, match="exceed"):
        result = project_surface_directors(
            surface, directors, max_tilt_degrees=30.0
        )

    np.testing.assert_array_equal(result.exceeded_indices, [0, 1, 2])


def test_project_surface_directors_zero_director_is_not_projectable():
    surface = _triangle_surface()
    directors = np.zeros((surface.n_points, 3))

    result = project_surface_directors(surface, directors)

    assert not np.any(result.is_projectable)
    assert np.all(np.isnan(result.tilt_angles_degrees))
    assert np.all(np.isnan(result.normal_fractions))
    assert np.all(np.isnan(result.tangent_fractions))


def test_project_surface_directors_rejects_wrong_director_count():
    surface = _triangle_surface()

    with pytest.raises(ValueError, match="one director per surface vertex"):
        project_surface_directors(surface, np.ones((2, 3)))


def test_projection_result_arrays_are_read_only():
    surface = _triangle_surface()
    result = project_surface_directors(surface, np.ones((surface.n_points, 3)))

    with pytest.raises(ValueError):
        result.projected_directors[0, 0] = 0.0
