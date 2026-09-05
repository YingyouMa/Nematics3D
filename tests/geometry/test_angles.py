import numpy as np
import pytest

from nematics3d.geometry import (
    azimuth_from_vector,
    plane_azimuth_from_direction,
    polar_angle_from_vector,
    vector_from_spherical_angles,
    wrap_angle_to_pi,
)


def test_vector_from_spherical_angles_scalar_convention():
    vector = vector_from_spherical_angles(np.pi / 2, np.pi / 2)
    np.testing.assert_allclose(vector, [0.0, 1.0, 0.0], atol=1e-15)


def test_vector_from_spherical_angles_broadcasts_with_coordinate_axis_last():
    vectors = vector_from_spherical_angles(
        np.array([0.0, np.pi / 2]),
        np.pi / 2,
    )
    assert vectors.shape == (2, 3)
    np.testing.assert_allclose(vectors, [[1, 0, 0], [0, 1, 0]], atol=1e-15)


def test_spherical_angle_round_trip_for_batched_vectors():
    vectors = np.array([[1, 1, 1], [-1, 1, 0], [0, 0, -2]], dtype=float)
    azimuth = azimuth_from_vector(vectors)
    polar = polar_angle_from_vector(vectors)
    recovered = vector_from_spherical_angles(azimuth, polar)
    expected = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
    np.testing.assert_allclose(recovered, expected, atol=1e-15)


def test_angle_conversions_do_not_modify_inputs():
    azimuth = np.array([0.0, np.pi / 2])
    polar = np.array([np.pi / 2, np.pi / 3])
    vectors = np.array([[1.0, 1.0, 1.0], [-1.0, 2.0, 3.0]])
    azimuth_before = azimuth.copy()
    polar_before = polar.copy()
    vectors_before = vectors.copy()

    vector_from_spherical_angles(azimuth, polar)
    azimuth_from_vector(vectors)
    polar_angle_from_vector(vectors)

    np.testing.assert_array_equal(azimuth, azimuth_before)
    np.testing.assert_array_equal(polar, polar_before)
    np.testing.assert_array_equal(vectors, vectors_before)


def test_azimuth_at_poles_is_zero_by_convention():
    np.testing.assert_allclose(azimuth_from_vector([[0, 0, 1], [0, 0, -1]]), 0)


def test_single_vector_angles_are_python_floats():
    assert isinstance(azimuth_from_vector([1, 0, 0]), float)
    assert isinstance(polar_angle_from_vector([1, 0, 0]), float)


def test_batched_vector_angles_preserve_leading_shape():
    vectors = np.ones((2, 4, 3))
    azimuth = azimuth_from_vector(vectors)
    polar = polar_angle_from_vector(vectors)
    assert isinstance(azimuth, np.ndarray)
    assert isinstance(polar, np.ndarray)
    assert azimuth.shape == (2, 4)
    assert polar.shape == (2, 4)


@pytest.mark.parametrize("function", [azimuth_from_vector, polar_angle_from_vector])
def test_vector_angles_reject_zero_nonfinite_and_wrong_shape(function):
    with pytest.raises(ValueError, match="zero vectors"):
        function([0, 0, 0])
    with pytest.raises(ValueError, match="finite"):
        function([1, np.nan, 0])
    with pytest.raises(ValueError, match="shape"):
        function([1, 0])


def test_vector_from_spherical_angles_rejects_invalid_inputs():
    with pytest.raises(TypeError, match="real numbers"):
        vector_from_spherical_angles(True, 0)
    with pytest.raises(ValueError, match="finite"):
        vector_from_spherical_angles(np.inf, 0)
    with pytest.raises(ValueError, match="broadcastable"):
        vector_from_spherical_angles(np.zeros(2), np.zeros(3))


@pytest.mark.parametrize(
    "function,args",
    [
        (azimuth_from_vector, ([1 + 1j, 0, 0],)),
        (polar_angle_from_vector, (["x", 0, 0],)),
        (vector_from_spherical_angles, (1 + 1j, 0)),
        (wrap_angle_to_pi, ("x",)),
    ],
)
def test_angle_functions_reject_nonreal_or_nonnumeric_inputs(function, args):
    with pytest.raises(TypeError, match="real numbers"):
        function(*args)


def test_plane_azimuth_uses_rotated_local_frame():
    assert plane_azimuth_from_direction([1, 0, 0], [0, 0, 1]) == pytest.approx(0)
    assert plane_azimuth_from_direction([0, 1, 0], [0, 0, 1]) == pytest.approx(
        np.pi / 2
    )


def test_plane_azimuth_projects_direction_onto_plane():
    angle = plane_azimuth_from_direction([1, 1, 3], [0, 0, 1])
    assert angle == pytest.approx(np.pi / 4)


def test_plane_azimuth_rejects_parallel_direction():
    with pytest.raises(ValueError, match="parallel"):
        plane_azimuth_from_direction([0, 0, 2], [0, 0, 1])


def test_plane_azimuth_does_not_modify_inputs():
    direction = np.array([1.0, 1.0, 2.0])
    normal = np.array([0.0, 0.0, 1.0])
    direction_before = direction.copy()
    normal_before = normal.copy()
    plane_azimuth_from_direction(direction, normal)
    np.testing.assert_array_equal(direction, direction_before)
    np.testing.assert_array_equal(normal, normal_before)


def test_wrap_angle_to_pi_scalar_and_array_boundaries():
    wrapped_scalar = wrap_angle_to_pi(np.pi)
    assert isinstance(wrapped_scalar, float)
    assert wrapped_scalar == pytest.approx(-np.pi)
    wrapped = wrap_angle_to_pi(np.array([-3 * np.pi, -np.pi, 0, np.pi, 3 * np.pi]))
    assert isinstance(wrapped, np.ndarray)
    np.testing.assert_allclose(wrapped, [-np.pi, -np.pi, 0, -np.pi, -np.pi])


def test_wrap_angle_to_pi_preserves_array_shape_and_input():
    angles = np.array([[0.0, np.pi], [2 * np.pi, -np.pi / 2]])
    angles_before = angles.copy()
    wrapped = wrap_angle_to_pi(angles)
    assert wrapped.shape == angles.shape
    np.testing.assert_array_equal(angles, angles_before)


def test_wrap_angle_to_pi_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        wrap_angle_to_pi([0, np.inf])
