import numpy as np
import pytest

from nematics3d.core.result_base import ResultBase
from nematics3d.geometry import (
    RotationAxisResult,
    find_rotation_axis,
    rotation_matrix_from_vectors,
)


def test_find_rotation_axis_returns_result_base():
    angles = np.linspace(0.0, np.pi / 2.0, 6)
    directors = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))

    result = find_rotation_axis(directors)

    assert isinstance(result, RotationAxisResult)
    assert isinstance(result, ResultBase)
    assert np.allclose(result.axis, [0.0, 0.0, 1.0])
    assert result.orthogonality_score == pytest.approx(1.0)
    assert result.rms_sin_theta == pytest.approx(0.0)
    assert result.tilt_angle_degrees == pytest.approx(0.0)
    assert result.rotation_consistency == pytest.approx(1.0)
    assert np.allclose(result.metric["eigenvalues"], result.eigenvalues)


def test_rotation_axis_orientation_follows_ordered_rotation():
    angles = np.linspace(0.0, -np.pi / 2.0, 6)
    directors = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))

    result = find_rotation_axis(directors)

    assert np.allclose(result.axis, [0.0, 0.0, -1.0])
    assert result.rotation_consistency == pytest.approx(1.0)


def test_rotation_axis_metric_matches_result_fields():
    angles = np.linspace(0.0, np.pi / 3.0, 5)
    z = 0.2
    xy = np.sqrt(1.0 - z**2)
    directors = np.column_stack(
        (xy * np.cos(angles), xy * np.sin(angles), np.full_like(angles, z))
    )

    result = find_rotation_axis(directors)
    metric = result.metric

    assert metric["orthogonality_score"] == result.orthogonality_score
    assert metric["rms_sin_theta"] == result.rms_sin_theta
    assert metric["tilt_angle_degrees"] == result.tilt_angle_degrees
    assert metric["rotation_consistency"] == result.rotation_consistency
    assert metric["eigenvalues"] is result.eigenvalues


def test_find_rotation_axis_requires_at_least_two_directors():
    with pytest.raises(ValueError):
        find_rotation_axis([[1.0, 0.0, 0.0]])


def test_find_rotation_axis_requires_unit_directors():
    with pytest.raises(ValueError, match="normalized unit vectors"):
        find_rotation_axis([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])


def test_find_rotation_axis_rejects_nonfinite_directors():
    with pytest.raises(ValueError):
        find_rotation_axis([[1.0, 0.0, 0.0], [0.0, np.nan, 1.0]])


def test_rotation_matrix_maps_source_to_target_and_is_proper_rotation():
    source = np.array([1.0, 0.0, 0.0])
    target = np.array([0.0, 1.0, 0.0])

    rotation = rotation_matrix_from_vectors(source, target)

    assert np.allclose(rotation @ source, target)
    assert np.allclose(rotation.T @ rotation, np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_rotation_matrix_parallel_vectors_returns_identity():
    vector = np.array([1.0, 2.0, 3.0])
    vector /= np.linalg.norm(vector)

    rotation = rotation_matrix_from_vectors(vector, vector)

    assert np.allclose(rotation, np.eye(3))


def test_rotation_matrix_antiparallel_vectors_is_deterministic():
    source = np.array([1.0, 0.0, 0.0])
    target = -source

    rotation = rotation_matrix_from_vectors(source, target)

    expected = np.diag([-1.0, -1.0, 1.0])
    assert np.allclose(rotation, expected)
    assert np.allclose(rotation @ source, target)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_rotation_matrix_rejects_non_unit_vectors():
    with pytest.raises(ValueError):
        rotation_matrix_from_vectors([2.0, 0.0, 0.0], [0.0, 1.0, 0.0])


def test_rotation_matrix_rejects_nonfinite_vectors():
    with pytest.raises(ValueError):
        rotation_matrix_from_vectors([1.0, 0.0, 0.0], [0.0, np.nan, 1.0])
