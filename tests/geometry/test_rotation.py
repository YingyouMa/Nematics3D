import numpy as np
import pytest

from nematics3d.core.result_base import ResultBase
from nematics3d.geometry import RotationAxisResult, find_rotation_axis


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
