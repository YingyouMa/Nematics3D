import numpy as np
import pytest

from nematics3d.core.result_base import ResultBase
from nematics3d.geometry import PlaneNormalResult, find_plane_normal


def test_find_plane_normal_returns_result_base_for_exact_plane():
    points = np.array(
        [
            [0.0, 0.0, 3.0],
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 4.0],
            [-1.0, 2.0, -1.0],
        ]
    )

    result = find_plane_normal(points)
    expected_normal = np.array([2.0, -1.0, -1.0])
    expected_normal /= np.linalg.norm(expected_normal)

    assert isinstance(result, PlaneNormalResult)
    assert isinstance(result, ResultBase)
    assert abs(float(np.dot(result.normal, expected_normal))) == pytest.approx(1.0)
    assert np.allclose(result.centroid, points.mean(axis=0))
    assert result.planarity_score == pytest.approx(1.0)
    assert result.thickness_rms == pytest.approx(0.0, abs=1e-12)


def test_plane_normal_metric_matches_result_fields():
    points = np.array(
        [
            [-1.0, -1.0, 0.05],
            [1.0, -1.0, -0.02],
            [1.0, 1.0, 0.03],
            [-1.0, 1.0, -0.04],
        ]
    )

    result = find_plane_normal(points)
    metric = result.metric

    assert metric["centroid"] is result.centroid
    assert metric["planarity_score"] == result.planarity_score
    assert metric["thickness_rms"] == result.thickness_rms
    assert metric["linearity_risk"] == result.linearity_risk
    assert metric["eigenvalues"] is result.eigenvalues


def test_current_internal_tuple_unpack_path_still_returns_result_base():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    result = find_plane_normal(points, is_return_metric=True)
    normal, metric = result

    assert isinstance(result, PlaneNormalResult)
    assert isinstance(result, ResultBase)
    assert np.allclose(normal, result.normal)
    assert metric["planarity_score"] == result.planarity_score


def test_linearity_risk_detects_exact_line():
    points = np.array(
        [
            [-2.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    result = find_plane_normal(points)

    assert result.linearity_risk == pytest.approx(1.0)
    assert result.planarity_score == pytest.approx(1.0)


def test_find_plane_normal_requires_three_points():
    with pytest.raises(ValueError):
        find_plane_normal([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def test_find_plane_normal_requires_3d_points():
    with pytest.raises(ValueError):
        find_plane_normal([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def test_find_plane_normal_rejects_nonfinite_points():
    with pytest.raises(ValueError):
        find_plane_normal(
            [[0.0, 0.0, 0.0], [1.0, np.nan, 0.0], [0.0, 1.0, 0.0]]
        )
