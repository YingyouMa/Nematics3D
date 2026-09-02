"""Tests for point collection validation and normalization."""

import numpy as np
import pytest

from nematics3d.datatypes import as_points


def test_as_points_promotes_one_point_and_returns_independent_float_array():
    point = np.array([1, 2, 3])

    points = as_points(point)

    assert points.shape == (1, 3)
    assert np.issubdtype(points.dtype, np.floating)
    assert not np.shares_memory(points, point)
    np.testing.assert_allclose(points, [[1.0, 2.0, 3.0]])


def test_as_points_accepts_arbitrary_dimension_when_d_is_none():
    points = as_points([[1.0, 2.0], [3.0, 4.0]], d=None)

    assert points.shape == (2, 2)


def test_as_points_normalizes_empty_input_using_requested_dimension():
    assert as_points([], d=4).shape == (0, 4)
    assert as_points([], d=None).shape == (0, 0)


def test_as_points_can_reject_empty_input():
    with pytest.raises(ValueError, match="at least one point"):
        as_points([], is_empty=False)


def test_as_points_rejects_nonfinite_values_by_default():
    with pytest.raises(ValueError, match="finite"):
        as_points([[0.0, np.nan, 1.0]])

    points = as_points([[0.0, np.inf, 1.0]], is_finite=False)
    assert np.isinf(points[0, 1])


@pytest.mark.parametrize(
    "points",
    [
        [[True, False, True]],
        [[True, 2.0, 3.0]],
        [["1", "2", "3"]],
        [[1.0 + 1.0j, 2.0, 3.0]],
    ],
)
def test_as_points_rejects_non_real_coordinate_types(points):
    with pytest.raises(TypeError, match="real numbers"):
        as_points(points)


@pytest.mark.parametrize("d", [True, 0, -1, 2.5])
def test_as_points_validates_dimension(d):
    with pytest.raises((TypeError, ValueError)):
        as_points([[1.0, 2.0, 3.0]], d=d)


def test_as_points_deduplicates_before_checking_minimum_count():
    points = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    with pytest.raises(ValueError, match="at least 2 point"):
        as_points(points, is_unique=True, min_num=2)
