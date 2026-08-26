"""Tests for periodic trajectory utilities."""

import numpy as np
import pytest

from nematics3d.grid import shift_to_box, unwrap_trajectory


def test_shift_to_box_returns_copy_by_default():
    points = np.array([[12.0, -3.0, 7.0], [13.0, -2.0, 8.0]])
    shifted = shift_to_box(points, [10.0, 10.0, np.inf])
    np.testing.assert_allclose(shifted, [[2.0, 7.0, 7.0], [3.0, 8.0, 8.0]])
    np.testing.assert_allclose(points, [[12.0, -3.0, 7.0], [13.0, -2.0, 8.0]])


def test_shift_to_box_can_modify_input_inplace():
    points = np.array([[12.0, -3.0, 7.0], [13.0, -2.0, 8.0]])
    shifted = shift_to_box(points, [10.0, 10.0, np.inf], is_inplace=True)
    assert shifted is points
    np.testing.assert_allclose(points, [[2.0, 7.0, 7.0], [3.0, 8.0, 8.0]])


def test_shift_to_box_validates_reference_index():
    points = np.zeros((2, 3))

    with pytest.raises(IndexError):
        shift_to_box(points, 10.0, ref_index=2)
    with pytest.raises(TypeError):
        shift_to_box(points, 10.0, ref_index=True)


def test_unwrap_trajectory_across_mixed_periodic_boundaries():
    points = np.array([[9.0, 1.0, 0.0], [1.0, 9.0, 2.0], [3.0, 7.0, 4.0]])
    result = unwrap_trajectory(points, [10.0, 10.0, np.inf])
    np.testing.assert_allclose(
        result, [[9.0, 1.0, 0.0], [11.0, -1.0, 2.0], [13.0, -3.0, 4.0]]
    )
    np.testing.assert_allclose(
        points,
        [[9.0, 1.0, 0.0], [1.0, 9.0, 2.0], [3.0, 7.0, 4.0]],
    )


def test_unwrap_trajectory_reverse_direction():
    points = np.array([[1.0, 0.0, 0.0], [9.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
    result = unwrap_trajectory(points, 10.0, is_reverse=True)
    np.testing.assert_allclose(
        result, [[11.0, 0.0, 0.0], [9.0, 0.0, 0.0], [7.0, 0.0, 0.0]]
    )


def test_unwrap_trajectory_places_reference_in_box():
    points = np.array([[9.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    result = unwrap_trajectory(points, 10.0, is_start_in_box=True, ref_index=1)
    np.testing.assert_allclose(
        result, [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )


def test_unwrap_trajectory_handles_empty_and_single_point_inputs():
    assert unwrap_trajectory([], 10.0).shape == (0, 3)
    np.testing.assert_allclose(
        unwrap_trajectory([1.0, 2.0, 3.0], 10.0), [[1.0, 2.0, 3.0]]
    )


def test_unwrap_trajectory_validates_reference():
    points = np.zeros((2, 3))
    with pytest.raises(IndexError):
        unwrap_trajectory(points, 10.0, is_start_in_box=True, ref_index=2)
    with pytest.raises(TypeError):
        unwrap_trajectory(points, 10.0, is_start_in_box=True, ref_index=True)


@pytest.mark.parametrize(
    "points",
    [np.zeros((2, 2)), np.zeros((2, 3, 1)), [[0.0, np.nan, 0.0]]],
)
def test_unwrap_trajectory_rejects_invalid_points(points):
    with pytest.raises(ValueError):
        unwrap_trajectory(points, 10.0)
