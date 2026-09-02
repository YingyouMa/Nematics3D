import numpy as np
import pytest

from nematics3d.geometry import points_membership_mask


def test_points_membership_mask_basic():
    points = np.array([[0, 0], [1, 2], [3, 4]])
    candidates = np.array([[3, 4], [0, 0]])

    result = points_membership_mask(points, candidates)

    np.testing.assert_array_equal(result, [True, False, True])


def test_points_membership_mask_promotes_integer_dtypes():
    points = np.array([[1, 2], [3, 4]], dtype=np.int32)
    candidates = np.array([[1, 2]], dtype=np.int64)

    result = points_membership_mask(points, candidates)

    np.testing.assert_array_equal(result, [True, False])


def test_points_membership_mask_promotes_numeric_dtypes():
    points = np.array([[1, 2], [3, 4]], dtype=np.int64)
    candidates = np.array([[1.0, 2.0]], dtype=np.float64)

    result = points_membership_mask(points, candidates)

    np.testing.assert_array_equal(result, [True, False])


def test_points_membership_mask_ignores_candidate_duplicates():
    points = np.array([[1, 2], [5, 6]])
    candidates = np.array([[1, 2], [1, 2], [1, 2]])

    result = points_membership_mask(points, candidates)

    np.testing.assert_array_equal(result, [True, False])


def test_points_membership_mask_empty_points():
    result = points_membership_mask(
        np.empty((0, 2)),
        np.array([[1, 2]]),
    )

    assert result.dtype == bool
    assert result.shape == (0,)


def test_points_membership_mask_empty_candidates():
    result = points_membership_mask(
        np.array([[1, 2], [3, 4]]),
        np.empty((0, 2)),
    )

    np.testing.assert_array_equal(result, [False, False])


def test_points_membership_mask_float_comparison_is_exact():
    points = np.array([[1.0, 2.0], [1.0, 2.0 + 1e-12]])
    candidates = np.array([[1.0, 2.0]])

    result = points_membership_mask(points, candidates)

    np.testing.assert_array_equal(result, [True, False])


def test_points_membership_mask_treats_signed_zero_as_equal():
    points = np.array([[-0.0, 1.0]])
    candidates = np.array([[0.0, 1.0]])

    result = points_membership_mask(points, candidates)

    np.testing.assert_array_equal(result, [True])


def test_points_membership_mask_rejects_non_2d_input():
    with pytest.raises(ValueError, match="two-dimensional"):
        points_membership_mask(np.array([1, 2]), np.array([[1, 2]]))


def test_points_membership_mask_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="same coordinate dimension"):
        points_membership_mask(np.array([[1, 2]]), np.array([[1, 2, 3]]))


def test_points_membership_mask_rejects_nonfinite_input():
    with pytest.raises(ValueError):
        points_membership_mask(np.array([[np.nan, 0.0]]), np.array([[0.0, 0.0]]))
