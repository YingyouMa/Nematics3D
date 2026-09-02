import numpy as np
import pytest

from nematics3d.geometry import find_nearest_point


def test_find_nearest_point_returns_nearest_point():
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]])

    result = find_nearest_point([1.1, 0.9], coords)

    np.testing.assert_allclose(result, [1.0, 1.0])


def test_find_nearest_point_can_return_index():
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]])

    point, index = find_nearest_point([1.8, 0.1], coords, is_return_idx=True)

    np.testing.assert_allclose(point, [2.0, 0.0])
    assert index == 1
    assert isinstance(index, int)


def test_find_nearest_point_uses_first_point_on_tie():
    coords = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 2.0]])

    point, index = find_nearest_point([0.0, 0.0], coords, is_return_idx=True)

    np.testing.assert_allclose(point, [-1.0, 0.0])
    assert index == 0


def test_find_nearest_point_supports_arbitrary_dimension():
    coords = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]])

    result = find_nearest_point([0.9, 2.1, 3.0, 4.1], coords)

    np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0])


def test_find_nearest_point_returns_copy():
    coords = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = find_nearest_point([1.0, 2.0], coords)
    result[0] = 99.0

    assert coords[0, 0] == 1.0


@pytest.mark.parametrize(
    ("query_pt", "coords", "message"),
    [
        ([[0.0, 0.0]], [[0.0, 0.0]], "one-dimensional"),
        ([0.0, 0.0], [0.0, 0.0], "two-dimensional"),
        ([0.0, 0.0], np.empty((0, 2)), "at least one point"),
        ([0.0, 0.0], [[0.0, 0.0, 0.0]], "same coordinate dimension"),
    ],
)
def test_find_nearest_point_rejects_invalid_shapes(query_pt, coords, message):
    with pytest.raises(ValueError, match=message):
        find_nearest_point(query_pt, coords)


@pytest.mark.parametrize(
    ("query_pt", "coords", "message"),
    [
        ([np.nan, 0.0], [[0.0, 0.0]], "query_pt"),
        ([0.0, np.inf], [[0.0, 0.0]], "query_pt"),
        ([0.0, 0.0], [[np.nan, 0.0]], "coords"),
        ([0.0, 0.0], [[0.0, np.inf]], "coords"),
    ],
)
def test_find_nearest_point_rejects_nonfinite_values(query_pt, coords, message):
    with pytest.raises(ValueError, match=message):
        find_nearest_point(query_pt, coords)


@pytest.mark.parametrize("is_return_idx", [0, 1, 1.0, "true", None])
def test_find_nearest_point_rejects_non_boolean_return_flag(is_return_idx):
    with pytest.raises(TypeError, match="is_return_idx"):
        find_nearest_point([0.0], [[0.0]], is_return_idx=is_return_idx)
