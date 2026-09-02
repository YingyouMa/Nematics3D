import numpy as np
import pytest

from nematics3d.geometry import closest_point_on_polyline, find_nearest_point


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


def test_find_nearest_point_tie_result_depends_on_input_order():
    query = [0.0, 0.0]
    coords_a = np.array([[-1.0, 0.0], [1.0, 0.0]])
    coords_b = coords_a[::-1].copy()
    point_a, index_a = find_nearest_point(query, coords_a, is_return_idx=True)
    point_b, index_b = find_nearest_point(query, coords_b, is_return_idx=True)
    np.testing.assert_allclose(point_a, [-1.0, 0.0])
    np.testing.assert_allclose(point_b, [1.0, 0.0])
    assert index_a == index_b == 0


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
        ([[0.0, 0.0]], [[0.0, 0.0]], "must have shape"),
        ([0.0, 0.0], [0.0, 0.0], "two-dimensional"),
        ([0.0, 0.0], np.empty((0, 2)), "at least one point"),
        ([0.0, 0.0], [[0.0, 0.0, 0.0]], "must have shape"),
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


@pytest.mark.parametrize("is_return_idx", ["true", None])
def test_find_nearest_point_rejects_non_boolean_return_flag(is_return_idx):
    with pytest.raises(TypeError, match="is_return_idx"):
        find_nearest_point([0.0], [[0.0]], is_return_idx=is_return_idx)


@pytest.mark.parametrize(
    "is_return_idx, expected", [(0, False), (1, True), (1.0, True)]
)
def test_find_nearest_point_accepts_numeric_boolean_return_flag(
    is_return_idx, expected
):
    result = find_nearest_point(
        [0.0],
        [[0.0]],
        is_return_idx=is_return_idx,
    )
    assert isinstance(result, tuple) is expected


def test_closest_point_on_polyline_projects_to_segment_interior():
    polyline = np.array([[0.0, 0.0], [2.0, 0.0]])
    result = closest_point_on_polyline([0.75, 1.0], polyline)
    np.testing.assert_allclose(result, [0.75, 0.0])


def test_closest_point_on_polyline_clamps_to_endpoint():
    polyline = np.array([[0.0, 0.0], [2.0, 0.0]])
    result = closest_point_on_polyline([3.0, 1.0], polyline)
    np.testing.assert_allclose(result, [2.0, 0.0])


def test_closest_point_on_polyline_checks_all_segments():
    polyline = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    result = closest_point_on_polyline([1.5, 1.25], polyline)
    np.testing.assert_allclose(result, [1.0, 1.25])


def test_closest_point_on_polyline_allows_repeated_vertices():
    polyline = np.array([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    result = closest_point_on_polyline([1.0, 1.0], polyline)
    np.testing.assert_allclose(result, [1.0, 0.0])


def test_closest_point_on_polyline_single_point_returns_copy():
    polyline = np.array([[1.0, 2.0, 3.0]])
    result = closest_point_on_polyline([9.0, 9.0, 9.0], polyline)
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0])
    result[0] = 99.0
    assert polyline[0, 0] == 1.0


def test_closest_point_on_polyline_supports_arbitrary_dimension():
    polyline = np.array([[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
    result = closest_point_on_polyline([0.5, 1.0, 1.0, 1.0], polyline)
    np.testing.assert_allclose(result, [0.5, 0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("query_pt", "poly_pts", "message"),
    [
        ([[0.0, 0.0]], [[0.0, 0.0]], "must have shape"),
        ([0.0, 0.0], [0.0, 0.0], "two-dimensional"),
        ([0.0, 0.0], np.empty((0, 2)), "at least one point"),
        ([0.0, 0.0], [[0.0, 0.0, 0.0]], "must have shape"),
    ],
)
def test_closest_point_on_polyline_rejects_invalid_shapes(query_pt, poly_pts, message):
    with pytest.raises(ValueError, match=message):
        closest_point_on_polyline(query_pt, poly_pts)


@pytest.mark.parametrize(
    ("query_pt", "poly_pts", "message"),
    [
        ([np.nan, 0.0], [[0.0, 0.0]], "query_pt"),
        ([0.0, np.inf], [[0.0, 0.0]], "query_pt"),
        ([0.0, 0.0], [[np.nan, 0.0]], "poly_pts"),
        ([0.0, 0.0], [[0.0, np.inf]], "poly_pts"),
    ],
)
def test_closest_point_on_polyline_rejects_nonfinite_values(
    query_pt, poly_pts, message
):
    with pytest.raises(ValueError, match=message):
        closest_point_on_polyline(query_pt, poly_pts)
