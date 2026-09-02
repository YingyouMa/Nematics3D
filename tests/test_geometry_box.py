import numpy as np
import pytest

from nematics3d.geometry import get_box_corners, select_points_in_box


def test_get_box_corners_preserves_corner_order():
    result = get_box_corners(1, 2, 3)
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 2.0, 0.0],
            [1.0, 0.0, 3.0],
            [0.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ]
    )

    assert result.shape == (8, 3)
    assert np.issubdtype(result.dtype, np.floating)
    np.testing.assert_array_equal(result, expected)


def test_get_box_corners_accepts_zero_and_numpy_scalars():
    result = get_box_corners(np.int64(0), np.float64(2.5), np.int32(1))

    assert np.all(result[:, 0] == 0.0)
    np.testing.assert_array_equal(result[-1], [0.0, 2.5, 1.0])


@pytest.mark.parametrize(
    "lengths",
    [
        (-1.0, 1.0, 1.0),
        (np.nan, 1.0, 1.0),
        (np.inf, 1.0, 1.0),
        (True, 1.0, 1.0),
        (1.0 + 1.0j, 1.0, 1.0),
        ("1", 1.0, 1.0),
    ],
)
def test_get_box_corners_rejects_invalid_lengths(lengths):
    with pytest.raises((TypeError, ValueError)):
        get_box_corners(*lengths)


def test_select_points_in_axis_aligned_box_returns_mask():
    corners = get_box_corners(2.0, 3.0, 4.0)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [2.1, 2.0, 3.0],
            [-0.1, 0.0, 0.0],
        ]
    )

    selected, mask = select_points_in_box(points, corners, is_return_mask=True)

    np.testing.assert_array_equal(mask, [True, True, True, False, False])
    np.testing.assert_array_equal(selected, points[:3])


def test_select_points_in_rotated_translated_box():
    origin = np.array([3.0, -2.0, 1.0])
    axis1 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    axis2 = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)
    axis3 = np.array([0.0, 0.0, 1.0])
    lengths = np.array([2.0, 1.0, 3.0])
    corners = np.array(
        [
            origin,
            origin + lengths[0] * axis1,
            origin + lengths[1] * axis2,
            origin + lengths[2] * axis3,
        ]
    )
    points = np.array(
        [
            origin + axis1 + 0.5 * axis2 + axis3,
            origin + 2.2 * axis1 + 0.5 * axis2 + axis3,
        ]
    )

    selected = select_points_in_box(points, corners)

    np.testing.assert_allclose(selected, points[:1])


def test_select_points_in_box_none_selects_everything_and_empty_is_supported():
    points = np.array([[1, 2, 3], [4, 5, 6]])
    selected, mask = select_points_in_box(points, None, is_return_mask=True)
    np.testing.assert_array_equal(selected, points.astype(float))
    np.testing.assert_array_equal(mask, [True, True])

    empty, empty_mask = select_points_in_box([], None, is_return_mask=True)
    assert empty.shape == (0, 3)
    assert empty_mask.shape == (0,)


def test_select_points_in_box_uses_face_tolerance():
    corners = get_box_corners(1.0, 1.0, 1.0)
    points = np.array([[1.0 + 5e-10, 0.5, 0.5], [1.0 + 2e-9, 0.5, 0.5]])

    _, mask = select_points_in_box(points, corners, is_return_mask=True)

    np.testing.assert_array_equal(mask, [True, False])


def test_select_points_in_box_rejects_degenerate_or_skew_edges():
    degenerate = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    skew = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=float)

    with pytest.raises(ValueError):
        select_points_in_box([[0, 0, 0]], degenerate)
    with pytest.raises(ValueError):
        select_points_in_box([[0, 0, 0]], skew)


def test_select_points_in_box_rejects_invalid_inputs():
    corners = get_box_corners(1.0, 1.0, 1.0)

    with pytest.raises(ValueError):
        select_points_in_box([[0.0, 0.0]], corners)
    with pytest.raises(ValueError):
        select_points_in_box([[0.0, 0.0, 0.0]], corners[:3])
    selected, mask = select_points_in_box(
        [[0.0, 0.0, 0.0]],
        corners,
        is_return_mask=1,
    )
    np.testing.assert_allclose(selected, [[0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(mask, [True])
    with pytest.raises((TypeError, ValueError)):
        select_points_in_box([[0.0, 0.0, 0.0]], corners, atol=-1.0)
