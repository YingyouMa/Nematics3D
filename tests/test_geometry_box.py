import numpy as np
import pytest

from nematics3d.geometry import get_box_corners


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
