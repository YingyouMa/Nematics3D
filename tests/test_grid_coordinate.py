import numpy as np
import pytest

from nematics3d.grid import generate_coordinate_grid


def test_generate_coordinate_grid_identity_3d():
    grid = generate_coordinate_grid((2, 3, 4), (2, 3, 4))

    assert grid.shape == (2, 3, 4, 3)
    assert grid.dtype == float
    np.testing.assert_array_equal(grid[1, 2, 3], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(grid[0, 0, 0], [0.0, 0.0, 0.0])


def test_generate_coordinate_grid_resampling_coordinates():
    grid = generate_coordinate_grid((5, 7), (3, 4))

    assert grid.shape == (3, 4, 2)
    np.testing.assert_allclose(grid[:, 0, 0], [0.0, 2.0, 4.0])
    np.testing.assert_allclose(grid[0, :, 1], [0.0, 2.0, 4.0, 6.0])
    np.testing.assert_allclose(grid[-1, -1], [4.0, 6.0])


def test_generate_coordinate_grid_upsampling():
    grid = generate_coordinate_grid((3,), (5,))
    np.testing.assert_allclose(grid[:, 0], [0.0, 0.5, 1.0, 1.5, 2.0])


def test_generate_coordinate_grid_single_target_sample_uses_zero_coordinate():
    grid = generate_coordinate_grid((9, 5), (1, 3))

    assert grid.shape == (1, 3, 2)
    np.testing.assert_allclose(grid[0, :, 0], 0.0)
    np.testing.assert_allclose(grid[0, :, 1], [0.0, 2.0, 4.0])


def test_generate_coordinate_grid_accepts_numpy_integer_shapes():
    grid = generate_coordinate_grid(
        (np.int64(2), np.int32(3)),
        (np.int64(4), np.int32(5)),
    )
    assert grid.shape == (4, 5, 2)


@pytest.mark.parametrize(
    "shape_source, shape_target, error_type",
    [
        ((2, 3), (2, 3, 4), ValueError),
        ((), (), ValueError),
        ((0, 3), (2, 3), ValueError),
        ((-1, 3), (2, 3), ValueError),
        ((2.0, 3), (2, 3), TypeError),
        ((True, 3), (2, 3), TypeError),
        ((2, 3), (2, 0), ValueError),
        ((2, 3), (2.5, 3), TypeError),
        ("23", (2, 3), TypeError),
    ],
)
def test_generate_coordinate_grid_rejects_invalid_shapes(
    shape_source, shape_target, error_type
):
    with pytest.raises(error_type):
        generate_coordinate_grid(shape_source, shape_target)


def test_generate_coordinate_grid_returns_array_not_legacy_tuple():
    grid = generate_coordinate_grid((2, 2), (3, 3))
    assert isinstance(grid, np.ndarray)
    assert grid.shape == (3, 3, 2)
