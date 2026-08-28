import numpy as np
import pytest

from nematics3d.datatypes import as_grid_shape


@pytest.mark.parametrize(
    "input_data, expected",
    [
        ((2, 3, 4), (2, 3, 4)),
        ([2, 3], (2, 3)),
        (np.array([2, 3], dtype=np.int64), (2, 3)),
        ((np.int32(2), np.int64(3)), (2, 3)),
    ],
)
def test_as_grid_shape_accepts_ordered_integer_iterables(input_data, expected):
    result = as_grid_shape(input_data)

    assert result == expected
    assert isinstance(result, tuple)
    assert all(type(value) is int for value in result)


def test_as_grid_shape_accepts_generator():
    result = as_grid_shape(value for value in [2, 3, 4])

    assert result == (2, 3, 4)


def test_as_grid_shape_strict_3d_accepts_exactly_three_dimensions():
    assert as_grid_shape([2, 3, 4], is_strict_3d=True) == (2, 3, 4)


@pytest.mark.parametrize("input_data", [(2,), (2, 3), (2, 3, 4, 5)])
def test_as_grid_shape_strict_3d_rejects_other_dimensionalities(input_data):
    with pytest.raises(ValueError, match="exactly three dimensions"):
        as_grid_shape(input_data, is_strict_3d=True)


@pytest.mark.parametrize(
    "input_data",
    [
        {2, 3},
        frozenset({2, 3}),
        {2: "x", 3: "y"},
        "23",
        b"23",
        3,
        None,
    ],
)
def test_as_grid_shape_rejects_non_ordered_or_non_iterable_inputs(input_data):
    with pytest.raises(TypeError, match="ordered iterable"):
        as_grid_shape(input_data)


@pytest.mark.parametrize(
    "input_data, error_type, message",
    [
        ((), ValueError, "at least one dimension"),
        ((0, 3), ValueError, "must be positive"),
        ((-1, 3), ValueError, "must be positive"),
        ((2.0, 3), TypeError, "must be an integer"),
        ((True, 3), TypeError, "must be an integer"),
        ((np.bool_(False), 3), TypeError, "must be an integer"),
        ((2 + 0j, 3), TypeError, "must be an integer"),
    ],
)
def test_as_grid_shape_rejects_invalid_dimensions(input_data, error_type, message):
    with pytest.raises(error_type, match=message):
        as_grid_shape(input_data, name="sample shape")


def test_as_grid_shape_rejects_invalid_strict_flag():
    with pytest.raises(TypeError):
        as_grid_shape((2, 3, 4), is_strict_3d="yes")
