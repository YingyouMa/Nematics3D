import numpy as np
import pytest

from nematics3d.datatypes import as_dimension_info


@pytest.mark.parametrize("value", [2, 2.5, np.int32(2), np.float32(2.5)])
def test_scalar_is_broadcast_to_xyz(value):
    result = as_dimension_info(value)

    np.testing.assert_array_equal(result, [value, value, value])
    assert result.shape == (3,)


def test_three_values_map_to_xyz_in_order():
    result = as_dimension_info([1, 2, 3])

    np.testing.assert_array_equal(result, [1, 2, 3])


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, [True, True, True]),
        (0, [False, False, False]),
        ([True, False, True], [True, False, True]),
        ([1, 0.0, np.bool_(True)], [True, False, True]),
    ],
)
def test_boolean_mode_accepts_booleans_and_numeric_zero_or_one(value, expected):
    result = as_dimension_info(value, is_bool=True)

    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.bool_


@pytest.mark.parametrize("value", [2, -1, 0.5, np.nan, np.inf, [1, 0, 2]])
def test_boolean_mode_rejects_other_numeric_values(value):
    with pytest.raises(ValueError, match="numeric 0/1"):
        as_dimension_info(value, is_bool=True)


def test_is_bool_option_must_be_boolean():
    with pytest.raises(TypeError, match="'is_bool' must be a boolean"):
        as_dimension_info(1, is_bool="yes")


def test_result_does_not_share_storage_with_input():
    source = np.array([1.0, 2.0, 3.0])

    result = as_dimension_info(source)
    result[0] = 10.0

    np.testing.assert_array_equal(source, [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "value",
    [
        [1],
        [1, 2],
        [1, 2, 3, 4],
        [[1, 2, 3]],
        np.ones((3, 1)),
    ],
)
def test_wrong_shapes_are_rejected(value):
    with pytest.raises(ValueError, match="one value or exactly three values"):
        as_dimension_info(value)


@pytest.mark.parametrize(
    "value",
    [
        "x",
        [1, "x", 3],
        1 + 2j,
        [1, 2, 3j],
    ],
)
def test_non_real_values_are_rejected(value):
    with pytest.raises(TypeError, match="real values"):
        as_dimension_info(value)
