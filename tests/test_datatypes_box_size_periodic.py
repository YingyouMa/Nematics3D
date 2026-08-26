import numpy as np
import pytest

from nematics3d.datatypes import as_box_size_periodic


@pytest.mark.parametrize(
    "value, expected",
    [
        (4, [4.0, 4.0, 4.0]),
        (2.5, [2.5, 2.5, 2.5]),
        (np.inf, [np.inf, np.inf, np.inf]),
        ([4, np.inf, 2.5], [4.0, np.inf, 2.5]),
    ],
)
def test_as_box_size_periodic_normalizes_shared_and_xyz_values(value, expected):
    result = as_box_size_periodic(value)

    np.testing.assert_array_equal(result, expected)
    assert result.shape == (3,)
    assert result.dtype == float


def test_as_box_size_periodic_returns_an_independent_array():
    source = np.array([4.0, 5.0, np.inf])

    result = as_box_size_periodic(source)
    result[0] = 10.0

    np.testing.assert_array_equal(source, [4.0, 5.0, np.inf])


@pytest.mark.parametrize("value", [True, [1, False, 2]])
def test_as_box_size_periodic_rejects_boolean_values(value):
    with pytest.raises(TypeError, match="not boolean"):
        as_box_size_periodic(value)


@pytest.mark.parametrize(
    "value, match",
    [
        (0, "must be positive"),
        (-1, "must be positive"),
        ([1, 0, 2], "must be positive"),
        (np.nan, "must not contain NaN"),
        (-np.inf, "only positive infinity"),
    ],
)
def test_as_box_size_periodic_rejects_invalid_sizes(value, match):
    with pytest.raises(ValueError, match=match):
        as_box_size_periodic(value)


@pytest.mark.parametrize("value", [[1, 2], [[1, 2, 3]], "periodic"])
def test_as_box_size_periodic_rejects_invalid_structure_or_type(value):
    with pytest.raises((TypeError, ValueError)):
        as_box_size_periodic(value)
