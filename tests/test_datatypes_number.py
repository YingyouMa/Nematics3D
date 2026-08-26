import numpy as np
import pytest

from nematics3d.datatypes import as_number, as_value_range


@pytest.mark.parametrize("value", [1, 1.5, np.int64(2), np.float32(2.5)])
def test_as_number_returns_a_python_float(value):
    result = as_number(value)

    assert type(result) is float
    assert result == float(value)


@pytest.mark.parametrize("value", [1, 1.0, np.int64(2), np.float32(2.0)])
def test_as_number_integer_mode_returns_a_python_int(value):
    result = as_number(value, is_integer=True)

    assert type(result) is int
    assert result == int(value)


@pytest.mark.parametrize("value", [True, False, np.bool_(True)])
def test_as_number_rejects_boolean_input(value):
    with pytest.raises(TypeError, match="not boolean"):
        as_number(value)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_as_number_rejects_nonfinite_values_by_default(value):
    with pytest.raises(ValueError):
        as_number(value)


def test_as_number_can_explicitly_allow_nan_and_infinity():
    assert np.isnan(as_number(np.nan, is_nan_allowed=True))
    assert as_number(np.inf, is_infinite_allowed=True) == np.inf
    assert as_number(-np.inf, is_infinite_allowed=True) == -np.inf


def test_as_number_enforces_an_inclusive_value_range():
    assert as_number(0, value_range=(0, 1)) == 0.0
    assert as_number(1, value_range=(0, 1)) == 1.0

    with pytest.raises(ValueError, match="inclusive range"):
        as_number(2, value_range=(0, 1))


def test_as_number_clips_to_the_value_range():
    assert as_number(-1, value_range=(0, 1), is_clipped=True) == 0.0
    assert as_number(2, value_range=(0, 1), is_clipped=True) == 1.0


def test_as_number_validates_the_replacement():
    assert as_number("invalid", value_range=(0, 1), replace=0.5, log_mode="none") == 0.5

    with pytest.raises(ValueError, match="inclusive range"):
        as_number("invalid", value_range=(0, 1), replace=2, log_mode="none")


@pytest.mark.parametrize(
    ("value_range", "error_type"),
    [
        ((0,), ValueError),
        ((1, 0), ValueError),
        ((0, np.nan), ValueError),
        ((0, 1 + 1j), TypeError),
        (("low", "high"), TypeError),
    ],
)
def test_as_value_range_rejects_invalid_configuration(value_range, error_type):
    with pytest.raises(error_type):
        as_value_range(value_range)


def test_as_number_rejects_non_boolean_options():
    with pytest.raises(TypeError, match="is_integer"):
        as_number(1, is_integer="yes")


def test_integer_clipping_requires_integer_valued_bounds():
    with pytest.raises(ValueError, match="integer-valued"):
        as_number(
            1,
            is_integer=True,
            value_range=(0.5, 2.5),
            is_clipped=True,
        )
