import numpy as np
import pytest

from nematics3d.format import fmt_value


def test_fmt_value_formats_real_scalars_with_fixed_decimals():
    assert fmt_value(1.234) == "1.23"
    assert fmt_value(np.float32(-2.5), ndigits=3) == "-2.500"
    assert fmt_value(3, ndigits=0) == "3"


def test_fmt_value_formats_zero_dimensional_array_like_scalar():
    assert fmt_value(np.array(1.2345), ndigits=3) == "1.234"


def test_fmt_value_keeps_one_dimensional_array_on_one_line():
    values = np.arange(100, dtype=float) / 10
    text = fmt_value(values, ndigits=1)

    assert "\n" not in text
    assert text.startswith("[0.0, 0.1, 0.2")
    assert text.endswith("9.9]")


def test_fmt_value_preserves_multidimensional_array_structure():
    values = np.array([[1, 2], [3.456, -4]])

    assert fmt_value(values) == "[[1.00, 2.00],\n [3.46, -4.00]]"


def test_fmt_value_supports_nan_and_infinity():
    values = np.array([np.nan, np.inf, -np.inf])

    assert fmt_value(values) == "[nan, inf, -inf]"


@pytest.mark.parametrize("ndigits", [True, np.bool_(False), 1.5, "2", None])
def test_fmt_value_rejects_non_integer_ndigits(ndigits):
    with pytest.raises(TypeError, match="ndigits"):
        fmt_value(1.0, ndigits=ndigits)


def test_fmt_value_rejects_negative_ndigits():
    with pytest.raises(ValueError, match="non-negative"):
        fmt_value(1.0, ndigits=-1)


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        "1.23",
        1 + 2j,
        np.array(["1.0", "2.0"]),
        np.array([1 + 2j]),
    ],
)
def test_fmt_value_rejects_non_real_values(value):
    with pytest.raises(TypeError, match="real"):
        fmt_value(value)
