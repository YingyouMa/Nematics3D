import numpy as np
import pytest

from nematics3d.datatypes import ColorRGB, as_ColorRGB, as_ColorRGB_array


def test_color_rgb_alias_describes_three_float_tuple():
    assert ColorRGB == tuple[float, float, float]


@pytest.mark.parametrize(
    "value, expected",
    [
        ((0.0, 0.5, 1.0), (0.0, 0.5, 1.0)),
        ([1, 0, 0.25], (1.0, 0.0, 0.25)),
        (np.array([0.1, 0.2, 0.3], dtype=np.float32), (0.1, 0.2, 0.3)),
    ],
)
def test_single_rgb_accepts_supported_sequence_types(value, expected):
    result = as_ColorRGB(value)

    assert isinstance(result, tuple)
    assert len(result) == 3
    np.testing.assert_allclose(result, expected)
    assert all(type(component) is float for component in result)


def test_single_rgb_result_does_not_share_storage_with_input():
    source = np.array([0.1, 0.2, 0.3])

    result = as_ColorRGB(source)
    source[:] = 0.0

    np.testing.assert_allclose(result, [0.1, 0.2, 0.3])


@pytest.mark.parametrize(
    "value",
    [
        0.5,
        "red",
        [0.1, 0.2],
        [[0.1, 0.2, 0.3]],
        np.ones((3, 1)),
    ],
)
def test_single_rgb_rejects_invalid_structure(value):
    with pytest.raises(ValueError):
        as_ColorRGB(value)


@pytest.mark.parametrize(
    "value, match",
    [
        ([0.0, 0.5, 1.1], "in \\[0, 1\\]"),
        ([-0.1, 0.5, 1.0], "in \\[0, 1\\]"),
        ([0.0, np.nan, 1.0], "finite"),
        ([0.0, np.inf, 1.0], "finite"),
        ([0.0, 1 + 0j, 1.0], "real numeric"),
        ([0.0, "x", 1.0], "real numeric"),
    ],
)
def test_single_rgb_rejects_invalid_values(value, match):
    with pytest.raises(ValueError, match=match):
        as_ColorRGB(value)


def test_single_rgb_normalization_uses_sum_of_powers_rule():
    result = as_ColorRGB([0.5, 0.5, 0.5], is_norm=True, norm_order=2)

    np.testing.assert_allclose(result, [2 / 3, 2 / 3, 2 / 3])


def test_single_rgb_near_zero_normalization_returns_zero():
    result = as_ColorRGB([0.01, 0.01, 0.01], is_norm=True, norm_order=2)

    np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])


def test_single_rgb_replacement_is_revalidated():
    result = as_ColorRGB([2.0, 0.0, 0.0], replace=[0.1, 0.2, 0.3], log_mode="none")
    np.testing.assert_allclose(result, [0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="replace must contain RGB values"):
        as_ColorRGB([2.0, 0.0, 0.0], replace=[2.0, 0.0, 0.0], log_mode="none")

    with pytest.raises(ValueError, match="replace must have shape"):
        as_ColorRGB([2.0, 0.0, 0.0], replace=[0.1, 0.2], log_mode="none")


def test_rgb_array_accepts_n_by_three_values_and_returns_float_copy():
    source = np.array([[0, 0.5, 1], [1, 0.25, 0]], dtype=np.float32)

    result = as_ColorRGB_array(source)

    np.testing.assert_allclose(result, source)
    assert result.shape == (2, 3)
    assert result.dtype == float
    assert not np.shares_memory(result, source)


def test_rgb_array_accepts_empty_n_by_three_array():
    result = as_ColorRGB_array(np.empty((0, 3)))

    assert result.shape == (0, 3)
    assert result.dtype == float


@pytest.mark.parametrize(
    "value",
    [
        1,
        "red",
        [0.1, 0.2, 0.3],
        np.ones((2, 2)),
        np.ones((2, 3, 1)),
    ],
)
def test_rgb_array_rejects_invalid_structure(value):
    with pytest.raises(ValueError):
        as_ColorRGB_array(value)


@pytest.mark.parametrize(
    "value, match",
    [
        ([[0.0, 0.5, 1.1]], "in \\[0, 1\\]"),
        ([[0.0, np.nan, 1.0]], "finite"),
        ([[0.0, np.inf, 1.0]], "finite"),
        ([[0.0, 1 + 0j, 1.0]], "real numeric"),
        ([[0.0, "x", 1.0]], "real numeric"),
    ],
)
def test_rgb_array_rejects_invalid_values(value, match):
    with pytest.raises(ValueError, match=match):
        as_ColorRGB_array(value)


def test_rgb_array_normalizes_each_row_independently():
    values = np.array([[0.5, 0.5, 0.5], [0.6, 0.8, 0.0], [0.01, 0.01, 0.01]])

    result = as_ColorRGB_array(values, is_norm=True, norm_order=2)

    np.testing.assert_allclose(
        result,
        [[2 / 3, 2 / 3, 2 / 3], [0.6, 0.8, 0.0], [0.0, 0.0, 0.0]],
    )


def test_rgb_array_single_color_replacement_is_broadcast_and_revalidated():
    values = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]

    result = as_ColorRGB_array(values, replace=[0.1, 0.2, 0.3], log_mode="none")
    np.testing.assert_allclose(result, [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])

    with pytest.raises(ValueError, match="replace must contain RGB values"):
        as_ColorRGB_array(values, replace=[2.0, 0.0, 0.0], log_mode="none")


def test_rgb_array_full_replacement_must_match_row_count():
    values = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    replacement = [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]

    result = as_ColorRGB_array(values, replace=replacement, log_mode="none")
    np.testing.assert_allclose(result, replacement)

    with pytest.raises(ValueError, match=r"shape \(3,\) or \(2, 3\)"):
        as_ColorRGB_array(values, replace=[[0.1, 0.2, 0.3]], log_mode="none")


def test_rgb_array_structural_errors_do_not_use_replacement():
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        as_ColorRGB_array([0.1, 0.2, 0.3], replace=[0.0, 0.0, 0.0], log_mode="none")
