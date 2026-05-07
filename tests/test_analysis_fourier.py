import numpy as np
import pytest

from nematics3d.analysis import (
    FourierResult,
    SpectrumResult,
    field_fourier,
    field_fourier_filter,
    field_inverse_fourier,
)


def test_field_fourier_spectrum_scalar_axis_returns_expected_shape_and_k():
    values = np.ones((8, 4, 3))

    result = field_fourier(values, axes=0, spacing=0.5)

    assert isinstance(result, SpectrumResult)
    assert result.axes == (0,)
    assert result.spacing == (0.5,)
    assert result.values_shape == values.shape
    assert result.spectrum.shape == (5,)
    np.testing.assert_allclose(result.k_axes[0], 2 * np.pi * np.fft.rfftfreq(8, 0.5))


def test_field_fourier_spectrum_handles_components_in_parallel():
    values = np.ones((8, 4, 3, 5))

    result = field_fourier(values, axes=0, spacing=1.0)

    assert result.spectrum.shape == (5, 5)


def test_field_fourier_spectrum_component_sum_reduces_trailing_axes():
    values = np.ones((8, 4, 3, 2, 2))

    result = field_fourier(
        values,
        axes=0,
        spacing=1.0,
        component_mode="sum",
    )

    assert result.spectrum.shape == (5,)


def test_field_fourier_fft_output_returns_coefficients():
    values = np.ones((8, 4, 3))

    result = field_fourier(
        values,
        axes=0,
        spacing=1.0,
        output="fft",
        is_subtract_mean=False,
    )

    assert isinstance(result, FourierResult)
    assert result.fft_values.shape == (5, 4, 3)
    assert result.fft_values[0, 0, 0] == 8.0
    assert result.is_mean_subtracted is False


def test_field_fourier_multi_axis_returns_k_axes_in_transform_order():
    values = np.ones((8, 6, 4, 3))

    result = field_fourier(values, axes=(0, 2), spacing=(0.5, 2.0))

    assert result.axes == (0, 2)
    assert result.spectrum.shape == (8, 3, 3)
    np.testing.assert_allclose(result.k_axes[0], 2 * np.pi * np.fft.fftfreq(8, 0.5))
    np.testing.assert_allclose(result.k_axes[1], 2 * np.pi * np.fft.rfftfreq(4, 2.0))


def test_field_fourier_subtracts_each_component_spatial_mean():
    values = np.ones((8, 4, 3, 2))
    values[..., 1] = 2.0

    result = field_fourier(values, axes=0, spacing=1.0, output="fft")

    np.testing.assert_allclose(result.fft_values, 0.0)
    assert result.is_mean_subtracted is True


def test_field_fourier_rejects_complex_values():
    values = np.ones((8, 4, 3), dtype=complex)

    with pytest.raises(TypeError, match="real-valued"):
        field_fourier(values, axes=0, spacing=1.0)


def test_field_fourier_filter_keeps_requested_k_band():
    values = np.zeros((8, 4, 3))
    values[0, :, :] = 1.0
    result = field_fourier(
        values,
        axes=0,
        spacing=1.0,
        output="fft",
        is_subtract_mean=False,
    )
    k = result.k_axes[0]

    filtered = field_fourier_filter(result, k_min=k[2] - 1e-12, k_max=k[2] + 1e-12)

    assert isinstance(filtered, FourierResult)
    assert filtered is not result
    np.testing.assert_allclose(filtered.fft_values[2], result.fft_values[2])
    kept = np.zeros_like(result.fft_values)
    kept[2] = result.fft_values[2]
    np.testing.assert_allclose(filtered.fft_values, kept)


def test_field_fourier_filter_broadcasts_over_untransformed_spatial_axes():
    values = np.zeros((8, 4, 3, 2))
    values[0, :, 0, :] = 1.0
    result = field_fourier(
        values,
        axes=(0, 2),
        spacing=(1.0, 1.0),
        output="fft",
        is_subtract_mean=False,
    )

    filtered = field_fourier_filter(result, k_max=0.0)

    assert filtered.fft_values.shape == result.fft_values.shape


def test_field_fourier_filter_without_bounds_returns_same_result():
    result = field_fourier(np.ones((8, 4, 3)), axes=0, spacing=1.0, output="fft")

    assert field_fourier_filter(result) is result


def test_field_fourier_filter_rejects_invalid_bounds():
    result = field_fourier(np.ones((8, 4, 3)), axes=0, spacing=1.0, output="fft")

    with pytest.raises(ValueError, match="non-negative"):
        field_fourier_filter(result, k_min=-1.0)
    with pytest.raises(ValueError, match="less than or equal"):
        field_fourier_filter(result, k_min=2.0, k_max=1.0)


def test_field_inverse_fourier_recovers_unpadded_values():
    x = np.arange(8, dtype=float)
    values = np.sin(2 * np.pi * x / 8)[:, None, None]
    result = field_fourier(
        values,
        axes=0,
        spacing=1.0,
        output="fft",
        is_subtract_mean=False,
    )

    recovered = field_inverse_fourier(result)

    np.testing.assert_allclose(recovered, values, atol=1e-12)


def test_field_inverse_fourier_padding_interpolates_shape_and_amplitude():
    x = np.arange(8, dtype=float)
    values = np.sin(2 * np.pi * x / 8)[:, None, None]
    result = field_fourier(
        values,
        axes=0,
        spacing=1.0,
        output="fft",
        is_subtract_mean=False,
    )

    interpolated = field_inverse_fourier(result, padding_num=8)
    expected_x = np.arange(16, dtype=float) * 8 / 16
    expected = np.sin(2 * np.pi * expected_x / 8)[:, None, None]

    assert interpolated.shape == (16, 1, 1)
    np.testing.assert_allclose(interpolated, expected, atol=1e-12)


def test_field_inverse_fourier_padding_supports_multi_axis_results():
    values = np.ones((4, 3, 5, 2))
    result = field_fourier(
        values,
        axes=(0, 2),
        spacing=(1.0, 1.0),
        output="fft",
        is_subtract_mean=False,
    )

    interpolated = field_inverse_fourier(result, padding_num=(2, 4))

    assert interpolated.shape == (6, 3, 9, 2)
    np.testing.assert_allclose(interpolated, 1.0, atol=1e-12)
