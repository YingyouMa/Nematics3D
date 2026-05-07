import numpy as np
import pytest

from nematics3d.analysis import (
    FourierResult,
    act_correlation,
    act_fourier,
    act_filter,
    act_inverse,
)


def test_act_fourier_spectrum_scalar_axis_returns_expected_shape_and_k():
    values = np.ones((8, 4, 3))

    result = act_fourier(values, axes=0, spacing=0.5)
    spectrum = result.act_spectrum()

    assert result.axes == (0,)
    assert result.spacing == (0.5,)
    assert result.values_shape == values.shape
    assert spectrum.shape == (5,)
    np.testing.assert_allclose(result.k_axes[0], 2 * np.pi * np.fft.rfftfreq(8, 0.5))


def test_act_fourier_spectrum_handles_components_in_parallel():
    values = np.ones((8, 4, 3, 5))

    spectrum = act_fourier(values, axes=0, spacing=1.0).act_spectrum()

    assert spectrum.shape == (5, 5)


def test_act_fourier_normalized_spectrum_preserves_mean_square():
    x = np.arange(8, dtype=float)
    values = np.sin(2 * np.pi * x / 8)[:, None, None]

    spectrum = act_fourier(values, axes=0, spacing=1.0).act_spectrum(
        is_normalized=True,
    )

    np.testing.assert_allclose(spectrum.sum(axis=0), np.mean(values**2), atol=1e-12)


def test_act_fourier_fft_output_returns_coefficients():
    values = np.ones((8, 4, 3))

    result = act_fourier(
        values,
        axes=0,
        spacing=1.0,
    )

    assert isinstance(result, FourierResult)
    assert result.fft_values.shape == (5, 4, 3)
    assert result.fft_values[0, 0, 0] == 8.0


def test_act_fourier_multi_axis_returns_k_axes_in_transform_order():
    values = np.ones((8, 6, 4, 3))

    result = act_fourier(values, axes=(0, 2), spacing=(0.5, 2.0))
    spectrum = result.act_spectrum()

    assert result.axes == (0, 2)
    assert spectrum.shape == (8, 3, 3)
    np.testing.assert_allclose(result.k_axes[0], 2 * np.pi * np.fft.fftfreq(8, 0.5))
    np.testing.assert_allclose(result.k_axes[1], 2 * np.pi * np.fft.rfftfreq(4, 2.0))


def test_act_fourier_preserves_mean_by_default():
    values = np.ones((8, 4, 3, 2))
    values[..., 1] = 2.0

    result = act_fourier(values, axes=0, spacing=1.0)

    assert result.fft_values[0, 0, 0, 0] == 8.0
    assert result.fft_values[0, 0, 0, 1] == 16.0


def test_fourier_result_mean_subtracted_values_supports_spatial_mode():
    values = np.ones((4, 3, 2, 2))
    values[..., 1] = 2.0
    result = act_fourier(values, axes=0, spacing=1.0)

    centered = result.act_mean_subtracted_values(mode="spatial")

    np.testing.assert_allclose(centered, 0.0, atol=1e-12)


def test_fourier_result_mean_subtracted_values_supports_axes_mode():
    x = np.arange(4, dtype=float)[:, None, None]
    y_offset = np.arange(3, dtype=float)[None, :, None]
    values = x + y_offset
    result = act_fourier(values, axes=0, spacing=1.0)

    centered = result.act_mean_subtracted_values(mode="axes")
    expected = values - values.mean(axis=0, keepdims=True)

    np.testing.assert_allclose(centered, expected, atol=1e-12)


def test_act_fourier_rejects_complex_values():
    values = np.ones((8, 4, 3), dtype=complex)

    with pytest.raises(TypeError, match="real-valued"):
        act_fourier(values, axes=0, spacing=1.0)


def test_act_filter_keeps_requested_k_band():
    values = np.zeros((8, 4, 3))
    values[0, :, :] = 1.0
    result = act_fourier(
        values,
        axes=0,
        spacing=1.0,
    )
    k = result.k_axes[0]

    filtered = act_filter(result, k_min=k[2] - 1e-12, k_max=k[2] + 1e-12)
    filtered_method = result.act_filter(k_min=k[2] - 1e-12, k_max=k[2] + 1e-12)

    assert isinstance(filtered, FourierResult)
    assert filtered is not result
    assert filtered_method is not result
    np.testing.assert_allclose(filtered.fft_values[2], result.fft_values[2])
    kept = np.zeros_like(result.fft_values)
    kept[2] = result.fft_values[2]
    np.testing.assert_allclose(filtered.fft_values, kept)
    np.testing.assert_allclose(filtered_method.fft_values, kept)


def test_act_filter_broadcasts_over_untransformed_spatial_axes():
    values = np.zeros((8, 4, 3, 2))
    values[0, :, 0, :] = 1.0
    result = act_fourier(
        values,
        axes=(0, 2),
        spacing=(1.0, 1.0),
    )

    filtered = act_filter(result, k_max=0.0)

    assert filtered.fft_values.shape == result.fft_values.shape


def test_act_filter_without_bounds_returns_same_result():
    result = act_fourier(np.ones((8, 4, 3)), axes=0, spacing=1.0)

    assert act_filter(result) is result
    assert result.act_filter() is result


def test_act_filter_rejects_invalid_bounds():
    result = act_fourier(np.ones((8, 4, 3)), axes=0, spacing=1.0)

    with pytest.raises(ValueError, match="non-negative"):
        act_filter(result, k_min=-1.0)
    with pytest.raises(ValueError, match="less than or equal"):
        act_filter(result, k_min=2.0, k_max=1.0)


def test_act_inverse_recovers_unpadded_values():
    x = np.arange(8, dtype=float)
    values = np.sin(2 * np.pi * x / 8)[:, None, None]
    result = act_fourier(
        values,
        axes=0,
        spacing=1.0,
    )

    recovered = act_inverse(result)
    recovered_method = result.act_inverse()

    np.testing.assert_allclose(recovered, values, atol=1e-12)
    np.testing.assert_allclose(recovered_method, values, atol=1e-12)


def test_act_correlation_returns_periodic_autocorrelation():
    x = np.arange(8, dtype=float)
    values = np.sin(2 * np.pi * x / 8)[:, None, None]
    result = act_fourier(
        values,
        axes=0,
        spacing=1.0,
    )

    correlation = act_correlation(result)
    correlation_method = result.act_correlation()
    expected = 0.5 * np.cos(2 * np.pi * x / 8)[:, None, None]

    np.testing.assert_allclose(correlation, expected, atol=1e-12)
    np.testing.assert_allclose(correlation_method, expected, atol=1e-12)


def test_act_inverse_padding_interpolates_shape_and_amplitude():
    x = np.arange(8, dtype=float)
    values = np.sin(2 * np.pi * x / 8)[:, None, None]
    result = act_fourier(
        values,
        axes=0,
        spacing=1.0,
    )

    interpolated = act_inverse(result, padding_num=8)
    expected_x = np.arange(16, dtype=float) * 8 / 16
    expected = np.sin(2 * np.pi * expected_x / 8)[:, None, None]

    assert interpolated.shape == (16, 1, 1)
    np.testing.assert_allclose(interpolated, expected, atol=1e-12)


def test_act_inverse_padding_supports_multi_axis_results():
    values = np.ones((4, 3, 5, 2))
    result = act_fourier(
        values,
        axes=(0, 2),
        spacing=(1.0, 1.0),
    )

    interpolated = act_inverse(result, padding_num=(2, 4))

    assert interpolated.shape == (6, 3, 9, 2)
    np.testing.assert_allclose(interpolated, 1.0, atol=1e-12)
