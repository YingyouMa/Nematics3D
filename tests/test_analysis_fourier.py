import numpy as np
import pytest

from nematics3d.analysis import (
    CorrelationResult,
    DistanceCorrelationResult,
    FourierResult,
    RadialSpectrumResult,
    act_correlation,
    act_correlation_values,
    act_distance,
    act_fourier,
    act_filter,
    act_inverse,
    act_radial_spectrum,
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


def test_act_radial_spectrum_1d_returns_non_negative_k():
    values = np.zeros((8, 1, 1))
    result = act_fourier(values, axes=0, spacing=1.0)
    fft_values = np.array([1, 2, 3, 4, 5], dtype=complex)[:, None, None]
    result = FourierResult(
        k_axes=result.k_axes,
        fft_values=fft_values,
        values_shape=values.shape,
        axes=result.axes,
        spacing=result.spacing,
    )

    radial = act_radial_spectrum(result)
    radial_method = result.act_radial_spectrum()

    assert isinstance(radial, RadialSpectrumResult)
    np.testing.assert_allclose(radial.k_values, result.k_axes[0])
    np.testing.assert_allclose(radial.spectrum_values.ravel(), [1, 4, 9, 16, 25])
    np.testing.assert_array_equal(radial.count_values, np.ones(5, dtype=int))
    np.testing.assert_allclose(radial.anisotropy_values, 0.0)
    np.testing.assert_allclose(
        radial_method.spectrum_values,
        radial.spectrum_values,
    )


def test_act_radial_spectrum_2d_averages_k_shells_and_reports_anisotropy():
    values = np.zeros((3, 1, 3))
    result = act_fourier(values, axes=(0, 2), spacing=(2 * np.pi / 3, 2 * np.pi / 3))
    fft_values = np.zeros_like(result.fft_values)
    fft_values[0, 0, 0] = 0.0
    fft_values[1, 0, 0] = 1.0
    fft_values[2, 0, 0] = 2.0
    fft_values[0, 0, 1] = 3.0
    result = FourierResult(
        k_axes=result.k_axes,
        fft_values=fft_values,
        values_shape=values.shape,
        axes=result.axes,
        spacing=result.spacing,
    )

    radial = result.act_radial_spectrum(k_max=np.sqrt(2.0), bin_width=1.0)

    np.testing.assert_allclose(
        radial.k_values,
        np.array([0.5, 0.5 * (1.0 + np.sqrt(2.0))]),
    )
    np.testing.assert_allclose(radial.spectrum_values, np.array([0.0, 14.0 / 5.0]))
    np.testing.assert_allclose(
        radial.anisotropy_values,
        np.array([0.0, np.std([1.0, 4.0, 9.0, 0.0, 0.0]) / np.sqrt(98.0 / 5.0)]),
    )
    np.testing.assert_array_equal(radial.count_values, np.array([1, 5]))


def test_act_radial_spectrum_validates_k_max_and_bin_width():
    result = act_fourier(np.ones((8, 4, 3)), axes=0, spacing=1.0)

    with pytest.raises(ValueError, match="k_max"):
        act_radial_spectrum(result, k_max=10.0)
    with pytest.raises(ValueError, match="bin_width"):
        act_radial_spectrum(result, k_max=2.0, bin_width=3.0)


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


def test_fourier_result_show_readable_attrs_returns_field_docs():
    result = act_fourier(np.ones((8, 4, 3)), axes=0, spacing=1.0)

    output = result.show_readable_attrs(is_return=True)

    assert "- k_axes" in output
    assert "Angular wave-number coordinate arrays" in output
    assert "Angular wave-number coordinate arrays" in result.show_attr_doc(
        "k_axes",
        is_return=True,
    )


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
    expected = 0.5 * np.cos(2 * np.pi * x / 8)

    assert isinstance(correlation, CorrelationResult)
    output = correlation.show_readable_attrs(is_return=True)

    assert "- lag_axes" in output
    assert "Real-space periodic lag coordinate arrays" in output
    assert "Real-space periodic lag coordinate arrays" in correlation.show_attr_doc(
        "lag_axes",
        is_return=True,
    )
    np.testing.assert_allclose(correlation.lag_axes[0], np.fft.fftfreq(8) * 8)
    np.testing.assert_allclose(correlation.mean_values, 0.0, atol=1e-12)
    np.testing.assert_allclose(correlation.correlation_values, expected, atol=1e-12)
    np.testing.assert_allclose(
        correlation_method.correlation_values,
        expected,
        atol=1e-12,
    )


def test_act_correlation_supports_mean_subtraction():
    x = np.arange(8, dtype=float)
    values = 3.0 + np.sin(2 * np.pi * x / 8)[:, None, None]
    result = act_fourier(
        values,
        axes=0,
        spacing=1.0,
    )

    correlation = act_correlation(result)
    expected = 0.5 * np.cos(2 * np.pi * x / 8)

    np.testing.assert_allclose(
        correlation.act_values(is_subtract_mean=True),
        expected,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        act_correlation_values(correlation, is_subtract_mean=True),
        expected,
        atol=1e-12,
    )
    np.testing.assert_allclose(correlation.mean_values, 3.0, atol=1e-12)


def test_act_correlation_supports_normalization():
    x = np.arange(8, dtype=float)
    values = 2.0 * np.sin(2 * np.pi * x / 8)[:, None, None]
    result = act_fourier(
        values,
        axes=0,
        spacing=1.0,
    )

    correlation = act_correlation(result)
    expected = np.cos(2 * np.pi * x / 8)

    np.testing.assert_allclose(
        correlation.act_values(is_normalized=True),
        expected,
        atol=1e-12,
    )


def test_act_correlation_supports_mean_subtraction_and_normalization():
    x = np.arange(8, dtype=float)
    values = 3.0 + 2.0 * np.sin(2 * np.pi * x / 8)[:, None, None]
    result = act_fourier(
        values,
        axes=0,
        spacing=1.0,
    )

    correlation = result.act_correlation()
    expected = np.cos(2 * np.pi * x / 8)

    np.testing.assert_allclose(
        correlation.act_values(is_subtract_mean=True, is_normalized=True),
        expected,
        atol=1e-12,
    )


def test_act_distance_1d_groups_positive_and_negative_lags():
    x = np.arange(8, dtype=float)
    values = np.sin(2 * np.pi * x / 8)[:, None, None]
    correlation = act_fourier(values, axes=0, spacing=1.0).act_correlation()

    distance = act_distance(correlation)
    distance_method = correlation.act_distance()
    expected_r = np.arange(5, dtype=float)
    expected_values = 0.5 * np.cos(2 * np.pi * expected_r / 8)

    assert isinstance(distance, DistanceCorrelationResult)
    np.testing.assert_allclose(distance.r_values, expected_r)
    np.testing.assert_allclose(distance.correlation_values, expected_values, atol=1e-12)
    np.testing.assert_allclose(
        distance_method.correlation_values,
        expected_values,
        atol=1e-12,
    )
    np.testing.assert_array_equal(distance.count_values, np.array([1, 2, 2, 2, 1]))
    np.testing.assert_allclose(distance.anisotropy_values, 0.0, atol=1e-12)


def test_act_distance_2d_averages_radial_bins_and_reports_anisotropy():
    lag_axis = np.array([0.0, 1.0, -1.0])
    lag_x, lag_y = np.meshgrid(lag_axis, lag_axis, indexing="ij")
    correlation_values = lag_x**2 + lag_y**2
    correlation = CorrelationResult(
        lag_axes=(lag_axis, lag_axis),
        correlation_values=correlation_values,
        mean_values=np.array(0.0),
        values_shape=(3, 3, 1),
        axes=(0, 1),
        spacing=(1.0, 1.0),
    )

    distance = correlation.act_distance(r_max=1.5, bin_width=1.0)

    np.testing.assert_allclose(distance.r_values, np.array([0.5, 1.25]))
    np.testing.assert_allclose(distance.correlation_values, np.array([0.0, 1.5]))
    np.testing.assert_allclose(distance.std_values, np.array([0.0, 0.5]))
    np.testing.assert_allclose(
        distance.anisotropy_values,
        np.array([0.0, 0.5 / np.sqrt(2.5)]),
    )
    np.testing.assert_array_equal(distance.count_values, np.array([1, 8]))


def test_act_distance_validates_radius_and_bin_width():
    correlation = act_fourier(np.ones((8, 4, 3)), axes=0, spacing=1.0).act_correlation()

    with pytest.raises(ValueError, match="r_max"):
        act_distance(correlation, r_max=5.0)
    with pytest.raises(ValueError, match="bin_width"):
        act_distance(correlation, r_max=2.0, bin_width=3.0)


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
