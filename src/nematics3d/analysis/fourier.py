"""Fourier helpers for real-valued lattice fields."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from typing import ClassVar

import numpy as np

from nematics3d.classes.result_base import ResultBase
from nematics3d.datatypes import as_Number
from nematics3d.datatypes import as_real_lattice_field


@dataclass(slots=True, frozen=True, repr=False)
class FourierResult(ResultBase):
    """Container returned by :func:`act_fourier`."""

    __result_name__: ClassVar[str] = "Fourier transform"
    __field_docs__: ClassVar[dict[str, str]] = {
        "k_axes": ("Angular wave-number coordinate arrays for the transformed axes."),
        "fft_values": (
            "Complex Fourier coefficients from np.fft.rfftn on the selected axes."
        ),
        "values_shape": "Shape of the original real-space lattice field.",
        "axes": "Spatial lattice axes included in the Fourier transform.",
        "spacing": "Real-space spacing associated with each transformed axis.",
    }

    k_axes: tuple[np.ndarray, ...]
    fft_values: np.ndarray
    values_shape: tuple[int, ...]
    axes: tuple[int, ...]
    spacing: tuple[float, ...]

    def act_filter(
        self,
        *,
        k_min: float | None = None,
        k_max: float | None = None,
    ) -> "FourierResult":
        """Return a copy with coefficients outside the requested k-band zeroed."""
        return act_filter(self, k_min=k_min, k_max=k_max)

    def act_inverse(
        self,
        *,
        padding_num: int | Sequence[int] = 0,
    ) -> np.ndarray:
        """Invert this Fourier result, optionally padding for interpolation."""
        return act_inverse(self, padding_num=padding_num)

    def act_spectrum(self, *, is_normalized: bool = False) -> np.ndarray:
        """Return the Fourier power spectrum derived from this transform."""
        return _spectrum_from_fourier(self, is_normalized=is_normalized)

    def act_radial_spectrum(
        self,
        *,
        is_normalized: bool = False,
        k_max: float | None = None,
        bin_width: float | None = None,
    ) -> "RadialSpectrumResult":
        """Return the radial wave-number averaged Fourier power spectrum."""
        return act_radial_spectrum(
            self,
            is_normalized=is_normalized,
            k_max=k_max,
            bin_width=bin_width,
        )

    def act_correlation(self) -> "CorrelationResult":
        """Return the periodic autocorrelation from this Fourier result."""
        return act_correlation(self)

    def act_relaxation_length(
        self,
        *,
        threshold: float = np.exp(-1),
        fit_head_factor: float | None = None,
        fit_tail_factor: float | None = 10.0,
        max_iteration_num: int = 20,
        fit_tolerance: float = 1e-3,
        min_fit_point_num: int = 4,
    ):
        """Return the relaxation length estimated from this Fourier result."""
        if len(self.axes) != 1:
            raise ValueError(
                "`FourierResult.act_relaxation_length()` currently supports "
                "only one-dimensional Fourier transforms."
            )

        correlation_result = self.act_correlation()
        correlation_values = correlation_result.correlation_values
        if correlation_values.ndim != 1:
            raise ValueError(
                "The Fourier correlation must be one-dimensional after "
                "averaging untransformed spatial axes."
            )

        max_lag_index = correlation_values.shape[0] // 2
        coordinate_axis = np.abs(correlation_result.lag_axes[0][: max_lag_index + 1])

        # Local import avoids a module-level cycle between Fourier and
        # relaxation helpers while keeping the result method convenient.
        from nematics3d.analysis.relaxation import act_relaxation_length

        return act_relaxation_length(
            correlation_values[: max_lag_index + 1],
            coordinate_axis=coordinate_axis,
            threshold=threshold,
            fit_head_factor=fit_head_factor,
            fit_tail_factor=fit_tail_factor,
            max_iteration_num=max_iteration_num,
            fit_tolerance=fit_tolerance,
            min_fit_point_num=min_fit_point_num,
        )

    def act_mean_subtracted_values(
        self,
        *,
        mode: str = "spatial",
        padding_num: int | Sequence[int] = 0,
    ) -> np.ndarray:
        """Return inverse-transformed values after subtracting a selected mean."""
        return act_mean_subtracted_values(
            self,
            mode=mode,
            padding_num=padding_num,
        )


@dataclass(slots=True, frozen=True, repr=False)
class CorrelationResult(ResultBase):
    """Container returned by :func:`act_correlation`."""

    __result_name__: ClassVar[str] = "Correlation"
    __field_docs__: ClassVar[dict[str, str]] = {
        "lag_axes": (
            "Real-space periodic lag coordinate arrays for the correlation axes."
        ),
        "correlation_values": (
            "Raw periodic autocorrelation values on the lag coordinate grid."
        ),
        "mean_values": (
            "Full spatial mean of the original field, stored per trailing component."
        ),
        "values_shape": "Shape of the original real-space lattice field.",
        "axes": "Spatial lattice axes represented as correlation lag axes.",
        "spacing": "Real-space spacing associated with each correlation axis.",
    }

    lag_axes: tuple[np.ndarray, ...]
    correlation_values: np.ndarray
    mean_values: np.ndarray
    values_shape: tuple[int, ...]
    axes: tuple[int, ...]
    spacing: tuple[float, ...]

    def act_values(
        self,
        *,
        is_subtract_mean: bool = False,
        is_normalized: bool = False,
    ) -> np.ndarray:
        """Return correlation values with optional mean subtraction/normalization."""
        return act_correlation_values(
            self,
            is_subtract_mean=is_subtract_mean,
            is_normalized=is_normalized,
        )

    def act_distance(
        self,
        *,
        is_subtract_mean: bool = False,
        is_normalized: bool = False,
        r_max: float | None = None,
        bin_width: float | None = None,
    ) -> "DistanceCorrelationResult":
        """Return the radial distance-averaged correlation."""
        return act_distance(
            self,
            is_subtract_mean=is_subtract_mean,
            is_normalized=is_normalized,
            r_max=r_max,
            bin_width=bin_width,
        )


@dataclass(slots=True, frozen=True, repr=False)
class DistanceCorrelationResult(ResultBase):
    """Container returned by :func:`act_distance`."""

    __result_name__: ClassVar[str] = "Distance correlation"
    __field_docs__: ClassVar[dict[str, str]] = {
        "r_values": "Radial lag distances used for the averaged correlation.",
        "correlation_values": (
            "Correlation values averaged over equal-distance points or radial bins."
        ),
        "std_values": (
            "Standard deviation of correlation values inside each distance group."
        ),
        "anisotropy_values": (
            "Relative angular variation std/rms inside each distance group; "
            "larger values indicate stronger directional dependence."
        ),
        "count_values": "Number of lag-grid samples used in each distance group.",
        "bin_edges": "Radial bin edges used for multi-dimensional correlations.",
        "r_max": "Maximum radial lag distance included in the result.",
        "bin_width": "Radial bin width used for multi-dimensional correlations.",
        "values_shape": "Shape of the original real-space lattice field.",
        "axes": "Spatial lattice axes represented as correlation lag axes.",
        "spacing": "Real-space spacing associated with each correlation axis.",
        "is_subtract_mean": "Whether the mean-squared contribution was subtracted.",
        "is_normalized": "Whether correlation values were normalized by zero lag.",
    }

    r_values: np.ndarray
    correlation_values: np.ndarray
    std_values: np.ndarray
    anisotropy_values: np.ndarray
    count_values: np.ndarray
    bin_edges: np.ndarray
    r_max: float
    bin_width: float
    values_shape: tuple[int, ...]
    axes: tuple[int, ...]
    spacing: tuple[float, ...]
    is_subtract_mean: bool
    is_normalized: bool


@dataclass(slots=True, frozen=True, repr=False)
class RadialSpectrumResult(ResultBase):
    """Container returned by :func:`act_radial_spectrum`."""

    __result_name__: ClassVar[str] = "Radial Fourier spectrum"
    __field_docs__: ClassVar[dict[str, str]] = {
        "k_values": "Radial angular wave-number values used for the averaged spectrum.",
        "spectrum_values": (
            "Fourier power spectrum values averaged over equal-|k| points "
            "or radial bins."
        ),
        "std_values": (
            "Standard deviation of spectrum values inside each wave-number group."
        ),
        "anisotropy_values": (
            "Relative angular variation std/rms inside each wave-number group; "
            "larger values indicate stronger directional dependence."
        ),
        "count_values": "Number of Fourier-grid samples used in each wave-number group.",
        "bin_edges": "Radial wave-number bin edges used for multi-dimensional spectra.",
        "k_max": "Maximum radial angular wave number included in the result.",
        "bin_width": "Radial wave-number bin width used for multi-dimensional spectra.",
        "values_shape": "Shape of the original real-space lattice field.",
        "axes": "Spatial lattice axes included in the Fourier transform.",
        "spacing": "Real-space spacing associated with each transformed axis.",
        "is_normalized": "Whether spectrum values were normalized by sample count.",
    }

    k_values: np.ndarray
    spectrum_values: np.ndarray
    std_values: np.ndarray
    anisotropy_values: np.ndarray
    count_values: np.ndarray
    bin_edges: np.ndarray
    k_max: float
    bin_width: float
    values_shape: tuple[int, ...]
    axes: tuple[int, ...]
    spacing: tuple[float, ...]
    is_normalized: bool


def _as_axes_tuple(axes: int | Sequence[int]) -> tuple[int, ...]:
    """Normalize FFT lattice axes to a tuple of unique spatial axes."""
    if isinstance(axes, int):
        axes_tuple = (axes,)
    else:
        axes_tuple = tuple(axes)

    if not axes_tuple:
        raise ValueError("`axes` must contain at least one lattice axis.")

    for axis in axes_tuple:
        if not isinstance(axis, int):
            raise TypeError("`axes` must contain only integer lattice axes.")
        if axis not in (0, 1, 2):
            raise ValueError("`axes` entries must be one of 0, 1, or 2.")

    if len(set(axes_tuple)) != len(axes_tuple):
        raise ValueError("`axes` must not contain duplicate entries.")
    if axes_tuple != tuple(sorted(axes_tuple)):
        raise ValueError(
            "`axes` must be ordered from low to high so `spacing` maps "
            "unambiguously to the transformed lattice axes."
        )

    return axes_tuple


def _as_spacing_tuple(
    spacing: float | Sequence[float],
    axes: tuple[int, ...],
) -> tuple[float, ...]:
    """Normalize spacing values so there is one positive spacing per FFT axis."""
    if np.isscalar(spacing):
        spacing_tuple = (float(spacing),) * len(axes)
    else:
        spacing_tuple = tuple(float(value) for value in spacing)

    if len(spacing_tuple) != len(axes):
        raise ValueError(
            "`spacing` must be a scalar or have the same length as `axes`. "
            f"Got {len(spacing_tuple)} spacing value(s) for {len(axes)} axis/axes."
        )

    if any(value <= 0 for value in spacing_tuple):
        raise ValueError("All `spacing` values must be positive.")

    return spacing_tuple


def _as_padding_tuple(
    padding_num: int | Sequence[int],
    axes: tuple[int, ...],
) -> tuple[int, ...]:
    """Normalize Fourier interpolation padding for each transformed axis."""
    if isinstance(padding_num, int):
        padding_tuple = (padding_num,) * len(axes)
    else:
        padding_tuple = tuple(padding_num)

    if len(padding_tuple) != len(axes):
        raise ValueError(
            "`padding_num` must be an int or have the same length as `result.axes`. "
            f"Got {len(padding_tuple)} padding value(s) for {len(axes)} axis/axes."
        )

    for value in padding_tuple:
        if not isinstance(value, int):
            raise TypeError("`padding_num` entries must be integers.")
        if value < 0:
            raise ValueError("`padding_num` entries must be non-negative.")

    return padding_tuple


def _build_k_axes(
    shape: Sequence[int],
    axes: tuple[int, ...],
    spacing: tuple[float, ...],
) -> tuple[np.ndarray, ...]:
    """Build angular wave-number arrays for the transformed lattice axes."""
    k_axes = []
    for i_axis, axis in enumerate(axes):
        n = shape[axis]
        d = spacing[i_axis]
        if i_axis == len(axes) - 1:
            k = 2 * np.pi * np.fft.rfftfreq(n, d=d)
        else:
            k = 2 * np.pi * np.fft.fftfreq(n, d=d)
        k_axes.append(k)
    return tuple(k_axes)


def _build_lag_axes(
    shape: Sequence[int],
    axes: tuple[int, ...],
    spacing: tuple[float, ...],
) -> tuple[np.ndarray, ...]:
    """Build real-space periodic lag arrays for the transformed axes."""
    lag_axes = []
    for i_axis, axis in enumerate(axes):
        n = shape[axis]
        d = spacing[i_axis]
        lag_axes.append(np.fft.fftfreq(n) * n * d)
    return tuple(lag_axes)


def _pad_full_fft_axis(
    fft_values: np.ndarray,
    *,
    axis: int,
    target_length: int,
) -> np.ndarray:
    """Center-pad one full complex FFT axis."""
    current_length = fft_values.shape[axis]
    if target_length == current_length:
        return fft_values

    shifted = np.fft.fftshift(fft_values, axes=axis)
    pad_total = target_length - current_length
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    pad_width = [(0, 0)] * shifted.ndim
    pad_width[axis] = (pad_before, pad_after)
    shifted_padded = np.pad(shifted, pad_width, mode="constant")
    return np.fft.ifftshift(shifted_padded, axes=axis)


def _pad_rfft_axis(
    fft_values: np.ndarray,
    *,
    axis: int,
    target_length: int,
) -> np.ndarray:
    """Pad the final real-input FFT axis at the high-frequency end."""
    target_rfft_length = target_length // 2 + 1
    current_length = fft_values.shape[axis]
    if target_rfft_length == current_length:
        return fft_values

    pad_width = [(0, 0)] * fft_values.ndim
    pad_width[axis] = (0, target_rfft_length - current_length)
    return np.pad(fft_values, pad_width, mode="constant")


def act_fourier(
    values,
    axes: int | Sequence[int],
    spacing: float | Sequence[float],
) -> FourierResult:
    """Compute a Fourier transform along lattice axes.

    Parameters
    ----------
    values
        Real-valued lattice field. The first three dimensions are interpreted
        as spatial lattice axes. Any trailing dimensions are treated as field
        components and transformed in parallel.
    axes
        Lattice axis or axes to transform. Valid entries are ``0``, ``1``, and
        ``2``. Multi-axis input must be ordered from low to high so ``spacing``
        maps unambiguously to the transformed lattice axes. These are lattice
        axes, not laboratory x/y/z directions.
    spacing
        Real-space spacing for each transformed lattice axis. A scalar applies
        the same spacing to every transformed axis; otherwise the sequence
        length must match ``axes``.

    Returns
    -------
    FourierResult
        Fourier transform result. Use ``result.act_spectrum()`` to derive a
        power spectrum, ``result.act_filter()`` to filter coefficients,
        ``result.act_inverse()`` to inverse transform,
        ``result.act_correlation()`` to derive an autocorrelation, or
        ``result.act_mean_subtracted_values()`` to derive mean-subtracted
        real-space values. This function uses ``np.fft.rfftn`` for real-valued
        input, so the last transformed axis uses ``np.fft.rfftfreq`` while
        earlier transformed axes use ``np.fft.fftfreq``. The original
        ``values_shape`` is retained so later inverse transforms can recover
        odd-length real-input axes.
    """
    axes = _as_axes_tuple(axes)
    spacing = _as_spacing_tuple(spacing, axes)
    values = as_real_lattice_field(values, name="values")
    shape = values.shape

    fft_values = np.fft.rfftn(values, axes=axes)
    k_axes = _build_k_axes(shape, axes, spacing)

    return FourierResult(
        k_axes=k_axes,
        fft_values=fft_values,
        values_shape=shape,
        axes=axes,
        spacing=spacing,
    )


def _transformed_sample_count(fft_result: FourierResult) -> int:
    """Return the number of real-space samples in the transformed subspace."""
    return int(np.prod([fft_result.values_shape[axis] for axis in fft_result.axes]))


def _rfft_power_weights(fft_result: FourierResult) -> np.ndarray:
    """Return one-sided rFFT weights for mean-square power accounting."""
    rfft_axis = fft_result.axes[-1]
    n = fft_result.values_shape[rfft_axis]
    weights = np.ones(fft_result.fft_values.shape[rfft_axis], dtype=float)

    if n % 2 == 0:
        weights[1:-1] = 2.0
    else:
        weights[1:] = 2.0

    weight_shape = [1] * fft_result.fft_values.ndim
    weight_shape[rfft_axis] = weights.size
    return weights.reshape(weight_shape)


def _average_untransformed_spatial_axes(
    values: np.ndarray,
    fft_result: FourierResult,
) -> np.ndarray:
    """Average over spatial lattice axes not included in the transform."""
    average_axes = tuple(axis for axis in (0, 1, 2) if axis not in fft_result.axes)
    if average_axes:
        return values.mean(axis=average_axes)
    return values


def _spectrum_from_fourier(
    fft_result: FourierResult,
    *,
    is_normalized: bool = False,
) -> np.ndarray:
    """Derive Fourier power from one Fourier transform result."""
    if not isinstance(fft_result, FourierResult):
        raise TypeError("`fft_result` must be a FourierResult.")

    power = np.abs(fft_result.fft_values) ** 2
    if is_normalized:
        sample_count = _transformed_sample_count(fft_result)
        power = power * _rfft_power_weights(fft_result) / sample_count**2

    return _average_untransformed_spatial_axes(power, fft_result)


def _max_radial_k(result: FourierResult) -> float:
    """Return the largest represented radial wave-number magnitude."""
    return float(np.sqrt(sum(np.max(np.abs(k_axis)) ** 2 for k_axis in result.k_axes)))


def _min_nonzero_k_step(result: FourierResult) -> float:
    """Return the smallest non-zero wave-number spacing among transformed axes."""
    steps = []
    for k_axis in result.k_axes:
        unique_abs_k = np.unique(np.abs(k_axis))
        positive_abs_k = unique_abs_k[unique_abs_k > 0]
        if positive_abs_k.size:
            steps.append(float(np.min(positive_abs_k)))
    if not steps:
        return _max_radial_k(result)
    return min(steps)


def _as_radial_spectrum_limits(
    result: FourierResult,
    *,
    k_max: float | None,
    bin_width: float | None,
) -> tuple[float, float]:
    """Validate radial spectrum limits."""
    max_k = _max_radial_k(result)
    tiny = float(np.finfo(float).tiny)

    if k_max is None:
        k_max = max_k
    else:
        k_max = float(
            as_Number(
                k_max,
                name="k_max",
                value_range=(tiny, max_k),
            )
        )

    if bin_width is None:
        bin_width = min(_min_nonzero_k_step(result), k_max)
    else:
        bin_width = float(
            as_Number(
                bin_width,
                name="bin_width",
                value_range=(tiny, k_max),
            )
        )

    return k_max, bin_width


def _anisotropy_from_shell(shell_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return standard deviation and relative angular variation for one shell."""
    std_values = shell_values.std(axis=0)
    rms_values = np.sqrt(np.mean(shell_values**2, axis=0))
    anisotropy_values = np.divide(
        std_values,
        rms_values,
        out=np.zeros_like(std_values, dtype=float),
        where=rms_values > 0,
    )
    return std_values, anisotropy_values


def _average_radial_groups(
    values: np.ndarray,
    labels: np.ndarray,
    group_num: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average values over precomputed radial groups."""
    radial_ndim = labels.ndim
    component_shape = values.shape[radial_ndim:]
    flat_values = values.reshape(labels.size, *component_shape)
    flat_labels = labels.ravel()

    output_shape = (group_num,) + component_shape
    average_values = np.empty(output_shape, dtype=float)
    std_values = np.empty(output_shape, dtype=float)
    anisotropy_values = np.empty(output_shape, dtype=float)
    count_values = np.empty(group_num, dtype=int)

    for i_group in range(group_num):
        shell_values = flat_values[flat_labels == i_group]
        count_values[i_group] = shell_values.shape[0]
        if shell_values.size == 0:
            average_values[i_group] = np.nan
            std_values[i_group] = np.nan
            anisotropy_values[i_group] = np.nan
            continue

        average_values[i_group] = shell_values.mean(axis=0)
        std_values[i_group], anisotropy_values[i_group] = _anisotropy_from_shell(
            shell_values,
        )

    return average_values, std_values, anisotropy_values, count_values


def _radial_spectrum_result_1d(
    result: FourierResult,
    spectrum: np.ndarray,
    *,
    k_max: float,
    bin_width: float,
    is_normalized: bool,
) -> RadialSpectrumResult:
    """Build radial spectrum for a one-dimensional transform."""
    k_abs = np.abs(result.k_axes[0])
    is_included = k_abs <= k_max
    k_values, labels = np.unique(k_abs[is_included], return_inverse=True)
    label_grid = np.full(k_abs.shape, -1, dtype=int)
    label_grid[is_included] = labels

    spectrum_values, std_values, anisotropy_values, count_values = (
        _average_radial_groups(spectrum, label_grid, len(k_values))
    )

    return RadialSpectrumResult(
        k_values=k_values,
        spectrum_values=spectrum_values,
        std_values=std_values,
        anisotropy_values=anisotropy_values,
        count_values=count_values,
        bin_edges=np.array([], dtype=float),
        k_max=k_max,
        bin_width=bin_width,
        values_shape=result.values_shape,
        axes=result.axes,
        spacing=result.spacing,
        is_normalized=is_normalized,
    )


def _radial_spectrum_result_binned(
    result: FourierResult,
    spectrum: np.ndarray,
    *,
    k_max: float,
    bin_width: float,
    is_normalized: bool,
) -> RadialSpectrumResult:
    """Build radial spectrum by averaging over wave-number shells."""
    k_mesh = np.meshgrid(*result.k_axes, indexing="ij")
    k_abs_sq = np.zeros_like(k_mesh[0], dtype=float)
    for k in k_mesh:
        k_abs_sq = k_abs_sq + k**2
    k_abs = np.sqrt(k_abs_sq)

    bin_edges = np.arange(0.0, k_max + bin_width, bin_width)
    if bin_edges[-1] < k_max:
        bin_edges = np.append(bin_edges, k_max)
    else:
        bin_edges[-1] = k_max

    group_num = len(bin_edges) - 1
    labels = np.digitize(k_abs, bin_edges, right=False) - 1
    labels[(k_abs == k_max) & (labels == group_num)] = group_num - 1
    labels[(k_abs > k_max) | (labels < 0) | (labels >= group_num)] = -1

    k_values = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    spectrum_values, std_values, anisotropy_values, count_values = (
        _average_radial_groups(spectrum, labels, group_num)
    )

    return RadialSpectrumResult(
        k_values=k_values,
        spectrum_values=spectrum_values,
        std_values=std_values,
        anisotropy_values=anisotropy_values,
        count_values=count_values,
        bin_edges=bin_edges,
        k_max=k_max,
        bin_width=bin_width,
        values_shape=result.values_shape,
        axes=result.axes,
        spacing=result.spacing,
        is_normalized=is_normalized,
    )


def act_radial_spectrum(
    result: FourierResult,
    *,
    is_normalized: bool = False,
    k_max: float | None = None,
    bin_width: float | None = None,
) -> RadialSpectrumResult:
    """Return the Fourier power spectrum averaged by radial wave number.

    One-dimensional spectra are grouped by exact non-negative ``|k|`` values.
    Multi-dimensional spectra are averaged over circular or spherical
    wave-number shells. ``anisotropy_values`` reports ``std / rms`` within
    each shell, so larger values indicate stronger directional variation.
    """
    if not isinstance(result, FourierResult):
        raise TypeError("`result` must be a FourierResult.")

    k_max, bin_width = _as_radial_spectrum_limits(
        result,
        k_max=k_max,
        bin_width=bin_width,
    )
    spectrum = _spectrum_from_fourier(result, is_normalized=is_normalized)

    if len(result.k_axes) == 1:
        return _radial_spectrum_result_1d(
            result,
            spectrum,
            k_max=k_max,
            bin_width=bin_width,
            is_normalized=is_normalized,
        )

    return _radial_spectrum_result_binned(
        result,
        spectrum,
        k_max=k_max,
        bin_width=bin_width,
        is_normalized=is_normalized,
    )


def act_inverse(
    result: FourierResult,
    *,
    padding_num: int | Sequence[int] = 0,
) -> np.ndarray:
    """Invert a Fourier result, optionally zero-padding for interpolation.

    Parameters
    ----------
    result
        Fourier result returned by ``act_fourier(...)``.
    padding_num
        Number of extra real-space samples to add along each transformed axis
        before inverse transforming. A scalar applies to every transformed
        axis; otherwise the sequence length must match ``result.axes``.

    Returns
    -------
    np.ndarray
        Inverse-transformed real-space field. If ``padding_num`` is positive,
        the output is Fourier-interpolated on a denser periodic lattice.
    """
    if not isinstance(result, FourierResult):
        raise TypeError("`result` must be a FourierResult.")

    padding_num = _as_padding_tuple(padding_num, result.axes)
    original_lengths = tuple(result.values_shape[axis] for axis in result.axes)
    target_lengths = tuple(
        original_length + padding
        for original_length, padding in zip(original_lengths, padding_num)
    )

    fft_values = result.fft_values
    for axis, target_length in zip(result.axes[:-1], target_lengths[:-1]):
        fft_values = _pad_full_fft_axis(
            fft_values,
            axis=axis,
            target_length=target_length,
        )
    fft_values = _pad_rfft_axis(
        fft_values,
        axis=result.axes[-1],
        target_length=target_lengths[-1],
    )

    values = np.fft.irfftn(fft_values, s=target_lengths, axes=result.axes)
    scale = np.prod(target_lengths) / np.prod(original_lengths)
    return values * scale


def _normalize_correlation_values(
    correlation_values: np.ndarray,
    lag_ndim: int,
) -> np.ndarray:
    """Normalize correlation values by their zero-lag value."""
    zero_lag_index = (0,) * lag_ndim + (slice(None),) * (
        correlation_values.ndim - lag_ndim
    )
    zero_lag = correlation_values[zero_lag_index]
    zero_lag_shape = (1,) * lag_ndim + zero_lag.shape
    return correlation_values / zero_lag.reshape(zero_lag_shape)


def _mean_values_from_fourier(result: FourierResult) -> np.ndarray:
    """Compute the full spatial mean for each trailing field component."""
    sample_count = _transformed_sample_count(result)
    zero_index = [slice(None)] * result.fft_values.ndim
    for axis in result.axes:
        zero_index[axis] = 0

    mean_values = result.fft_values[tuple(zero_index)].real / sample_count
    remaining_spatial_axes = []
    shifted_axis = 0
    for axis in range(result.fft_values.ndim):
        if axis in result.axes:
            continue
        if axis in (0, 1, 2):
            remaining_spatial_axes.append(shifted_axis)
        shifted_axis += 1

    if remaining_spatial_axes:
        mean_values = mean_values.mean(axis=tuple(remaining_spatial_axes))
    return np.asarray(mean_values)


def _subtract_correlation_mean(
    correlation_values: np.ndarray,
    mean_values: np.ndarray,
    lag_ndim: int,
) -> np.ndarray:
    """Subtract the squared field mean from correlation values."""
    mean_sq = np.asarray(mean_values) ** 2
    mean_shape = (1,) * lag_ndim + mean_sq.shape
    return correlation_values - mean_sq.reshape(mean_shape)


def _correlation_values_from_fourier(result: FourierResult) -> np.ndarray:
    """Compute averaged periodic autocorrelation values from Fourier power."""
    sample_count = _transformed_sample_count(result)
    lengths = tuple(result.values_shape[axis] for axis in result.axes)
    power = np.abs(result.fft_values) ** 2 / sample_count
    correlation = np.fft.irfftn(power, s=lengths, axes=result.axes)

    return _average_untransformed_spatial_axes(correlation, result)


def _correlation_result_from_fourier(
    result: FourierResult,
) -> CorrelationResult:
    """Build a correlation result from one Fourier result."""
    return CorrelationResult(
        lag_axes=_build_lag_axes(result.values_shape, result.axes, result.spacing),
        correlation_values=_correlation_values_from_fourier(result),
        mean_values=_mean_values_from_fourier(result),
        values_shape=result.values_shape,
        axes=result.axes,
        spacing=result.spacing,
    )


def act_correlation(result: FourierResult) -> CorrelationResult:
    """Return the periodic autocorrelation from a Fourier result.

    The correlation is normalized as an average over the transformed lattice
    axes, so lag zero is ``mean(values**2)`` over those axes. Spatial lattice
    axes not included in the transform are averaged after the inverse
    transform, matching ``act_spectrum(...)``.
    """
    if not isinstance(result, FourierResult):
        raise TypeError("`result` must be a FourierResult.")

    return _correlation_result_from_fourier(result)


def act_correlation_values(
    result: CorrelationResult,
    *,
    is_subtract_mean: bool = False,
    is_normalized: bool = False,
) -> np.ndarray:
    """Return correlation values with optional mean subtraction/normalization."""
    if not isinstance(result, CorrelationResult):
        raise TypeError("`result` must be a CorrelationResult.")

    lag_ndim = len(result.lag_axes)
    values = result.correlation_values
    if is_subtract_mean:
        values = _subtract_correlation_mean(values, result.mean_values, lag_ndim)
    if is_normalized:
        values = _normalize_correlation_values(values, lag_ndim)
    return values


def _as_distance_limits(
    result: CorrelationResult,
    *,
    r_max: float | None,
    bin_width: float | None,
) -> tuple[float, float]:
    """Validate radial averaging limits for a correlation result."""
    axis_lengths = [
        result.values_shape[axis] * spacing
        for axis, spacing in zip(result.axes, result.spacing)
    ]
    default_r_max = min(axis_lengths) / 2.0
    tiny = float(np.finfo(float).tiny)

    if r_max is None:
        r_max = default_r_max
    else:
        r_max = as_Number(
            r_max,
            name="r_max",
            value_range=(tiny, default_r_max),
        )
        r_max = float(r_max)

    if bin_width is None:
        bin_width = min(min(result.spacing), r_max)
    else:
        bin_width = as_Number(
            bin_width,
            name="bin_width",
            value_range=(tiny, r_max),
        )
        bin_width = float(bin_width)

    return r_max, bin_width


def _anisotropy_from_shell(shell_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return standard deviation and relative angular variation for one shell."""
    std_values = shell_values.std(axis=0)
    rms_values = np.sqrt(np.mean(shell_values**2, axis=0))
    anisotropy_values = np.divide(
        std_values,
        rms_values,
        out=np.zeros_like(std_values, dtype=float),
        where=rms_values > 0,
    )
    return std_values, anisotropy_values


def _average_distance_groups(
    values: np.ndarray,
    labels: np.ndarray,
    group_num: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average correlation values over precomputed radial distance groups."""
    lag_ndim = labels.ndim
    component_shape = values.shape[lag_ndim:]
    flat_values = values.reshape(labels.size, *component_shape)
    flat_labels = labels.ravel()

    output_shape = (group_num,) + component_shape
    correlation_values = np.empty(output_shape, dtype=float)
    std_values = np.empty(output_shape, dtype=float)
    anisotropy_values = np.empty(output_shape, dtype=float)
    count_values = np.empty(group_num, dtype=int)

    for i_group in range(group_num):
        shell_values = flat_values[flat_labels == i_group]
        count_values[i_group] = shell_values.shape[0]
        if shell_values.size == 0:
            correlation_values[i_group] = np.nan
            std_values[i_group] = np.nan
            anisotropy_values[i_group] = np.nan
            continue

        correlation_values[i_group] = shell_values.mean(axis=0)
        std_values[i_group], anisotropy_values[i_group] = _anisotropy_from_shell(
            shell_values,
        )

    return correlation_values, std_values, anisotropy_values, count_values


def _distance_result_1d(
    result: CorrelationResult,
    values: np.ndarray,
    *,
    r_max: float,
    bin_width: float,
    is_subtract_mean: bool,
    is_normalized: bool,
) -> DistanceCorrelationResult:
    """Build distance correlation for a one-dimensional lag axis."""
    distances = np.abs(result.lag_axes[0])
    is_included = distances <= r_max
    r_values, labels = np.unique(distances[is_included], return_inverse=True)
    label_grid = np.full(distances.shape, -1, dtype=int)
    label_grid[is_included] = labels

    correlation_values, std_values, anisotropy_values, count_values = (
        _average_distance_groups(values, label_grid, len(r_values))
    )

    return DistanceCorrelationResult(
        r_values=r_values,
        correlation_values=correlation_values,
        std_values=std_values,
        anisotropy_values=anisotropy_values,
        count_values=count_values,
        bin_edges=np.array([], dtype=float),
        r_max=r_max,
        bin_width=bin_width,
        values_shape=result.values_shape,
        axes=result.axes,
        spacing=result.spacing,
        is_subtract_mean=is_subtract_mean,
        is_normalized=is_normalized,
    )


def _distance_result_binned(
    result: CorrelationResult,
    values: np.ndarray,
    *,
    r_max: float,
    bin_width: float,
    is_subtract_mean: bool,
    is_normalized: bool,
) -> DistanceCorrelationResult:
    """Build distance correlation by averaging over radial shells."""
    lag_mesh = np.meshgrid(*result.lag_axes, indexing="ij")
    distance_sq = np.zeros_like(lag_mesh[0], dtype=float)
    for lag in lag_mesh:
        distance_sq = distance_sq + lag**2
    distances = np.sqrt(distance_sq)

    bin_edges = np.arange(0.0, r_max + bin_width, bin_width)
    if bin_edges[-1] < r_max:
        bin_edges = np.append(bin_edges, r_max)
    else:
        bin_edges[-1] = r_max

    group_num = len(bin_edges) - 1
    labels = np.digitize(distances, bin_edges, right=False) - 1
    labels[(distances == r_max) & (labels == group_num)] = group_num - 1
    labels[(distances > r_max) | (labels < 0) | (labels >= group_num)] = -1

    r_values = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    correlation_values, std_values, anisotropy_values, count_values = (
        _average_distance_groups(values, labels, group_num)
    )

    return DistanceCorrelationResult(
        r_values=r_values,
        correlation_values=correlation_values,
        std_values=std_values,
        anisotropy_values=anisotropy_values,
        count_values=count_values,
        bin_edges=bin_edges,
        r_max=r_max,
        bin_width=bin_width,
        values_shape=result.values_shape,
        axes=result.axes,
        spacing=result.spacing,
        is_subtract_mean=is_subtract_mean,
        is_normalized=is_normalized,
    )


def act_distance(
    result: CorrelationResult,
    *,
    is_subtract_mean: bool = False,
    is_normalized: bool = False,
    r_max: float | None = None,
    bin_width: float | None = None,
) -> DistanceCorrelationResult:
    """Return correlation averaged by radial lag distance.

    One-dimensional correlations are grouped by exact non-negative distance,
    combining positive and negative periodic lags. Multi-dimensional
    correlations are averaged over circular or spherical radial bins.
    ``anisotropy_values`` reports the relative angular variation in each
    distance group as ``std / rms``; larger values indicate that the radial
    average hides stronger direction dependence.
    """
    if not isinstance(result, CorrelationResult):
        raise TypeError("`result` must be a CorrelationResult.")

    r_max, bin_width = _as_distance_limits(
        result,
        r_max=r_max,
        bin_width=bin_width,
    )
    values = act_correlation_values(
        result,
        is_subtract_mean=is_subtract_mean,
        is_normalized=is_normalized,
    )

    if len(result.lag_axes) == 1:
        return _distance_result_1d(
            result,
            values,
            r_max=r_max,
            bin_width=bin_width,
            is_subtract_mean=is_subtract_mean,
            is_normalized=is_normalized,
        )

    return _distance_result_binned(
        result,
        values,
        r_max=r_max,
        bin_width=bin_width,
        is_subtract_mean=is_subtract_mean,
        is_normalized=is_normalized,
    )


def act_mean_subtracted_values(
    result: FourierResult,
    *,
    mode: str = "spatial",
    padding_num: int | Sequence[int] = 0,
) -> np.ndarray:
    """Return inverse-transformed values with a selected mean removed.

    Parameters
    ----------
    result
        Fourier result returned by ``act_fourier(...)``.
    mode
        Mean-subtraction mode. ``"spatial"`` subtracts the mean over all three
        lattice axes. ``"axes"`` subtracts the mean only over the axes
        transformed by ``result``. Component axes are preserved in both modes.
    padding_num
        Optional Fourier interpolation padding passed to
        ``act_inverse(...)`` before subtracting the mean.

    Returns
    -------
    np.ndarray
        Real-space values with the requested mean removed.
    """
    if not isinstance(result, FourierResult):
        raise TypeError("`result` must be a FourierResult.")

    if mode not in {"spatial", "axes"}:
        raise ValueError("`mode` must be either 'spatial' or 'axes'.")

    values = act_inverse(result, padding_num=padding_num)
    mean_axes = (0, 1, 2) if mode == "spatial" else result.axes

    return values - values.mean(axis=mean_axes, keepdims=True)


def act_filter(
    result: FourierResult,
    *,
    k_min: float | None = None,
    k_max: float | None = None,
) -> FourierResult:
    """Filter Fourier coefficients by radial wave-number magnitude.

    Parameters
    ----------
    result
        Fourier result returned by ``act_fourier(...)``.
    k_min
        Optional lower bound for retained angular wave-number magnitude.
    k_max
        Optional upper bound for retained angular wave-number magnitude.

    Returns
    -------
    FourierResult
        A new Fourier result with coefficients outside the requested band set
        to zero. Metadata is preserved from ``result``.
    """
    if not isinstance(result, FourierResult):
        raise TypeError("`result` must be a FourierResult.")
    if k_min is None and k_max is None:
        return result
    if k_min is not None:
        k_min = float(k_min)
        if k_min < 0:
            raise ValueError("`k_min` must be non-negative.")
    if k_max is not None:
        k_max = float(k_max)
        if k_max < 0:
            raise ValueError("`k_max` must be non-negative.")
    if k_min is not None and k_max is not None and k_min > k_max:
        raise ValueError("`k_min` must be less than or equal to `k_max`.")

    k_mesh = np.meshgrid(*result.k_axes, indexing="ij")
    k_abs_sq = np.zeros_like(k_mesh[0], dtype=float)
    for k in k_mesh:
        k_abs_sq = k_abs_sq + k**2
    k_abs = np.sqrt(k_abs_sq)

    mask = np.ones_like(k_abs, dtype=bool)
    if k_min is not None:
        mask &= k_abs >= k_min
    if k_max is not None:
        mask &= k_abs <= k_max

    mask_shape = [1] * result.fft_values.ndim
    for i_axis, axis in enumerate(result.axes):
        mask_shape[axis] = mask.shape[i_axis]
    mask = mask.reshape(mask_shape)
    filtered_fft_values = np.where(mask, result.fft_values, 0)

    return replace(result, fft_values=filtered_fft_values)
