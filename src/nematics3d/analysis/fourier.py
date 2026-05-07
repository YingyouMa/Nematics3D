"""Fourier helpers for real-valued lattice fields."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from typing import ClassVar

import numpy as np

from nematics3d.classes.result_base import ResultBase
from nematics3d.datatypes import as_real_lattice_field, as_str


@dataclass(slots=True, frozen=True, repr=False)
class FourierResult(ResultBase):
    """Container returned by :func:`field_fourier` with ``output="fft"``."""

    __result_name__: ClassVar[str] = "Fourier transform"

    k_axes: tuple[np.ndarray, ...]
    fft_values: np.ndarray
    values_shape: tuple[int, ...]
    axes: tuple[int, ...]
    spacing: tuple[float, ...]
    is_mean_subtracted: bool


@dataclass(slots=True, frozen=True, repr=False)
class SpectrumResult(ResultBase):
    """Container returned by :func:`field_fourier` with ``output="spectrum"``."""

    __result_name__: ClassVar[str] = "Fourier spectrum"

    k_axes: tuple[np.ndarray, ...]
    spectrum: np.ndarray
    values_shape: tuple[int, ...]
    axes: tuple[int, ...]
    spacing: tuple[float, ...]
    is_mean_subtracted: bool
    component_mode: str


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


def _compute_field_fft(
    values,
    axes: int | Sequence[int],
    spacing: float | Sequence[float],
    *,
    is_subtract_mean: bool = True,
) -> FourierResult:
    """Compute a real-input FFT along one or more lattice axes.

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
    is_subtract_mean
        If ``True``, subtract the spatial mean before the transform. For
        component fields, each component has its own spatial mean removed.

    Returns
    -------
    FourierResult
        Result object containing ``k_axes`` and complex ``fft_values``. This
        function uses ``np.fft.rfftn`` for real-valued input, so the last
        transformed axis uses ``np.fft.rfftfreq`` while earlier transformed
        axes use ``np.fft.fftfreq``. The original ``values_shape`` is retained
        so later inverse transforms can recover odd-length real-input axes.
    """
    axes = _as_axes_tuple(axes)
    spacing = _as_spacing_tuple(spacing, axes)
    values = as_real_lattice_field(values, name="values")
    shape = values.shape

    if is_subtract_mean:
        values = values - values.mean(axis=(0, 1, 2), keepdims=True)

    fft_values = np.fft.rfftn(values, axes=axes)
    k_axes = _build_k_axes(shape, axes, spacing)

    return FourierResult(
        k_axes=k_axes,
        fft_values=fft_values,
        values_shape=shape,
        axes=axes,
        spacing=spacing,
        is_mean_subtracted=is_subtract_mean,
    )


def field_fourier(
    values,
    axes: int | Sequence[int],
    spacing: float | Sequence[float],
    *,
    output: str = "spectrum",
    is_subtract_mean: bool = True,
    component_mode: str = "component",
) -> FourierResult | SpectrumResult:
    """Compute a Fourier transform or power spectrum along lattice axes.

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
    output
        Selects the returned result. ``"spectrum"`` returns Fourier power.
        ``"fft"`` returns complex Fourier coefficients.
    is_subtract_mean
        If ``True``, subtract the spatial mean before the transform. For
        component fields, each component has its own spatial mean removed.
    component_mode
        Used only when ``output="spectrum"``. ``"component"`` keeps each
        component spectrum. ``"sum"`` sums power over all trailing component
        axes and returns one total spectrum.

    Returns
    -------
    FourierResult or SpectrumResult
        ``FourierResult`` is returned for ``output="fft"``. ``SpectrumResult``
        is returned for ``output="spectrum"``.
    """
    output = as_str(output, name="output", pool=("spectrum", "fft"))
    component_mode = as_str(
        component_mode,
        name="component_mode",
        pool=("component", "sum"),
    )
    fft_result = _compute_field_fft(
        values,
        axes=axes,
        spacing=spacing,
        is_subtract_mean=is_subtract_mean,
    )

    if output == "fft":
        return fft_result

    power = np.abs(fft_result.fft_values) ** 2
    average_axes = tuple(axis for axis in (0, 1, 2) if axis not in fft_result.axes)
    if average_axes:
        spectrum = power.mean(axis=average_axes)
    else:
        spectrum = power

    if component_mode == "sum":
        component_axes = tuple(range(len(fft_result.axes), spectrum.ndim))
        if component_axes:
            spectrum = spectrum.sum(axis=component_axes)

    return SpectrumResult(
        k_axes=fft_result.k_axes,
        spectrum=spectrum,
        values_shape=fft_result.values_shape,
        axes=fft_result.axes,
        spacing=fft_result.spacing,
        is_mean_subtracted=fft_result.is_mean_subtracted,
        component_mode=component_mode,
    )


def field_inverse_fourier(
    result: FourierResult,
    *,
    padding_num: int | Sequence[int] = 0,
) -> np.ndarray:
    """Invert a Fourier result, optionally zero-padding for interpolation.

    Parameters
    ----------
    result
        Fourier result returned by ``field_fourier(..., output="fft")``.
    padding_num
        Number of extra real-space samples to add along each transformed axis
        before inverse transforming. A scalar applies to every transformed
        axis; otherwise the sequence length must match ``result.axes``.

    Returns
    -------
    np.ndarray
        Inverse-transformed real-space field. If ``padding_num`` is positive,
        the output is Fourier-interpolated on a denser periodic lattice. If
        ``result.is_mean_subtracted`` is ``True``, the returned field is the
        mean-subtracted field because the removed mean is not stored.
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


def field_fourier_filter(
    result: FourierResult,
    *,
    k_min: float | None = None,
    k_max: float | None = None,
) -> FourierResult:
    """Filter Fourier coefficients by radial wave-number magnitude.

    Parameters
    ----------
    result
        Fourier result returned by ``field_fourier(..., output="fft")``.
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
