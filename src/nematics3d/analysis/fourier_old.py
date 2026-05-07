"""Fourier spectrum helpers for real-valued lattice fields."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from nematics3d.classes.result_base import ResultBase
from nematics3d.datatypes import as_str


@dataclass(slots=True, frozen=True, repr=False)
class SpectrumResult(ResultBase):
    """Container returned by :func:`field_spectrum`."""

    __result_name__: ClassVar[str] = "Fourier spectrum"

    k_axes: tuple[np.ndarray, ...]
    spectrum: np.ndarray
    axes: tuple[int, ...]
    spacing: tuple[float, ...]
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


def field_spectrum(
    values,
    axes: int | Sequence[int],
    spacing: float | Sequence[float],
    *,
    is_subtract_mean: bool = True,
    component_mode: str = "component",
) -> SpectrumResult:
    """Compute a Fourier power spectrum along one or more lattice axes.

    Parameters
    ----------
    values
        Real-valued lattice field. The first three dimensions are interpreted
        as spatial lattice axes. Any trailing dimensions are treated as field
        components and transformed in parallel, so scalar, vector, Q5, and
        Q9-like fields share one code path.
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
    component_mode
        Controls how trailing component axes are handled after the Fourier
        power is computed. ``"component"`` keeps each component spectrum.
        ``"sum"`` sums power over all trailing component axes and returns one
        total spectrum.

    Returns
    -------
    SpectrumResult
        Result object containing ``k_axes`` and ``spectrum``. The last
        transformed axis uses ``np.fft.rfftfreq``; earlier transformed axes use
        ``np.fft.fftfreq``.
    """
    axes = _as_axes_tuple(axes)
    spacing = _as_spacing_tuple(spacing, axes)
    component_mode = as_str(
        component_mode,
        name="component_mode",
        pool=("component", "sum"),
    )

    values = np.asarray(values)
    if values.ndim < 3:
        raise ValueError("`values` must have at least three spatial axes.")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("`values` must contain numeric data.")
    if np.iscomplexobj(values):
        raise TypeError("`values` must be real-valued; complex fields are unsupported.")
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(float, copy=False)

    if is_subtract_mean:
        values = values - values.mean(axis=(0, 1, 2), keepdims=True)

    fft_values = np.fft.rfftn(values, axes=axes)
    power = np.abs(fft_values) ** 2

    average_axes = tuple(axis for axis in (0, 1, 2) if axis not in axes)
    if average_axes:
        spectrum = power.mean(axis=average_axes)
    else:
        spectrum = power

    if component_mode == "component":
        pass
    else:
        component_axes = tuple(range(len(axes), spectrum.ndim))
        if component_axes:
            spectrum = spectrum.sum(axis=component_axes)

    k_axes = []
    for i_axis, axis in enumerate(axes):
        n = values.shape[axis]
        d = spacing[i_axis]
        if i_axis == len(axes) - 1:
            k = 2 * np.pi * np.fft.rfftfreq(n, d=d)
        else:
            k = 2 * np.pi * np.fft.fftfreq(n, d=d)
        k_axes.append(k)

    return SpectrumResult(
        k_axes=tuple(k_axes),
        spectrum=spectrum,
        axes=axes,
        spacing=spacing,
        component_mode=component_mode,
    )
