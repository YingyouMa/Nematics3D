"""Gaussian smoothing helpers for ``GridFieldDataset``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.ndimage import gaussian_filter

from nematics3d.datatypes import (
    as_number,
    as_real_lattice_field,
    as_str,
)
from ...grid import as_readonly_grid_offset, as_readonly_grid_transform

from ..npy_array_payload import NpyArrayPayload
from ..result_base import ResultBase


@dataclass(slots=True, frozen=True, repr=False)
class GaussianSmoothInfo(ResultBase):
    """Payload-free metadata for a dataset Gaussian smoothing operation."""

    __result_name__: ClassVar[str] = "dataset Gaussian smoothing metadata"
    __field_docs__: ClassVar[dict[str, str]] = {
        "operator": "Name of the dataset operator that produced this result.",
        "source_name": (
            "Registered source field name when the smoothing input came from the "
            "dataset; None for direct array inputs."
        ),
        "source_shape": (
            "Shape of the original input values before smoothing, including any "
            "trailing component axes."
        ),
        "coord_type": (
            "Coordinate system used to interpret the user-facing sigma values: "
            "'physical' or 'index'."
        ),
        "sigma": "User-requested Gaussian width along the three dataset axes.",
        "sigma_index": (
            "Gaussian width converted into lattice-index units for the actual "
            "separable convolution."
        ),
        "truncate": (
            "Kernel cutoff radius in units of sigma, used to truncate each 1D "
            "Gaussian."
        ),
        "boundary": (
            "Per-axis boundary modes applied during convolution, for example "
            "'wrap' or 'reflect'."
        ),
        "input_component_shape": (
            "Trailing non-spatial component shape of the input field; empty for "
            "scalar fields."
        ),
        "box_periodic_flag": (
            "Dataset periodic-boundary flags along the three lattice axes."
        ),
        "grid_transform": (
            "Read-only snapshot of the dataset grid transform used for this result."
        ),
        "grid_offset": (
            "Read-only snapshot of the dataset grid offset used for this result."
        ),
        "weights_source_name": (
            "Registered weight-field name when weighted smoothing used a dataset "
            "field; None for direct-array weights or unweighted smoothing."
        ),
        "weights_floor": (
            "Minimum smoothed weight treated as valid for weighted normalization; "
            "None for unweighted smoothing."
        ),
    }

    operator: str
    source_name: str | None
    source_shape: tuple[int, ...]
    coord_type: str
    sigma: tuple[float, float, float]
    sigma_index: tuple[float, float, float]
    truncate: float
    boundary: tuple[str, str, str]
    input_component_shape: tuple[int, ...]
    box_periodic_flag: tuple[bool, bool, bool]
    grid_transform: object
    grid_offset: np.ndarray | None
    weights_source_name: str | None
    weights_floor: float | None


def _helper_as_gaussian_sigma_3(
    self,
    sigma,
    *,
    coord: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return user-space and index-space Gaussian widths as length-3 tuples."""
    coord = as_str(
        coord,
        name="Gaussian smoothing coordinate system",
        pool=("physical", "index"),
    )
    sigma_name = "Gaussian smoothing sigma"

    if isinstance(sigma, (tuple, list, np.ndarray)):
        sigma_array = np.asarray(sigma, dtype=float)
        if sigma_array.shape != (3,):
            raise ValueError(
                f"{sigma_name} must be a scalar or a length-3 sequence. "
                f"Got shape {sigma_array.shape}."
            )
        sigma_user = tuple(
            float(
                as_number(
                    value,
                    name=f"{sigma_name} axis {axis}",
                    value_range=(0.0, np.inf),
                )
            )
            for axis, value in enumerate(sigma_array)
        )
    else:
        sigma_scalar = float(
            as_number(
                sigma,
                name=sigma_name,
                value_range=(0.0, np.inf),
            )
        )
        sigma_user = (sigma_scalar, sigma_scalar, sigma_scalar)

    if coord == "index":
        return sigma_user, sigma_user

    spacing = np.asarray(self.calc_grid_spacing, dtype=float)
    sigma_index = tuple(float(value / step) for value, step in zip(sigma_user, spacing))
    return sigma_user, sigma_index


def _helper_as_gaussian_boundary_mode(
    self,
    boundary: str = "auto",
) -> tuple[str, str, str]:
    """Return per-axis boundary modes for Gaussian smoothing."""
    boundary = as_str(
        boundary,
        name="Gaussian smoothing boundary mode",
        pool=("auto", "wrap", "reflect"),
    )
    if boundary == "auto":
        return tuple(
            "wrap" if is_periodic else "reflect"
            for is_periodic in self.raw_box_periodic_flag
        )
    return (boundary, boundary, boundary)


def _helper_gaussian_smooth_info(
    self,
    values: np.ndarray,
    *,
    source_name: str | None,
    source_shape: tuple[int, ...],
    coord_type: str,
    sigma: tuple[float, float, float],
    sigma_index: tuple[float, float, float],
    truncate: float,
    boundary: tuple[str, str, str],
    weights_source_name: str | None,
    weights_floor: float | None,
) -> GaussianSmoothInfo:
    """Build payload-free metadata for an immediate Gaussian smoothing result."""
    grid_offset = as_readonly_grid_offset(self.raw_grid_offset)
    grid_transform = as_readonly_grid_transform(self.raw_grid_transform)
    return GaussianSmoothInfo(
        operator="gaussian_smooth",
        source_name=source_name,
        source_shape=source_shape,
        coord_type=coord_type,
        sigma=sigma,
        sigma_index=sigma_index,
        truncate=truncate,
        boundary=boundary,
        input_component_shape=source_shape[3:],
        box_periodic_flag=tuple(bool(flag) for flag in self.raw_box_periodic_flag),
        grid_transform=grid_transform,
        grid_offset=grid_offset,
        weights_source_name=weights_source_name,
        weights_floor=weights_floor,
    )


def _helper_gaussian_smooth_result(
    self,
    values: np.ndarray,
    *,
    source_name: str | None,
    source_shape: tuple[int, ...],
    coord_type: str,
    sigma: tuple[float, float, float],
    sigma_index: tuple[float, float, float],
    truncate: float,
    boundary: tuple[str, str, str],
    weights_source_name: str | None,
    weights_floor: float | None,
) -> NpyArrayPayload[GaussianSmoothInfo]:
    """Build an immediate Gaussian smoothing payload plus metadata."""
    info = _helper_gaussian_smooth_info(
        self,
        values,
        source_name=source_name,
        source_shape=source_shape,
        coord_type=coord_type,
        sigma=sigma,
        sigma_index=sigma_index,
        truncate=truncate,
        boundary=boundary,
        weights_source_name=weights_source_name,
        weights_floor=weights_floor,
    )
    return NpyArrayPayload(raw_values=values, raw_info=info)


def _helper_as_gaussian_weights(
    self,
    weights,
) -> tuple[np.ndarray, str | None, tuple[int, ...]]:
    """Return validated per-voxel weights for weighted Gaussian smoothing."""
    weights_source_name = self._helper_source_name_for_field_values(weights)
    weights_values = as_real_lattice_field(
        self._helper_as_field_values_on_grid(
            weights,
            name="Gaussian smoothing weights",
        ),
        name="Gaussian smoothing weights",
        extra_ndim=0,
        is_finite=True,
        value_range=(0.0, 1.0),
        bounded=False,
    )
    dataset_shape = tuple(np.asarray(self.raw_shape, dtype=int).tolist())
    if weights_values.shape != dataset_shape:
        raise ValueError(
            "Gaussian smoothing weights must be a scalar field whose shape "
            "exactly matches the dataset grid shape. "
            f"Dataset shape is {dataset_shape}; got {weights_values.shape}."
        )
    return weights_values, weights_source_name, weights_values.shape


def _helper_gaussian_kernel_radius(
    self,
    sigma_axis: float,
    *,
    truncate: float,
) -> int:
    """Return the truncated half-width of one Gaussian kernel axis."""
    del self
    if sigma_axis <= 0.0:
        return 0
    return int(np.ceil(truncate * sigma_axis))


def _helper_build_gaussian_kernel_1d(
    self,
    sigma_axis: float,
    *,
    truncate: float,
) -> np.ndarray:
    """Return one normalized 1D Gaussian kernel."""
    radius = self._helper_gaussian_kernel_radius(
        sigma_axis,
        truncate=truncate,
    )
    if radius == 0:
        return np.array([1.0], dtype=float)

    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_axis) ** 2)
    kernel_sum = float(np.sum(kernel))
    if kernel_sum <= 0.0:
        return np.array([1.0], dtype=float)
    return kernel / kernel_sum


def _helper_pad_for_gaussian_axis(
    self,
    values: np.ndarray,
    *,
    axis: int,
    radius: int,
    mode: str,
) -> np.ndarray:
    """Return values padded only along one requested axis."""
    del self
    if radius <= 0:
        return values

    pad_width = [(0, 0)] * values.ndim
    pad_width[axis] = (radius, radius)
    return np.pad(values, pad_width, mode=mode)


def _helper_convolve_gaussian_axis(
    self,
    values: np.ndarray,
    *,
    kernel: np.ndarray,
    axis: int,
    mode: str,
) -> np.ndarray:
    """Return one-axis Gaussian convolution with boundary handling."""
    if kernel.ndim != 1:
        raise ValueError("Gaussian convolution kernel must be one-dimensional.")
    if kernel.size == 1:
        return values.copy()

    radius = kernel.size // 2
    padded = self._helper_pad_for_gaussian_axis(
        values,
        axis=axis,
        radius=radius,
        mode=mode,
    )
    result = np.zeros_like(values, dtype=float)

    base_slices = [slice(None)] * padded.ndim
    axis_length = values.shape[axis]
    for offset, weight in enumerate(kernel):
        shifted_slices = list(base_slices)
        shifted_slices[axis] = slice(offset, offset + axis_length)
        result += float(weight) * padded[tuple(shifted_slices)]

    return result


def _helper_gaussian_smooth_values(
    self,
    values: np.ndarray,
    *,
    sigma_index: tuple[float, float, float],
    truncate: float,
    boundary: tuple[str, str, str],
) -> np.ndarray:
    """Return Gaussian-smoothed values via scipy.ndimage.gaussian_filter."""
    if len(set(boundary)) == 1:
        result = gaussian_filter(
            np.asarray(values, dtype=float),
            sigma=sigma_index,
            mode=boundary[0],
            truncate=truncate,
        )
    else:
        result = np.asarray(values, dtype=float)
        for axis, (sigma_axis, boundary_mode) in enumerate(zip(sigma_index, boundary)):
            result = gaussian_filter(
                result,
                sigma=[s if i == axis else 0.0 for i, s in enumerate(sigma_index[:3])],
                mode=boundary_mode,
                truncate=truncate,
            )
    return result


def act_gaussian_smooth(
    self,
    field_or_values,
    sigma,
    *,
    coord: str = "physical",
    weights=None,
    weights_floor: float = 1e-12,
    truncate: float | None = None,
    boundary: str = "auto",
    is_result: bool = False,
) -> np.ndarray | NpyArrayPayload[GaussianSmoothInfo]:
    """
    Return Gaussian-smoothed field values on this dataset grid.

    The smoothing is applied by separable real-space Gaussian convolution
    along the first three lattice axes. Inputs are normalized onto the
    dataset grid, and callers may request either the smoothed values directly
    or an ``NpyArrayPayload`` carrying payload-free metadata.
    """
    source_name = self._helper_source_name_for_field_values(field_or_values)
    values = np.asarray(
        self._helper_as_field_values_on_grid(
            field_or_values,
            name="Gaussian smoothing input values",
        ),
        dtype=float,
    )
    source_shape = values.shape
    coord_type = as_str(
        coord,
        name="Gaussian smoothing coordinate system",
        pool=("physical", "index"),
    )
    sigma_user, sigma_index = self._helper_as_gaussian_sigma_3(
        sigma,
        coord=coord_type,
    )
    weights_values = None
    weights_source_name = None
    weights_floor_value = None
    if weights is not None:
        weights_values, weights_source_name, _ = self._helper_as_gaussian_weights(
            weights
        )
        weights_floor_value = float(
            as_number(
                weights_floor,
                name="Gaussian smoothing weights_floor",
                value_range=(0.0, np.inf),
            )
        )
    if truncate is None:
        truncate_value = 4.0
    else:
        truncate_value = float(
            as_number(
                truncate,
                name="Gaussian smoothing truncate",
                value_range=(0.0, np.inf),
            )
        )
    boundary_modes = self._helper_as_gaussian_boundary_mode(boundary)

    if weights_values is None:
        smoothed_values = self._helper_gaussian_smooth_values(
            values,
            sigma_index=sigma_index,
            truncate=truncate_value,
            boundary=boundary_modes,
        )
    else:
        if values.ndim > 3:
            weights_expanded = weights_values.reshape(
                weights_values.shape + (1,) * (values.ndim - 3)
            )
        else:
            weights_expanded = weights_values
        weighted_values = values * weights_expanded
        smoothed_weighted_values = self._helper_gaussian_smooth_values(
            weighted_values,
            sigma_index=sigma_index,
            truncate=truncate_value,
            boundary=boundary_modes,
        )
        smoothed_weights = self._helper_gaussian_smooth_values(
            weights_values,
            sigma_index=sigma_index,
            truncate=truncate_value,
            boundary=boundary_modes,
        )
        smoothed_values = values.copy()
        valid = smoothed_weights > weights_floor_value
        if values.ndim > 3:
            valid_weights = smoothed_weights[valid].reshape((np.count_nonzero(valid),))
            valid_weights = valid_weights.reshape(
                valid_weights.shape + (1,) * (values.ndim - 3)
            )
            smoothed_values[valid, ...] = (
                smoothed_weighted_values[valid, ...] / valid_weights
            )
        else:
            smoothed_values[valid] = (
                smoothed_weighted_values[valid] / smoothed_weights[valid]
            )

    if not is_result:
        return smoothed_values
    return self._helper_gaussian_smooth_result(
        smoothed_values,
        source_name=source_name,
        source_shape=source_shape,
        coord_type=coord_type,
        sigma=sigma_user,
        sigma_index=sigma_index,
        truncate=truncate_value,
        boundary=boundary_modes,
        weights_source_name=weights_source_name,
        weights_floor=weights_floor_value,
    )
