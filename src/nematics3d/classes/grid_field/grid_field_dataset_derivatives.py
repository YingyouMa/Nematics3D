"""Spatial derivative helpers for ``GridFieldDataset``."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar, Iterator

import numpy as np

from nematics3d.datatypes import as_qfield9

from ..result_base import ResultBase
from ...grid import is_grid_transform_identity
from .input_grid_field import as_grid_shape


@dataclass(slots=True, frozen=True, repr=False)
class SpatialDerivativeInfo(ResultBase):
    """Payload-free metadata for a dataset spatial-derivative operation."""

    __result_name__: ClassVar[str] = "dataset spatial derivative metadata"

    operator: str
    source: str | None
    source_shape: tuple[int, ...]
    coord: str
    derivative_axis: int | None
    component_axis: int | None
    input_component_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    box_periodic_flag: tuple[bool, bool, bool]
    grid_transform: object
    grid_offset: np.ndarray | None
    stencil: str
    edge_order: int


@dataclass(slots=True, frozen=True, repr=False)
class SpatialDerivativeResult(ResultBase):
    """Inspectable result for a dataset spatial-derivative operation."""

    __result_name__: ClassVar[str] = "dataset spatial derivative result"

    raw_values: np.ndarray | None
    raw_info: SpatialDerivativeInfo
    raw_path: str | None = None

    def _helper_load_values_from_path(self) -> np.ndarray:
        """Load derivative values from the saved array path."""
        if self.raw_path is None:
            raise ValueError("No in-memory values or saved path are available.")
        return np.load(self.raw_path, allow_pickle=False)

    def act_save_values(
        self,
        path,
        *,
        is_release: bool = False,
        is_overwrite: bool = False,
    ) -> SpatialDerivativeResult:
        """
        Save result values to a local ``.npy`` file.

        When ``is_release`` is true, the returned result keeps only the saved
        path and releases the in-memory array reference.
        """
        save_path = Path(path)
        if save_path.suffix != ".npy":
            save_path = Path(f"{save_path}.npy")
        if save_path.exists() and not is_overwrite:
            raise FileExistsError(f"Derivative result path already exists: {save_path}")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        values = self.raw_values
        if values is None:
            values = self._helper_load_values_from_path()
        np.save(save_path, values)

        raw_values = None if is_release else self.raw_values
        return replace(self, raw_values=raw_values, raw_path=str(save_path))

    def act_release_values(self) -> SpatialDerivativeResult:
        """Return a copy without the in-memory array reference."""
        if self.raw_path is None:
            raise ValueError("Cannot release derivative values before saving them.")
        return replace(self, raw_values=None)

    def act_load_values(self) -> SpatialDerivativeResult:
        """Return a copy with values loaded into memory."""
        if self.raw_values is not None:
            return self
        return replace(self, raw_values=self._helper_load_values_from_path())

    @contextmanager
    def act_with_values(self) -> Iterator[np.ndarray]:
        """
        Temporarily expose derivative values.

        If values are already in memory, the existing array is yielded. If only
        ``raw_path`` is available, values are loaded for the ``with`` block and
        the temporary reference is dropped when the block exits.
        """
        if self.raw_values is not None:
            yield self.raw_values
            return

        values = self._helper_load_values_from_path()
        try:
            yield values
        finally:
            del values


def _helper_first_derivative_index(
    self,
    values: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Return the first derivative along one lattice/index axis."""
    axis_size = values.shape[axis]
    if axis_size == 1:
        return np.zeros_like(values, dtype=float)

    if self.raw_box_periodic_flag[axis]:
        return (np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)) / 2.0

    derivative = np.empty_like(values, dtype=float)

    index_first = [slice(None)] * values.ndim
    index_last = [slice(None)] * values.ndim
    index_first[axis] = 0
    index_last[axis] = axis_size - 1

    index_next = [slice(None)] * values.ndim
    index_prev = [slice(None)] * values.ndim
    index_next[axis] = 1
    index_prev[axis] = 0
    derivative[tuple(index_first)] = (
        values[tuple(index_next)] - values[tuple(index_prev)]
    )

    index_next[axis] = axis_size - 1
    index_prev[axis] = axis_size - 2
    derivative[tuple(index_last)] = (
        values[tuple(index_next)] - values[tuple(index_prev)]
    )

    if axis_size > 2:
        index_mid = [slice(None)] * values.ndim
        index_next = [slice(None)] * values.ndim
        index_prev = [slice(None)] * values.ndim
        index_mid[axis] = slice(1, axis_size - 1)
        index_next[axis] = slice(2, axis_size)
        index_prev[axis] = slice(0, axis_size - 2)
        derivative[tuple(index_mid)] = (
            values[tuple(index_next)] - values[tuple(index_prev)]
        ) / 2.0

    return derivative


def _helper_second_derivative_index(
    self,
    values: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Return the second derivative along one lattice/index axis."""
    axis_size = values.shape[axis]
    if axis_size < 3:
        return np.zeros_like(values, dtype=float)

    if self.raw_box_periodic_flag[axis]:
        return np.roll(values, -1, axis=axis) - 2.0 * values + np.roll(
            values, 1, axis=axis
        )

    derivative = np.empty_like(values, dtype=float)

    index_first = [slice(None)] * values.ndim
    index_last = [slice(None)] * values.ndim
    index_first[axis] = 0
    index_last[axis] = axis_size - 1

    index_0 = [slice(None)] * values.ndim
    index_1 = [slice(None)] * values.ndim
    index_2 = [slice(None)] * values.ndim
    index_0[axis] = 0
    index_1[axis] = 1
    index_2[axis] = 2
    derivative[tuple(index_first)] = (
        values[tuple(index_0)]
        - 2.0 * values[tuple(index_1)]
        + values[tuple(index_2)]
    )

    index_0[axis] = axis_size - 3
    index_1[axis] = axis_size - 2
    index_2[axis] = axis_size - 1
    derivative[tuple(index_last)] = (
        values[tuple(index_0)]
        - 2.0 * values[tuple(index_1)]
        + values[tuple(index_2)]
    )

    if axis_size > 2:
        index_mid = [slice(None)] * values.ndim
        index_prev = [slice(None)] * values.ndim
        index_next = [slice(None)] * values.ndim
        index_mid[axis] = slice(1, axis_size - 1)
        index_prev[axis] = slice(0, axis_size - 2)
        index_next[axis] = slice(2, axis_size)
        derivative[tuple(index_mid)] = (
            values[tuple(index_next)]
            - 2.0 * values[tuple(index_mid)]
            + values[tuple(index_prev)]
        )

    return derivative


def _helper_physical_direction_weights(self, direction_index: int) -> np.ndarray:
    """Return index-derivative weights for one physical derivative axis."""
    if is_grid_transform_identity(self.raw_grid_transform):
        weights = np.zeros(3, dtype=float)
        weights[direction_index] = 1.0
        return weights
    transform_inv = np.linalg.inv(self.raw_grid_transform)
    return transform_inv.T[:, direction_index]


def _helper_is_diagonal_grid_transform(self) -> bool:
    """Return whether physical axes align with lattice axes."""
    if is_grid_transform_identity(self.raw_grid_transform):
        return True
    transform = np.asarray(self.raw_grid_transform, dtype=float)
    off_diag = transform.copy()
    np.fill_diagonal(off_diag, 0.0)
    return bool(np.allclose(off_diag, 0.0))


def _helper_spatial_derivative_info(
    self,
    values: np.ndarray,
    *,
    operator: str,
    source: str | None,
    source_shape: tuple[int, ...],
    coord: str,
    derivative_axis: int | None,
    component_axis: int | None = None,
) -> SpatialDerivativeInfo:
    """Build payload-free metadata for an immediate derivative result."""
    grid_offset = self._helper_readonly_grid_array_copy(self.raw_grid_offset)
    grid_transform = self._helper_readonly_grid_array_copy(self.raw_grid_transform)
    return SpatialDerivativeInfo(
        operator=operator,
        source=source,
        source_shape=source_shape,
        coord=coord,
        derivative_axis=derivative_axis,
        component_axis=component_axis,
        input_component_shape=source_shape[3:],
        output_shape=values.shape,
        box_periodic_flag=tuple(bool(flag) for flag in self.raw_box_periodic_flag),
        grid_transform=grid_transform,
        grid_offset=grid_offset,
        stencil="centered difference with periodic wrapping or one-sided boundary",
        edge_order=1,
    )


def _helper_spatial_derivative_result(
    self,
    values: np.ndarray,
    *,
    operator: str,
    source: str | None,
    source_shape: tuple[int, ...],
    coord: str,
    derivative_axis: int | None,
    component_axis: int | None = None,
) -> SpatialDerivativeResult:
    """Build an immediate derivative result plus payload-free metadata."""
    info = self._helper_spatial_derivative_info(
        values,
        operator=operator,
        source=source,
        source_shape=source_shape,
        coord=coord,
        derivative_axis=derivative_axis,
        component_axis=component_axis,
    )
    return SpatialDerivativeResult(raw_values=values, raw_info=info)


def _helper_vector_gradient_split(
    self,
    field_or_values,
    *,
    coord: str,
    name: str,
) -> tuple[str | None, np.ndarray, np.ndarray, np.ndarray]:
    """Return source, values, symmetric strain, and antisymmetric vorticity."""
    source = self._helper_source_name_for_field_values(field_or_values)
    values = self._helper_as_field_values_on_grid(
        field_or_values,
        name=name,
    )
    if values.shape[3:] != (3,):
        raise ValueError(
            f"{name} must be a vector field with shape "
            f"{tuple(self.raw_shape)} + (3,). Got shape {values.shape}."
        )

    grad = self.act_gradient(values, coord=coord)
    grad_transposed = np.swapaxes(grad, -1, -2)
    strain_rate = 0.5 * (grad + grad_transposed)
    vorticity_tensor = 0.5 * (grad - grad_transposed)
    return source, values, strain_rate, vorticity_tensor


def act_gradient(
    self,
    field_or_values,
    *,
    coord: str = "physical",
    is_norm: bool = False,
    is_result: bool = False,
) -> np.ndarray | SpatialDerivativeResult:
    """
    Return the spatial gradient of a field on this dataset grid.

    Finite differences are computed along the first three lattice/index
    axes. The returned array has one additional final axis of length 3,
    representing the derivative direction. When ``is_norm`` is true, return
    the gradient norm instead, preserving any trailing component axes.
    """
    source = self._helper_source_name_for_field_values(field_or_values)
    values = np.asarray(
        self._helper_as_field_values_on_grid(
            field_or_values,
            name="gradient input values",
        ),
        dtype=float,
    )
    source_shape = values.shape

    if is_norm:
        norm_squared = np.zeros_like(values, dtype=float)
        if coord == "index":
            for axis in range(3):
                derivative = self._helper_first_derivative_index(values, axis)
                norm_squared += derivative * derivative
        elif coord == "physical":
            for direction in range(3):
                derivative = self.act_derivative(
                    values,
                    direction=direction,
                    coord=coord,
                )
                norm_squared += derivative * derivative
        else:
            raise ValueError("coord must be either 'index' or 'physical'.")
        result_values = np.sqrt(norm_squared)

        if not is_result:
            return result_values
        return self._helper_spatial_derivative_result(
            result_values,
            operator="gradient_norm",
            source=source,
            source_shape=source_shape,
            coord=coord,
            derivative_axis=None,
        )

    grad = np.empty(values.shape + (3,), dtype=float)

    for axis in range(3):
        derivative = self._helper_first_derivative_index(values, axis)
        grad[..., axis] = derivative

    if coord == "index":
        result_values = grad
    elif coord == "physical":
        if is_grid_transform_identity(self.raw_grid_transform):
            result_values = grad
        else:
            transform_inv = np.linalg.inv(self.raw_grid_transform)
            result_values = np.einsum("...i,ij->...j", grad, transform_inv.T)
    else:
        raise ValueError("coord must be either 'index' or 'physical'.")

    if not is_result:
        return result_values
    return self._helper_spatial_derivative_result(
        result_values,
        operator="gradient",
        source=source,
        source_shape=source_shape,
        coord=coord,
        derivative_axis=None,
    )


def act_derivative(
    self,
    field_or_values,
    direction: str | int,
    *,
    coord: str = "physical",
    is_result: bool = False,
) -> np.ndarray | SpatialDerivativeResult:
    """
    Return one spatial derivative direction of a field on this dataset grid.

    `direction` may be 0/1/2 or "x"/"y"/"z". Only the requested derivative
    direction is evaluated, avoiding the full gradient allocation.
    """
    direction_index = self._helper_as_direction_index(direction)
    source_values = np.asarray(
        self._helper_as_field_values_on_grid(
            field_or_values,
            name="derivative input values",
        ),
        dtype=float,
    )

    if coord == "index":
        values = self._helper_first_derivative_index(
            source_values,
            direction_index,
        )
    elif coord == "physical":
        weights = self._helper_physical_direction_weights(direction_index)
        values = np.zeros_like(source_values, dtype=float)
        for axis, weight in enumerate(weights):
            if weight != 0.0:
                values += weight * self._helper_first_derivative_index(
                    source_values,
                    axis,
                )
    else:
        raise ValueError("coord must be either 'index' or 'physical'.")

    if not is_result:
        return values
    return self._helper_spatial_derivative_result(
        values,
        operator="derivative",
        source=self._helper_source_name_for_field_values(field_or_values),
        source_shape=np.shape(source_values),
        coord=coord,
        derivative_axis=direction_index,
    )


def act_second_derivative(
    self,
    field_or_values,
    direction: str | int,
    *,
    coord: str = "physical",
    is_result: bool = False,
) -> np.ndarray | SpatialDerivativeResult:
    """
    Return one repeated second derivative of a field on this dataset grid.

    This computes ``d2 / d direction**2``. In physical coordinates, the
    implementation follows the same repeated-direction semantics as
    ``act_derivative(act_derivative(...))``.
    """
    direction_index = self._helper_as_direction_index(direction)
    values = np.asarray(
        self._helper_as_field_values_on_grid(
            field_or_values,
            name="second derivative input values",
        ),
        dtype=float,
    )

    if coord == "index":
        second = self._helper_second_derivative_index(values, direction_index)
    elif coord == "physical":
        if is_grid_transform_identity(self.raw_grid_transform):
            second = self._helper_second_derivative_index(
                values,
                direction_index,
            )
        else:
            first = self.act_derivative(
                values,
                direction=direction_index,
                coord=coord,
            )
            second = self.act_derivative(
                first,
                direction=direction_index,
                coord=coord,
            )
    else:
        raise ValueError("coord must be either 'index' or 'physical'.")

    if not is_result:
        return second
    return self._helper_spatial_derivative_result(
        second,
        operator="second_derivative",
        source=self._helper_source_name_for_field_values(field_or_values),
        source_shape=values.shape,
        coord=coord,
        derivative_axis=direction_index,
    )


def act_divergence(
    self,
    field_or_values,
    *,
    coord: str = "physical",
    is_result: bool = False,
) -> np.ndarray | SpatialDerivativeResult:
    """
    Return the divergence of a vector field on this dataset grid.

    The input field must have shape `(Nx, Ny, Nz, 3)`. The vector component
    axis is contracted with the derivative direction.
    """
    values = self._helper_as_field_values_on_grid(
        field_or_values,
        name="divergence input values",
    )
    if values.shape[3:] != (3,):
        raise ValueError(
            "divergence input values must be a vector field with shape "
            f"{tuple(self.raw_shape)} + (3,). Got shape {values.shape}."
        )

    div = np.zeros(values.shape[:3], dtype=float)
    for direction in range(3):
        div += self.act_derivative(
            values[..., direction],
            direction=direction,
            coord=coord,
        )

    if not is_result:
        return div
    return self._helper_spatial_derivative_result(
        div,
        operator="divergence",
        source=self._helper_source_name_for_field_values(field_or_values),
        source_shape=values.shape,
        coord=coord,
        derivative_axis=None,
    )


def act_tensor_divergence(
    self,
    field_or_values,
    *,
    vector_axis: int = -1,
    coord: str = "physical",
    is_result: bool = False,
) -> np.ndarray | SpatialDerivativeResult:
    """
    Return the divergence over one length-3 component axis of a tensor field.

    `vector_axis` selects the component axis contracted with the derivative
    direction. Other trailing component axes are preserved.
    """
    source = self._helper_source_name_for_field_values(field_or_values)
    values = self._helper_as_field_values_on_grid(
        field_or_values,
        name="tensor divergence input values",
    )
    component_axis = vector_axis
    if component_axis < 0:
        component_axis += values.ndim
    if component_axis < 3 or component_axis >= values.ndim:
        raise ValueError(
            "vector_axis must select a trailing component axis, not one of "
            "the first three spatial grid axes."
        )
    if values.shape[component_axis] != 3:
        raise ValueError(
            "tensor divergence vector_axis must have length 3. "
            f"Axis {vector_axis!r} has length {values.shape[component_axis]} "
            f"for input shape {values.shape}."
        )

    values_moved = np.moveaxis(values, component_axis, -1)
    div = np.zeros(values_moved.shape[:-1], dtype=float)
    for direction in range(3):
        div += self.act_derivative(
            values_moved[..., direction],
            direction=direction,
            coord=coord,
        )

    if not is_result:
        return div
    return self._helper_spatial_derivative_result(
        div,
        operator="tensor_divergence",
        source=source,
        source_shape=values.shape,
        coord=coord,
        derivative_axis=None,
        component_axis=component_axis,
    )


def act_directional_derivative(
    self,
    field_or_values,
    direction,
    *,
    coord: str = "physical",
    is_normalize: bool = False,
    is_result: bool = False,
) -> np.ndarray | SpatialDerivativeResult:
    """
    Return the derivative along a supplied direction vector.

    `direction` is interpreted in the same coordinate mode requested by
    `coord`. A single length-3 vector applies globally; an array with
    trailing length-3 axis can provide a spatially varying direction.
    """
    values = self._helper_as_field_values_on_grid(
        field_or_values,
        name="directional derivative input values",
    )
    direction_values = np.asarray(direction, dtype=float)
    if direction_values.shape == (3,):
        direction_values = np.broadcast_to(
            direction_values,
            values.shape[:3] + (3,),
        )
    elif direction_values.shape[-1:] != (3,):
        raise ValueError(
            "direction must be a length-3 vector or an array whose final "
            "axis has length 3."
        )
    else:
        expected_shape = values.shape[:3] + (3,)
        if direction_values.shape != expected_shape:
            raise ValueError(
                "spatially varying direction must have shape "
                f"{expected_shape}. Got {direction_values.shape}."
            )

    if is_normalize:
        norms = np.linalg.norm(direction_values, axis=-1, keepdims=True)
        if np.any(norms == 0.0):
            raise ValueError("direction cannot contain zero vectors.")
        direction_values = direction_values / norms

    extra_component_axes = values.ndim - 3
    direction_expanded = direction_values
    for _ in range(extra_component_axes):
        direction_expanded = np.expand_dims(direction_expanded, axis=-2)

    directional = np.zeros_like(values, dtype=float)
    for direction_index in range(3):
        weight = direction_expanded[..., direction_index]
        directional += weight * self.act_derivative(
            values,
            direction=direction_index,
            coord=coord,
        )

    if not is_result:
        return directional
    return self._helper_spatial_derivative_result(
        directional,
        operator="directional_derivative",
        source=self._helper_source_name_for_field_values(field_or_values),
        source_shape=values.shape,
        coord=coord,
        derivative_axis=None,
    )


def act_curl(
    self,
    field_or_values,
    *,
    coord: str = "physical",
    is_result: bool = False,
) -> np.ndarray | SpatialDerivativeResult:
    """
    Return the curl of a vector field on this dataset grid.

    The input field must have shape `(Nx, Ny, Nz, 3)`. Tensor-valued fields
    are intentionally rejected so their axis convention can be handled by a
    tensor-specific curl helper.
    """
    values = self._helper_as_field_values_on_grid(
        field_or_values,
        name="curl input values",
    )
    if values.shape[3:] != (3,):
        if values.ndim > 4 and any(axis_size == 3 for axis_size in values.shape[3:]):
            raise ValueError(
                "curl input values look like a tensor-valued field. "
                "Please use a tensor-specific curl helper so the tensor "
                "component axis convention is explicit."
            )
        raise ValueError(
            "curl input values must be a vector field with shape "
            f"{tuple(self.raw_shape)} + (3,). Got shape {values.shape}."
        )

    curl = np.empty_like(values, dtype=float)
    curl[..., 0] = self.act_derivative(
        values[..., 2],
        direction=1,
        coord=coord,
    ) - self.act_derivative(
        values[..., 1],
        direction=2,
        coord=coord,
    )
    curl[..., 1] = self.act_derivative(
        values[..., 0],
        direction=2,
        coord=coord,
    ) - self.act_derivative(
        values[..., 2],
        direction=0,
        coord=coord,
    )
    curl[..., 2] = self.act_derivative(
        values[..., 1],
        direction=0,
        coord=coord,
    ) - self.act_derivative(
        values[..., 0],
        direction=1,
        coord=coord,
    )

    if not is_result:
        return curl
    return self._helper_spatial_derivative_result(
        curl,
        operator="curl",
        source=self._helper_source_name_for_field_values(field_or_values),
        source_shape=values.shape,
        coord=coord,
        derivative_axis=None,
    )


def act_tensor_curl(
    self,
    field_or_values,
    *,
    vector_axis: int = -1,
    coord: str = "physical",
    is_result: bool = False,
) -> np.ndarray | SpatialDerivativeResult:
    """
    Return the curl along one length-3 component axis of a tensor field.

    `vector_axis` selects the component axis used as the vector axis for the
    curl. Other trailing component axes are preserved and broadcast exactly
    as stored. The returned array has the same shape as the input field.
    """
    source = self._helper_source_name_for_field_values(field_or_values)
    values = self._helper_as_field_values_on_grid(
        field_or_values,
        name="tensor curl input values",
    )
    component_axis = vector_axis
    if component_axis < 0:
        component_axis += values.ndim
    if component_axis < 3 or component_axis >= values.ndim:
        raise ValueError(
            "vector_axis must select a trailing component axis, not one of "
            "the first three spatial grid axes."
        )
    if values.shape[component_axis] != 3:
        raise ValueError(
            "tensor curl vector_axis must have length 3. "
            f"Axis {vector_axis!r} has length {values.shape[component_axis]} "
            f"for input shape {values.shape}."
        )

    values_moved = np.moveaxis(values, component_axis, -1)
    curl_moved = np.empty_like(values_moved, dtype=float)
    curl_moved[..., 0] = self.act_derivative(
        values_moved[..., 2],
        direction=1,
        coord=coord,
    ) - self.act_derivative(
        values_moved[..., 1],
        direction=2,
        coord=coord,
    )
    curl_moved[..., 1] = self.act_derivative(
        values_moved[..., 0],
        direction=2,
        coord=coord,
    ) - self.act_derivative(
        values_moved[..., 2],
        direction=0,
        coord=coord,
    )
    curl_moved[..., 2] = self.act_derivative(
        values_moved[..., 1],
        direction=0,
        coord=coord,
    ) - self.act_derivative(
        values_moved[..., 0],
        direction=1,
        coord=coord,
    )
    curl = np.moveaxis(curl_moved, -1, component_axis)

    if not is_result:
        return curl
    return self._helper_spatial_derivative_result(
        curl,
        operator="tensor_curl",
        source=source,
        source_shape=values.shape,
        coord=coord,
        derivative_axis=None,
        component_axis=component_axis,
    )


def act_elastic_deformation(
    self,
    field_or_values,
    *,
    coord: str = "physical",
    is_return_scalar: bool = True,
    is_return_vector: bool = True,
) -> dict[str, np.ndarray]:
    """
    Return elastic deformation measures computed from a Q-tensor field.

    The input must be one 3D Q field in either 5-component form
    ``(Nx, Ny, Nz, 5)`` or full tensor form ``(Nx, Ny, Nz, 3, 3)``.
    Spatial derivatives are obtained from ``act_gradient()`` and then
    contracted with the original discrete formulas from the legacy elastic
    helper without trimming boundary cells.
    """
    if not is_return_scalar and not is_return_vector:
        raise ValueError(
            "At least one elastic deformation output must be requested. "
            "Set is_return_scalar and/or is_return_vector to True."
        )

    q_values = as_qfield9(
        self._helper_as_field_values_on_grid(
            field_or_values,
            name="elastic deformation input values",
        ),
        name="elastic deformation input values",
    )
    grad_q = self.act_gradient(q_values, coord=coord)
    diff_q = np.moveaxis(grad_q, -1, 3)

    levi = np.zeros((3, 3, 3), dtype=float)
    levi[0, 1, 2], levi[1, 2, 0], levi[2, 0, 1] = 1.0, 1.0, 1.0
    levi[1, 0, 2], levi[2, 1, 0], levi[0, 2, 1] = -1.0, -1.0, -1.0

    twist_linear = np.einsum("abc,...ad,...bcd->...", levi, q_values, diff_q)
    temp1 = np.einsum("...ab,...aib->...i", q_values, diff_q)
    temp2 = np.einsum("...ia,...bab->...i", q_values, diff_q)

    result: dict[str, np.ndarray] = {}

    if is_return_vector:
        result["splay_vector"] = temp1 + 2.0 * temp2
        result["twist_linear"] = twist_linear
        result["bend_vector"] = -2.0 * temp1 - temp2

    if is_return_scalar:
        splay_vector = temp1 + 2.0 * temp2
        bend_vector = -2.0 * temp1 - temp2
        result["splay"] = np.sum(splay_vector * splay_vector, axis=-1)
        result["twist"] = twist_linear * twist_linear
        result["bend"] = np.sum(bend_vector * bend_vector, axis=-1)

    return result


def act_strain_rate_and_vorticity_tensor(
    self,
    field_or_values,
    *,
    which: str = "both",
    coord: str = "physical",
    is_result: bool = False,
) -> (
    np.ndarray
    | SpatialDerivativeResult
    | tuple[np.ndarray, np.ndarray]
    | tuple[SpatialDerivativeResult, SpatialDerivativeResult]
):
    """
    Return strain-rate and vorticity tensors for a velocity field.

    The velocity input must have shape `(Nx, Ny, Nz, 3)`. Both outputs have
    shape `(Nx, Ny, Nz, 3, 3)`. They are computed from one shared velocity
    gradient so callers that need both tensors avoid duplicate finite
    differences. `which` selects "both", "strain_rate", or
    "vorticity_tensor".
    """
    which = str(which)
    if which not in ("both", "strain_rate", "vorticity_tensor"):
        raise ValueError(
            "which must be 'both', 'strain_rate', or 'vorticity_tensor'."
        )

    source, values, strain_rate, vorticity_tensor = self._helper_vector_gradient_split(
        field_or_values,
        coord=coord,
        name="velocity gradient split input values",
    )

    if not is_result:
        if which == "strain_rate":
            return strain_rate
        if which == "vorticity_tensor":
            return vorticity_tensor
        return strain_rate, vorticity_tensor

    strain_result = self._helper_spatial_derivative_result(
        strain_rate,
        operator="strain_rate",
        source=source,
        source_shape=values.shape,
        coord=coord,
        derivative_axis=None,
    )
    vorticity_result = self._helper_spatial_derivative_result(
        vorticity_tensor,
        operator="vorticity_tensor",
        source=source,
        source_shape=values.shape,
        coord=coord,
        derivative_axis=None,
    )
    if which == "strain_rate":
        return strain_result
    if which == "vorticity_tensor":
        return vorticity_result
    return strain_result, vorticity_result


def act_laplacian(
    self,
    field_or_values,
    *,
    coord: str = "physical",
    is_result: bool = False,
) -> np.ndarray | SpatialDerivativeResult:
    """
    Return the scalar Laplacian of a field on this dataset grid.

    The input field must have shape `(Nx, Ny, Nz)`. Component-wise
    Laplacians for vector or tensor fields are intentionally not inferred by
    this helper. Direct second-derivative stencils are used for index
    coordinates and for physical coordinates whose axes align with lattice
    axes. Non-diagonal physical transforms fall back to repeated physical
    derivatives so mixed second-derivative contributions are preserved.
    """
    source = self._helper_source_name_for_field_values(field_or_values)
    values = self._helper_as_field_values_on_grid(
        field_or_values,
        name="laplacian input values",
    )
    dataset_shape = as_grid_shape(self.raw_shape, name="dataset grid shape")
    if values.shape != dataset_shape:
        if values.shape[:3] == dataset_shape and values.ndim > 3:
            raise ValueError(
                "laplacian input values look like a vector- or "
                "tensor-valued field. Please use "
                "act_componentwise_laplacian() to apply the scalar "
                "Laplacian to each component."
            )
        raise ValueError(
            "laplacian input values must be a scalar field with shape "
            f"{dataset_shape}. Got shape {values.shape}."
        )

    if coord == "index":
        axis_weights = np.ones(3, dtype=float)
    elif coord == "physical" and self._helper_is_diagonal_grid_transform():
        axis_weights = 1.0 / np.asarray(self.calc_grid_spacing, dtype=float) ** 2
    elif coord == "physical":
        grad = self.act_gradient(values, coord=coord)
        laplacian = self.act_divergence(grad, coord=coord)

        if not is_result:
            return laplacian
        return self._helper_spatial_derivative_result(
            laplacian,
            operator="laplacian",
            source=source,
            source_shape=values.shape,
            coord=coord,
            derivative_axis=None,
        )
    else:
        raise ValueError("coord must be either 'index' or 'physical'.")

    laplacian = np.zeros_like(values, dtype=float)
    for direction, weight in enumerate(axis_weights):
        laplacian += weight * self._helper_second_derivative_index(
            values,
            direction,
        )

    if not is_result:
        return laplacian
    return self._helper_spatial_derivative_result(
        laplacian,
        operator="laplacian",
        source=source,
        source_shape=values.shape,
        coord=coord,
        derivative_axis=None,
    )


def act_componentwise_laplacian(
    self,
    field_or_values,
    *,
    coord: str = "physical",
    is_result: bool = False,
) -> np.ndarray | SpatialDerivativeResult:
    """
    Return the component-wise Laplacian of a field on this dataset grid.

    The first three axes are spatial grid axes. Any trailing axes are
    treated as independent components, and the scalar Laplacian is applied
    to each component independently.
    """
    source = self._helper_source_name_for_field_values(field_or_values)
    values = self._helper_as_field_values_on_grid(
        field_or_values,
        name="component-wise laplacian input values",
    )

    if coord == "index":
        axis_weights = np.ones(3, dtype=float)
    elif coord == "physical" and self._helper_is_diagonal_grid_transform():
        axis_weights = 1.0 / np.asarray(self.calc_grid_spacing, dtype=float) ** 2
    elif coord == "physical":
        laplacian = np.zeros_like(values, dtype=float)
        for direction in range(3):
            first_derivative = self.act_derivative(
                values,
                direction=direction,
                coord=coord,
            )
            laplacian += self.act_derivative(
                first_derivative,
                direction=direction,
                coord=coord,
            )

        if not is_result:
            return laplacian
        return self._helper_spatial_derivative_result(
            laplacian,
            operator="componentwise_laplacian",
            source=source,
            source_shape=values.shape,
            coord=coord,
            derivative_axis=None,
        )
    else:
        raise ValueError("coord must be either 'index' or 'physical'.")

    laplacian = np.zeros_like(values, dtype=float)
    for direction, weight in enumerate(axis_weights):
        laplacian += weight * self._helper_second_derivative_index(
            values,
            direction,
        )

    if not is_result:
        return laplacian
    return self._helper_spatial_derivative_result(
        laplacian,
        operator="componentwise_laplacian",
        source=source,
        source_shape=values.shape,
        coord=coord,
        derivative_axis=None,
    )
