"""Shared-grid dataset for binding multiple physical fields."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar, Mapping

import numpy as np

from nematics3d.datatypes import (
    UNSET,
    Unset,
)

from ..bounds import as_bounds
from ..class_base import ClassBase
from ..registry_base import RegistryBase
from ..result_base import ResultBase
from ...grid import (
    apply_linear_transform,
    generate_coordinate_grid,
    is_grid_transform_identity,
)
from ...general import get_box_corners
from .input_grid_field import InputGridField, as_grid_shape


def as_field_values(value, name: str = "field values") -> np.ndarray:
    """Convert field values to a numeric NumPy array."""
    values = np.asarray(value)
    if values.ndim < 3:
        raise ValueError(
            f"{name!r} must have at least three grid axes. "
            f"Got shape {values.shape} instead."
        )
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"{name!r} must contain numeric values. Got {values.dtype}.")
    return values


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

    raw_values: np.ndarray
    raw_info: SpatialDerivativeInfo


class FieldData(ClassBase):
    """Thin wrapper for one physical field living on a GridFieldDataset."""

    # fmt: off
    __attr_defs__: ClassVar[Mapping[str, dict[str, Any]]] = {
        **dict(ClassBase.__attr_defs__),
        "raw_name": {
            **dict(ClassBase.__attr_defs__["raw_name"]),
            "doc": "Name identifier of this grid field.",
        },
        "raw_values": {
            "doc": "Field values with leading axes matching the dataset grid.",
            "validator": as_field_values,
        },
        "raw_info": {
            "doc": "Optional user-provided metadata or provenance for this field.",
        },
        "interpolator": {
            "doc": "The generic interpolator associated with this field.",
            "kind": "entity",
        },
    }
    # fmt: on

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.get("kind") not in ("relation", "property")
        and name not in ClassBase.__slots__
    )

    def __init__(
        self,
        name: str,
        values,
        info=None,
    ):
        values = self.__attr_defs__["raw_values"]["validator"](
            values,
            self.__attr_defs__["raw_values"]["doc"],
        )

        super().__init__(name=name, name_replace="field")
        object.__setattr__(self, "raw_values", values)
        object.__setattr__(self, "raw_info", info)
        object.__setattr__(self, "interpolator", None)

    def act_add_interpolator(self):
        """Create and bind a GridInterpolator if one is not already present."""
        from .grid_interpolator import GridInterpolator

        interpolator_old = self.interpolator
        if isinstance(interpolator_old, GridInterpolator):
            return interpolator_old

        interpolator = GridInterpolator(self, name=f"{self.name} interpolator")
        object.__setattr__(self, "interpolator", interpolator)
        return self.interpolator

    def act_interpolate(
        self,
        points: np.ndarray,
        is_index: bool = False,
        is_out_warning: bool = False,
    ):
        """Interpolate this field at arbitrary sample points."""
        if self.interpolator is None:
            self.act_add_interpolator()
        return self.interpolator.interpolate(
            points,
            is_index=is_index,
            is_out_warning=is_out_warning,
        )


class GridFieldDataset(ClassBase):
    """Container for physical fields sharing one lattice and boundary model."""

    # fmt: off
    __attr_defs__: ClassVar[Mapping[str, dict[str, Any]]] = {
        **dict(ClassBase.__attr_defs__),
        "raw_name": {
            **dict(ClassBase.__attr_defs__["raw_name"]),
            "doc": "Name identifier of this shared-grid field dataset.",
        },
        "raw_shape": {
            "doc": "Shared lattice grid shape (Nx, Ny, Nz), or UNSET before inference.",
        },
        "raw_box_periodic_flag": {
            "doc": "Per-dimension periodic boundary condition flags.",
        },
        "raw_grid_offset": {
            "doc": "Grid translation offset mapping lattice indices to real space.",
        },
        "raw_grid_transform": {
            "doc": "Linear transform mapping lattice indices to real space.",
        },
        "calc_grid_index": {
            "doc": "Lattice coordinate grid in index space.",
            "kind": "calc",
        },
        "calc_grid": {
            "doc": "Coordinate grid in real space after transform and offset.",
            "kind": "calc",
        },
        "calc_corners_index": {
            "doc": "Box corners in lattice-index space.",
            "kind": "calc",
        },
        "calc_corners": {
            "doc": "Bounds object describing the dataset box in real-space coordinates.",
            "kind": "calc",
        },
        "calc_box_size_periodic_index": {
            "doc": (
                "Effective periodic box size in index units. "
                "For periodic dims equals grid size, otherwise inf."
            ),
            "kind": "calc",
        },
        "fields": {
            "doc": "Registry of physical fields bound to this shared grid.",
            "kind": "relation",
            "is_weak_by_default": False,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
    }
    # fmt: on

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.get("kind") not in ("relation", "property")
        and name not in ClassBase.__slots__
    )

    def __init__(
        self,
        inputValue: InputGridField | None = None,
        name: str = "grid field dataset",
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            name_replace="grid field dataset",
            is_fixed=True,
        )

        if inputValue is None:
            inputValue = InputGridField()
        if kwargs:
            input_kwargs = {
                f.name: getattr(inputValue, f.name) for f in fields(inputValue)
            }
            input_kwargs.update(kwargs)
            inputValue = replace(inputValue, **input_kwargs)

        object.__setattr__(self, "raw_shape", inputValue.shape)
        object.__setattr__(
            self,
            "raw_box_periodic_flag",
            inputValue.box_periodic_flag,
        )
        object.__setattr__(self, "raw_grid_offset", inputValue.grid_offset)
        object.__setattr__(self, "raw_grid_transform", inputValue.grid_transform)
        self._helper_refresh_grid_cache()

        registry = RegistryBase(
            "fields manager",
            info=f"physical fields attached to dataset {self.name!r}",
        )
        self.act_bind_relation_base("fields", registry, is_weak=False)
        registry.act_bind_relation_base("owner", self, is_weak=True)

    def _helper_ensure_or_infer_shape(
        self,
        values: np.ndarray,
    ) -> tuple[int, int, int]:
        """Infer the dataset shape once, then require every field to match it."""
        field_shape = as_grid_shape(np.shape(values)[:3], name="field grid shape")
        if self.raw_shape is UNSET:
            object.__setattr__(self, "raw_shape", field_shape)
            self._helper_refresh_grid_cache()
            return field_shape
        if field_shape != tuple(self.raw_shape):
            raise ValueError(
                "Field grid shape must match the dataset grid shape. "
                f"Dataset shape is {self.raw_shape}; field shape is {field_shape}."
            )
        return field_shape

    def _helper_as_field_values_on_grid(
        self,
        field_or_values,
        *,
        name: str = "field values",
    ) -> np.ndarray:
        """
        Return numeric field values whose leading axes match this dataset grid.

        `field_or_values` may be a registered field name/index, a `FieldData`
        object owned by this dataset, or a temporary NumPy-like array. Temporary
        arrays are not registered or cached.
        """
        if self.raw_shape is UNSET:
            raise ValueError(
                "Dataset shape must be known before spatial derivatives can be "
                "computed. Add a field first or initialize the dataset with a "
                "shape."
            )

        if isinstance(field_or_values, FieldData):
            field = field_or_values
            if field.owner is not self:
                raise ValueError(
                    "FieldData input must belong to this GridFieldDataset."
                )
            values = field.raw_values
        elif (
            field_or_values is None
            or isinstance(field_or_values, str)
            or isinstance(field_or_values, int)
        ):
            values = self.act_get_field(field_or_values).raw_values
        else:
            values = as_field_values(field_or_values, name=name)

        field_shape = as_grid_shape(np.shape(values)[:3], name=f"{name} grid shape")
        dataset_shape = as_grid_shape(self.raw_shape, name="dataset grid shape")
        if field_shape != dataset_shape:
            raise ValueError(
                f"{name!r} leading grid shape must match the dataset grid shape. "
                f"Dataset shape is {dataset_shape}; got {field_shape}."
            )
        return values

    def _helper_source_name_for_field_values(self, field_or_values) -> str | None:
        """Return a registered field name when derivative input has one."""
        if isinstance(field_or_values, FieldData):
            return field_or_values.name
        if (
            field_or_values is None
            or isinstance(field_or_values, str)
            or isinstance(field_or_values, int)
        ):
            return self.act_get_field(field_or_values).name
        return None

    def _helper_as_direction_index(self, direction: str | int) -> int:
        """Return a derivative direction index from 0/1/2 or x/y/z input."""
        direction_map = {"x": 0, "y": 1, "z": 2}
        if isinstance(direction, str):
            try:
                return direction_map[direction.lower()]
            except KeyError as exc:
                raise ValueError(
                    "direction must be one of 0, 1, 2, 'x', 'y', or 'z'."
                ) from exc
        if isinstance(direction, int) and direction in (0, 1, 2):
            return direction
        raise ValueError("direction must be one of 0, 1, 2, 'x', 'y', or 'z'.")

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
        grid_offset = (
            None
            if self.raw_grid_offset is None
            else np.asarray(self.raw_grid_offset).copy()
        )
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
            grid_transform=self.raw_grid_transform,
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

    # -------------------------------
    # Shared-grid geometry cache
    # -------------------------------

    def _helper_refresh_grid_cache(self) -> None:
        """Refresh geometry caches from the current shared grid metadata."""
        if self.raw_shape is UNSET:
            object.__setattr__(self, "calc_grid_index", UNSET)
            object.__setattr__(self, "calc_grid", UNSET)
            object.__setattr__(self, "calc_corners_index", UNSET)
            object.__setattr__(self, "calc_corners", UNSET)
            object.__setattr__(self, "calc_box_size_periodic_index", UNSET)
            return

        grid_shape = as_grid_shape(self.raw_shape, name="dataset grid shape")

        box_size_periodic_index = np.zeros(3, dtype=float)
        for i, is_periodic in enumerate(self.raw_box_periodic_flag):
            if is_periodic:
                box_size_periodic_index[i] = grid_shape[i]
            else:
                box_size_periodic_index[i] = np.inf

        grid_index = generate_coordinate_grid(grid_shape, grid_shape)[0]
        grid = apply_linear_transform(
            grid_index,
            transform=self.raw_grid_transform,
            offset=self.raw_grid_offset,
        )

        lengths_index = np.asarray(grid_shape) - np.array([1, 1, 1])
        corners_index = get_box_corners(*lengths_index)
        corners_coord = apply_linear_transform(
            corners_index,
            transform=self.raw_grid_transform,
            offset=self.raw_grid_offset,
        )
        corners = as_bounds(
            corners_coord,
            name=f"Bounds of grid field dataset {self.name!r}",
        )

        object.__setattr__(
            self,
            "calc_box_size_periodic_index",
            box_size_periodic_index,
        )
        object.__setattr__(self, "calc_grid_index", grid_index)
        object.__setattr__(self, "calc_grid", grid)
        object.__setattr__(self, "calc_corners_index", corners_index)
        object.__setattr__(self, "calc_corners", corners)

    def act_add_field(
        self,
        name: str,
        values,
        *,
        info=None,
        is_replace: bool = False,
    ) -> FieldData:
        """Validate and bind one physical field to this dataset."""
        values = as_field_values(values, name=f"field {name!r} values")
        self._helper_ensure_or_infer_shape(values)

        try:
            existing_field = self.fields[name]
        except KeyError:
            existing_field = None
        if existing_field is not None and not is_replace:
            raise ValueError(
                f"Field {name!r} already exists in this dataset. "
                "Pass is_replace=True to replace it."
            )

        if is_replace:
            if existing_field is not None:
                self.fields.act_unregister(existing_field, is_missing_ok=True)

        field = FieldData(name=name, values=values, info=info)
        field.act_bind_relation_base("owner", self, is_weak=True)
        self.fields.act_register(field)
        return field

    def act_add_result_field(
        self,
        name: str,
        result: SpatialDerivativeResult,
        *,
        is_replace: bool = False,
    ) -> FieldData:
        """
        Register a spatial derivative result as a field with payload-free info.

        This stores `result.raw_values` as the field values and
        `result.raw_info` as the field info, avoiding duplication of the result
        payload inside metadata.
        """
        if not isinstance(result, SpatialDerivativeResult):
            raise TypeError(
                "result must be a SpatialDerivativeResult returned by a "
                "dataset spatial derivative helper."
            )
        return self.act_add_field(
            name,
            result.raw_values,
            info=result.raw_info,
            is_replace=is_replace,
        )

    def act_gradient(
        self,
        field_or_values,
        *,
        coord: str = "physical",
        is_result: bool = False,
    ) -> np.ndarray | SpatialDerivativeResult:
        """
        Return the spatial gradient of a field on this dataset grid.

        Finite differences are computed along the first three lattice/index
        axes. The returned array has one additional final axis of length 3,
        representing the derivative direction.
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
        grad = np.empty(values.shape + (3,), dtype=float)

        for axis, is_periodic in enumerate(self.raw_box_periodic_flag):
            axis_size = values.shape[axis]
            if axis_size == 1:
                derivative = np.zeros_like(values, dtype=float)
            elif is_periodic:
                derivative = (
                    np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)
                ) / 2.0
            else:
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

        `direction` may be 0/1/2 or "x"/"y"/"z". The result is a slice of
        `act_gradient(..., coord=coord)` along the final derivative axis.
        """
        direction_index = self._helper_as_direction_index(direction)
        gradient = self.act_gradient(field_or_values, coord=coord)
        values = gradient[..., direction_index]

        if not is_result:
            return values
        source_values = self._helper_as_field_values_on_grid(
            field_or_values,
            name="derivative input values",
        )
        return self._helper_spatial_derivative_result(
            values,
            operator="derivative",
            source=self._helper_source_name_for_field_values(field_or_values),
            source_shape=np.shape(source_values),
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
        axis is contracted with the final derivative axis returned by
        `act_gradient()`.
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

        grad = self.act_gradient(values, coord=coord)
        div = np.einsum("...ii->...", grad)

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
            if values.ndim > 4 and any(
                axis_size == 3 for axis_size in values.shape[3:]
            ):
                raise ValueError(
                    "curl input values look like a tensor-valued field. "
                    "Please use a tensor-specific curl helper so the tensor "
                    "component axis convention is explicit."
                )
            raise ValueError(
                "curl input values must be a vector field with shape "
                f"{tuple(self.raw_shape)} + (3,). Got shape {values.shape}."
            )

        grad = self.act_gradient(values, coord=coord)
        curl = np.empty_like(values, dtype=float)
        curl[..., 0] = grad[..., 2, 1] - grad[..., 1, 2]
        curl[..., 1] = grad[..., 0, 2] - grad[..., 2, 0]
        curl[..., 2] = grad[..., 1, 0] - grad[..., 0, 1]

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
        grad = self.act_gradient(values_moved, coord=coord)
        curl_moved = np.empty_like(values_moved, dtype=float)
        curl_moved[..., 0] = grad[..., 2, 1] - grad[..., 1, 2]
        curl_moved[..., 1] = grad[..., 0, 2] - grad[..., 2, 0]
        curl_moved[..., 2] = grad[..., 1, 0] - grad[..., 0, 1]
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

    def act_symmetric_gradient(
        self,
        field_or_values,
        *,
        coord: str = "physical",
        is_result: bool = False,
    ) -> np.ndarray | SpatialDerivativeResult:
        """
        Return the symmetric part of a vector-field gradient.

        The input field must have shape `(Nx, Ny, Nz, 3)`. The returned tensor
        is `0.5 * (grad_v + grad_v.T)` over the vector-component and derivative
        axes.
        """
        source = self._helper_source_name_for_field_values(field_or_values)
        values = self._helper_as_field_values_on_grid(
            field_or_values,
            name="symmetric gradient input values",
        )
        if values.shape[3:] != (3,):
            raise ValueError(
                "symmetric gradient input values must be a vector field with "
                f"shape {tuple(self.raw_shape)} + (3,). Got shape {values.shape}."
            )

        grad = self.act_gradient(values, coord=coord)
        symmetric = 0.5 * (grad + np.swapaxes(grad, -1, -2))

        if not is_result:
            return symmetric
        return self._helper_spatial_derivative_result(
            symmetric,
            operator="symmetric_gradient",
            source=source,
            source_shape=values.shape,
            coord=coord,
            derivative_axis=None,
        )

    def act_antisymmetric_gradient(
        self,
        field_or_values,
        *,
        coord: str = "physical",
        is_result: bool = False,
    ) -> np.ndarray | SpatialDerivativeResult:
        """
        Return the antisymmetric part of a vector-field gradient.

        The input field must have shape `(Nx, Ny, Nz, 3)`. The returned tensor
        is `0.5 * (grad_v - grad_v.T)` over the vector-component and derivative
        axes.
        """
        source = self._helper_source_name_for_field_values(field_or_values)
        values = self._helper_as_field_values_on_grid(
            field_or_values,
            name="antisymmetric gradient input values",
        )
        if values.shape[3:] != (3,):
            raise ValueError(
                "antisymmetric gradient input values must be a vector field "
                f"with shape {tuple(self.raw_shape)} + (3,). Got shape {values.shape}."
            )

        grad = self.act_gradient(values, coord=coord)
        antisymmetric = 0.5 * (grad - np.swapaxes(grad, -1, -2))

        if not is_result:
            return antisymmetric
        return self._helper_spatial_derivative_result(
            antisymmetric,
            operator="antisymmetric_gradient",
            source=source,
            source_shape=values.shape,
            coord=coord,
            derivative_axis=None,
        )

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
        this helper. This is the discrete composition of `act_gradient()` and
        `act_divergence()`, so one-sided boundary stencils affect one additional
        layer of points near non-periodic boundaries.
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

    def act_get_field(self, name: str | int | None):
        """Return one bound field by name or registry index."""
        return self.fields[name]

    def __getitem__(self, name: str | int | None):
        """Shortcut for act_get_field."""
        return self.act_get_field(name)
