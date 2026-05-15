"""Shared-grid dataset for binding multiple physical fields."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, ClassVar, Iterator, Mapping

import numpy as np

from nematics3d.datatypes import (
    UNSET,
    Unset,
    as_real_lattice_field,
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
            "validator": as_real_lattice_field,
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

        super().__init__(name=name, name_replace="field", is_fixed=True)
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
            "doc": "Box corners in real-space coordinates.",
            "kind": "calc",
        },
        "calc_bounds": {
            "doc": "Bounds object describing the dataset box in real-space coordinates.",
            "kind": "calc",
        },
        "calc_grid_spacing": {
            "doc": "Real-space spacing along each lattice axis.",
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
        object.__setattr__(
            self,
            "raw_grid_offset",
            self._helper_readonly_grid_array_copy(inputValue.grid_offset),
        )
        object.__setattr__(
            self,
            "raw_grid_transform",
            self._helper_readonly_grid_array_copy(inputValue.grid_transform),
        )
        self._helper_refresh_grid_cache()

        registry = RegistryBase(
            "fields manager",
            info=f"physical fields attached to dataset {self.name!r}",
        )
        self.act_bind_relation_base("fields", registry, is_weak=False)
        registry.act_bind_relation_base("owner", self, is_weak=True)

    @staticmethod
    def _helper_readonly_grid_array_copy(value):
        """Return a read-only copy for mutable grid geometry arrays."""
        if value is None or is_grid_transform_identity(value):
            return value
        value = np.asarray(value, dtype=float).copy()
        value.setflags(write=False)
        return value

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
            values = as_real_lattice_field(field_or_values, name=name)

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
            return (
                np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)
            ) / 2.0

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
            return (
                np.roll(values, -1, axis=axis)
                - 2.0 * values
                + np.roll(values, 1, axis=axis)
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
            object.__setattr__(self, "calc_bounds", UNSET)
            object.__setattr__(self, "calc_grid_spacing", UNSET)
            object.__setattr__(self, "calc_box_size_periodic_index", UNSET)
            return

        grid_shape = as_grid_shape(self.raw_shape, name="dataset grid shape")
        if is_grid_transform_identity(self.raw_grid_transform):
            grid_spacing = np.ones(3, dtype=float)
        else:
            grid_spacing = np.linalg.norm(self.raw_grid_transform, axis=0)

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
        bounds = as_bounds(
            corners_coord,
            name=f"Bounds of grid field dataset {self.name!r}",
        )
        bounds.act_register_protected_opts_all()

        object.__setattr__(
            self,
            "calc_box_size_periodic_index",
            box_size_periodic_index,
        )
        object.__setattr__(self, "calc_grid_index", grid_index)
        object.__setattr__(self, "calc_grid", grid)
        object.__setattr__(self, "calc_corners_index", corners_index)
        object.__setattr__(self, "calc_corners", corners_coord)
        object.__setattr__(self, "calc_bounds", bounds)
        object.__setattr__(self, "calc_grid_spacing", grid_spacing)

    def act_add_field(
        self,
        name: str,
        values,
        *,
        info=None,
        is_replace: bool = False,
    ) -> FieldData:
        """Validate and bind one physical field to this dataset."""
        values = as_real_lattice_field(values, name=f"field {name!r} values")
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

        This stores the result values as the field payload and
        `result.raw_info` as the field info, avoiding duplication of the result
        payload inside metadata. Released results are loaded temporarily from
        `result.raw_path` through `act_with_values()`.
        """
        if not isinstance(result, SpatialDerivativeResult):
            raise TypeError(
                "result must be a SpatialDerivativeResult returned by a "
                "dataset spatial derivative helper."
            )
        with result.act_with_values() as values:
            return self.act_add_field(
                name,
                values,
                info=result.raw_info,
                is_replace=is_replace,
            )

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

        source, values, strain_rate, vorticity_tensor = (
            self._helper_vector_gradient_split(
                field_or_values,
                coord=coord,
                name="velocity gradient split input values",
            )
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

    def act_get_field(self, name: str | int | None):
        """Return one bound field by name or registry index."""
        return self.fields[name]

    def __getitem__(self, name: str | int | None):
        """Shortcut for act_get_field."""
        return self.act_get_field(name)
