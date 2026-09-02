"""Shared-grid dataset core for binding multiple physical fields."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import ClassVar, Literal

import numpy as np

from nematics3d.datatypes import (
    UNSET,
    Unset,
    as_grid_shape,
    as_lattice_mask,
    as_readonly_array,
    as_real_lattice_field,
)

from ..bounds import as_bounds
from ...core.class_base import AttrDef, ClassBase
from ...core.registry_base import RegistryBase
from ...grid import (
    VALIDITY_FIELD_NAME,
    apply_linear_transform,
    as_grid_offset,
    as_grid_transform,
    generate_coordinate_grid,
    is_grid_transform_identity,
)
from ...core.npy_array_payload import NpyArrayPayload
from ...geometry import get_box_corners
from .grid_field_dataset_derivatives import (
    GridFieldDatasetDerivativeMixin,
    SpatialDerivativeInfo,
)
from .grid_field_dataset_smoothing import (
    GaussianSmoothInfo,
    GridFieldDatasetSmoothingMixin,
)
from .input_grid_field import InputGridField


class FieldData(ClassBase):
    """Thin wrapper for one physical field living on a GridFieldDataset."""

    # fmt: off
    __attr_defs__: ClassVar = {
        "raw_values": AttrDef(
            doc="Field values with leading axes matching the dataset grid.",
            kind="raw",
            validator=as_real_lattice_field,
        ),
        "raw_info": AttrDef(
            doc="Optional user-provided metadata or provenance for this field.",
            kind="raw",
        ),
        "entity_interpolator": AttrDef(
            doc="The generic interpolator associated with this field.",
            kind="entity",
        ),
        "interpolator": AttrDef(
            doc="Read-only public access to the field-owned interpolator.",
            kind="property",
        ),
    }
    # fmt: on

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
        and name not in ClassBase.__slots__
    )

    def __init__(
        self,
        name: str,
        values,
        info=None,
    ):
        if name == VALIDITY_FIELD_NAME:
            values = as_lattice_mask(values, name="dataset validity mask")
        else:
            values = (
                type(self)
                .__attr_defs__["raw_values"]
                .validator(
                    values,
                    type(self).__attr_defs__["raw_values"].doc,
                )
            )
        values = as_readonly_array(
            values,
            dtype=bool if name == VALIDITY_FIELD_NAME else float,
        )

        super().__init__(name=name, name_replace="field", is_fixed=True)
        object.__setattr__(self, "raw_values", values)
        object.__setattr__(self, "raw_info", info)
        object.__setattr__(self, "entity_interpolator", None)

    def act_add_interpolator(self):
        """Create and bind a GridInterpolator if one is not already present."""
        from .grid_interpolator import GridInterpolator

        interpolator_old = self.entity_interpolator
        if isinstance(interpolator_old, GridInterpolator):
            return interpolator_old

        interpolator = GridInterpolator(self, name=f"{self.name} interpolator")
        object.__setattr__(self, "entity_interpolator", interpolator)
        return self.entity_interpolator

    @property
    def interpolator(self):
        """Return the field-owned interpolator, or None before creation."""
        return self.entity_interpolator

    def act_interpolate(
        self,
        points: np.ndarray,
        is_index: bool = False,
        is_out_warning: bool = False,
        is_return_validity: bool = False,
    ):
        """Interpolate this field at arbitrary sample points."""
        if self.entity_interpolator is None:
            self.act_add_interpolator()
        return self.entity_interpolator.interpolate(
            points,
            is_index=is_index,
            is_out_warning=is_out_warning,
            is_return_validity=is_return_validity,
        )


class GridFieldDataset(
    GridFieldDatasetDerivativeMixin,
    GridFieldDatasetSmoothingMixin,
    ClassBase,
):
    """Container for physical fields sharing one immutable lattice.

    Shape, periodicity, offset, and transform are fixed after construction.
    When shape is initially ``UNSET``, the first registered field may infer it
    exactly once; all later fields must match. Create a new dataset to use
    different grid geometry.
    """

    # fmt: off
    __attr_defs__: ClassVar = {
        "raw_shape": AttrDef(
            doc="Shared lattice grid shape (Nx, Ny, Nz), or UNSET before inference.",
            kind="raw",
        ),
        "raw_box_periodic_flag": AttrDef(
            doc="Per-dimension periodic boundary condition flags.",
            kind="raw",
        ),
        "raw_grid_offset": AttrDef(
            doc="Grid translation offset mapping lattice indices to real space.",
            kind="raw",
        ),
        "raw_grid_transform": AttrDef(
            doc="Linear transform mapping lattice indices to real space.",
            kind="raw",
        ),
        "calc_corners_index": AttrDef(
            doc="Box corners in lattice-index space.",
            kind="calc",
        ),
        "calc_corners": AttrDef(
            doc="Box corners in real-space coordinates.",
            kind="calc",
        ),
        "calc_bounds": AttrDef(
            doc="Bounds object describing the dataset box in real-space coordinates.",
            kind="calc",
        ),
        "calc_grid_spacing": AttrDef(
            doc="Real-space spacing along each lattice axis.",
            kind="calc",
        ),
        "calc_center": AttrDef(
            doc=(
                "Read-only: Geometric center of the dataset box in real-space "
                "coordinates after the grid transform and offset."
            ),
            kind="property",
        ),
        "calc_box_size_periodic_index": AttrDef(
            doc=(
                "Effective periodic box size in index units. "
                "For periodic dims equals grid size, otherwise inf."
            ),
            kind="calc",
        ),
        "calc_is_has_mask": AttrDef(
            doc=(
                "Read-only: whether this dataset carries a per-voxel validity "
                "mask field bound at construction."
            ),
            kind="property",
        ),
        "mask": AttrDef(
            doc=(
                "Read-only canonical boolean validity mask, or None when the "
                "dataset was constructed without one."
            ),
            kind="property",
        ),
        "fields": AttrDef(
            doc="Registry of physical fields bound to this shared grid.",
            kind="relation",
            is_weak_by_default=False,
        ),
    }
    # fmt: on

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
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
            as_readonly_array(inputValue.box_periodic_flag, dtype=bool),
        )
        object.__setattr__(
            self,
            "raw_grid_offset",
            as_grid_offset(inputValue.grid_offset, is_readonly=True),
        )
        object.__setattr__(
            self,
            "raw_grid_transform",
            as_grid_transform(inputValue.grid_transform, is_readonly=True),
        )
        self._helper_initialize_geometry()

        registry = RegistryBase(
            "fields manager",
            info=f"physical fields attached to dataset {self.name!r}",
        )
        self.act_bind_relation_base("fields", registry, is_weak=False)
        registry.act_bind_relation_base("owner", self, is_weak=True)

        # The validity mask is the only field that may be supplied through the
        # dataset constructor. It is bound here, once, via the internal channel
        # that bypasses the act_add_field guard. A dataset built without a mask
        # can never gain one later; this keeps every mask-dependent result
        # (defects, smoothing, interpolation validity) from silently going
        # stale, because the mask is fixed before any of them is computed.
        if inputValue.mask is not UNSET:
            # Validate dtype/range/rank here; the grid-shape contract (match an
            # existing shape, or infer and lock it when unset) is enforced once
            # for every field by _helper_ensure_or_infer_shape inside
            # _helper_add_field, so it is not duplicated here.
            mask_values = as_lattice_mask(
                inputValue.mask,
                name="dataset validity mask",
            )
            self._helper_add_field(VALIDITY_FIELD_NAME, mask_values)

    def _helper_ensure_or_infer_shape(
        self,
        values: np.ndarray,
    ) -> tuple[int, int, int]:
        """Infer the dataset shape once, then require every field to match it."""
        field_shape = as_grid_shape(np.shape(values)[:3], name="field grid shape")
        if self.raw_shape is UNSET:
            object.__setattr__(self, "raw_shape", field_shape)
            self._helper_initialize_geometry()
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

    # -------------------------------
    # Shared-grid geometry initialization
    # -------------------------------

    def _helper_initialize_geometry(self) -> None:
        """Initialize canonical geometry once after shape becomes available."""
        try:
            bounds_existing = object.__getattribute__(self, "calc_bounds")
        except AttributeError:
            bounds_existing = UNSET

        if bounds_existing is not UNSET:
            raise RuntimeError(
                "GridFieldDataset geometry is already initialized and cannot "
                "be refreshed or replaced. Create a new dataset for different "
                "geometry."
            )

        if self.raw_shape is UNSET:
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
            grid_spacing = np.linalg.norm(self.raw_grid_transform, axis=1)

        box_size_periodic_index = np.zeros(3, dtype=float)
        for i, is_periodic in enumerate(self.raw_box_periodic_flag):
            if is_periodic:
                box_size_periodic_index[i] = grid_shape[i]
            else:
                box_size_periodic_index[i] = np.inf

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
        object.__setattr__(self, "calc_corners_index", corners_index)
        object.__setattr__(self, "calc_corners", corners_coord)
        object.__setattr__(self, "calc_bounds", bounds)
        object.__setattr__(self, "calc_grid_spacing", grid_spacing)

    # -------------------------------
    # Coordinate conversion and explicit full-grid allocation
    # -------------------------------

    def act_index_to_coord(self, indices) -> np.ndarray:
        """Convert lattice-index points to physical coordinates.

        ``indices`` may be one point with shape ``(3,)`` or an array with any
        leading shape and a trailing coordinate axis of length three.
        """
        return apply_linear_transform(
            indices,
            transform=self.raw_grid_transform,
            offset=self.raw_grid_offset,
        )

    def act_coord_to_index(self, coords) -> np.ndarray:
        """Convert physical-coordinate points to lattice-index coordinates.

        ``coords`` may be one point with shape ``(3,)`` or an array with any
        leading shape and a trailing coordinate axis of length three.
        """
        return apply_linear_transform(
            coords,
            transform=self.raw_grid_transform,
            offset=self.raw_grid_offset,
            is_inv=True,
        )

    def act_generate_grid(
        self,
        coord: Literal["index", "physical"] = "physical",
        *,
        dtype=np.float64,
    ) -> np.ndarray:
        """Allocate and return the complete lattice coordinate grid.

        This operation is intentionally explicit because the returned array
        has shape ``(*raw_shape, 3)`` and can be very large. The result is not
        cached by the dataset.

        Physical grids are written directly into their final array, avoiding
        allocation of a second full-size index grid as an intermediate.
        """
        if self.raw_shape is UNSET:
            raise ValueError(
                "Dataset shape must be known before a full coordinate grid "
                "can be generated. Add a field first or initialize the "
                "dataset with a shape."
            )
        if coord not in ("index", "physical"):
            raise ValueError("coord must be either 'index' or 'physical'.")

        dtype = np.dtype(dtype)
        if not np.issubdtype(dtype, np.floating):
            raise TypeError("dtype must be a floating NumPy dtype.")

        grid_shape = as_grid_shape(self.raw_shape, name="dataset grid shape")
        if coord == "index":
            return generate_coordinate_grid(grid_shape, grid_shape).astype(
                dtype,
                copy=False,
            )

        grid = np.zeros((*grid_shape, 3), dtype=dtype)
        if is_grid_transform_identity(self.raw_grid_transform):
            transform = np.eye(3, dtype=dtype)
        else:
            transform = np.asarray(self.raw_grid_transform, dtype=dtype)
        ndim = len(grid_shape)
        for index_axis, axis_size in enumerate(grid_shape):
            axis = np.arange(axis_size, dtype=dtype)
            axis_shape = [1] * ndim
            axis_shape[index_axis] = axis_size
            axis = axis.reshape(axis_shape)
            for coord_axis in range(3):
                coefficient = transform[index_axis, coord_axis]
                if coefficient != 0.0:
                    grid[..., coord_axis] += axis * coefficient

        if self.raw_grid_offset is not None:
            grid += np.asarray(self.raw_grid_offset, dtype=dtype)
        return grid

    def act_add_field(
        self,
        name: str,
        values,
        *,
        info=None,
    ) -> FieldData:
        """Validate and bind one immutable physical-field snapshot."""
        if name == VALIDITY_FIELD_NAME:
            raise ValueError(
                f"{VALIDITY_FIELD_NAME!r} is a reserved validity mask field and "
                "cannot be added through act_add_field. Supply the "
                "mask when constructing the dataset (InputGridField.mask); a "
                "dataset built without a mask cannot gain one later."
            )
        return self._helper_add_field(name, values, info=info)

    def _helper_add_field(
        self,
        name: str,
        values,
        *,
        info=None,
    ) -> FieldData:
        """Validate and bind one physical field, bypassing the name guard."""
        if name == VALIDITY_FIELD_NAME:
            values = as_lattice_mask(values, name="dataset validity mask")
        else:
            values = as_real_lattice_field(values, name=f"field {name!r} values")
        self._helper_ensure_or_infer_shape(values)

        try:
            existing_field = self.fields[name]
        except KeyError:
            existing_field = None
        if existing_field is not None:
            raise ValueError(
                f"Field {name!r} already exists in this dataset. Registered "
                "fields are immutable snapshots and cannot be replaced."
            )

        field = FieldData(name=name, values=values, info=info)
        field.act_bind_relation_base("owner", self, is_weak=True)
        self.fields.act_register(field)
        return field

    def act_add_result_field(
        self,
        name: str,
        result: NpyArrayPayload,
    ) -> FieldData:
        """
        Register a dataset operator result as a field with payload-free info.

        This stores the result values as the field payload and
        `result.raw_info` as the field info, avoiding duplication of the result
        payload inside metadata. Released results are loaded temporarily from
        `result.raw_path` through `act_with_values()`.
        """
        if not isinstance(result, NpyArrayPayload):
            raise TypeError(
                "result must be a dataset result returned by a supported "
                "dataset operator helper."
            )
        with result.act_with_values() as values:
            return self.act_add_field(
                name,
                values,
                info=result.raw_info,
            )

    def act_get_field(self, name: str | int | None):
        """Return one bound field by name or registry index."""
        return self.fields[name]

    def __getitem__(self, name: str | int | None):
        """Shortcut for act_get_field."""
        return self.act_get_field(name)

    def _helper_read_validity_mask(self):
        """Return the validity mask as a bool array, or None when absent."""
        return self.mask

    @property
    def mask(self):
        """Return the canonical read-only boolean validity mask, if present."""
        try:
            return self.fields[VALIDITY_FIELD_NAME].raw_values
        except KeyError:
            return None

    @property
    def calc_is_has_mask(self) -> bool:
        """Return whether a validity mask field is bound to this dataset."""
        try:
            self.fields[VALIDITY_FIELD_NAME]
        except KeyError:
            return False
        return True

    @property
    def calc_center(self):
        """Return the transformed geometric center of the dataset box."""
        if self.raw_shape is UNSET:
            return UNSET

        center_index = 0.5 * (np.asarray(self.raw_shape, dtype=float) - 1.0)
        return apply_linear_transform(
            center_index[np.newaxis, :],
            transform=self.raw_grid_transform,
            offset=self.raw_grid_offset,
        )[0]
