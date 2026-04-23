"""Shared-grid dataset for binding multiple physical fields."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any, ClassVar, Mapping

import numpy as np

from nematics3d.datatypes import (
    UNSET,
    Unset,
)

from ..bounds import as_bounds
from ..class_base import ClassBase
from ..registry_base import RegistryBase
from ...field import apply_linear_transform, generate_coordinate_grid
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
    ):
        values = self.__attr_defs__["raw_values"]["validator"](
            values,
            self.__attr_defs__["raw_values"]["doc"],
        )

        super().__init__(name=name, name_replace="field")
        object.__setattr__(self, "raw_values", values)


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
        super().__init__(name=name, name_replace="grid field dataset")

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

        field = FieldData(name=name, values=values)
        field.act_bind_relation_base("owner", self, is_weak=True)
        self.fields.act_register(field)
        return field

    def act_get_field(self, name: str | int | None):
        """Return one bound field by name or registry index."""
        return self.fields[name]

    def __getitem__(self, name: str | int | None):
        """Shortcut for act_get_field."""
        return self.act_get_field(name)
