"""Input validation bundle for shared-grid field datasets."""

from dataclasses import dataclass
from typing import ClassVar, Mapping

from nematics3d.datatypes import (
    DimensionInfo,
    MaskField,
    UNSET,
    Unset,
    Vect,
    as_dimension_info,
    as_grid_shape,
    as_lattice_mask,
)
from nematics3d.grid import (
    GRID_TRANSFORM_IDENTITY,
    GridTransform,
    as_grid_offset,
    as_grid_transform,
)


@dataclass(slots=True)
class InputGridField:
    """Validated input bundle for a shared-grid field dataset."""

    shape: tuple[int, int, int] | Unset = UNSET
    box_periodic_flag: DimensionInfo = False
    grid_offset: Vect(3) | None = None
    grid_transform: GridTransform = GRID_TRANSFORM_IDENTITY
    mask: MaskField | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        "shape": "lattice grid shape (Nx, Ny, Nz)",
        "box_periodic_flag": (
            "flag indicating whether periodic boundary condition is applied "
            "along each dimension"
        ),
        "grid_offset": (
            "grid translation offset to map lattice indices to real-space "
            "coordinates"
        ),
        "grid_transform": (
            "grid transform matrix to map lattice indices to real-space "
            "coordinates (3x3)"
        ),
        "mask": (
            "per-voxel validity mask marking which voxels carry physically "
            "meaningful data"
        ),
    }

    _validators: ClassVar[Mapping[str, object]] = {
        "shape": lambda v, d: as_grid_shape(v, name=d, is_strict_3d=True),
        "box_periodic_flag": lambda v, d: as_dimension_info(
            v,
            name=d,
            is_bool=True,
        ),
        "grid_offset": lambda v, d: as_grid_offset(v, name=d),
        "grid_transform": lambda v, d: as_grid_transform(v, name=d),
        "mask": lambda v, d: as_lattice_mask(v, name=d),
    }

    def __setattr__(self, key, value):
        if key in self._validators and value is not UNSET:
            desc = f"{key!r}: {self.__class__.__attrs__[key]}"
            value = self._validators[key](value, desc)
        object.__setattr__(self, key, value)
