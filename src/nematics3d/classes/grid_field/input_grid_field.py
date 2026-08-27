"""Input validation bundle for shared-grid field datasets."""

from dataclasses import dataclass
from typing import ClassVar, Mapping

import numpy as np

from nematics3d.datatypes import (
    DimensionInfo,
    MaskField,
    UNSET,
    Unset,
    Vect,
    as_dimension_info,
    as_lattice_mask,
)
from nematics3d.grid import (
    GRID_TRANSFORM_IDENTITY,
    GridTransform,
    as_grid_offset,
    as_grid_transform,
)


def as_grid_shape(value, name: str = "grid shape") -> tuple[int, int, int]:
    """Validate a 3D grid shape and return it as a tuple of positive integers."""
    if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != 3:
        raise ValueError(
            f"{name!r} must be a length-3 shape like (Nx, Ny, Nz). "
            f"Got {value!r} instead."
        )

    shape = []
    for dim in value:
        if not isinstance(dim, (int, np.integer)):
            raise TypeError(
                f"{name!r} entries must be integers. Got {value!r} instead."
            )
        dim = int(dim)
        if dim <= 0:
            raise ValueError(
                f"{name!r} entries must be positive. Got {value!r} instead."
            )
        shape.append(dim)

    return tuple(shape)


@dataclass(slots=True)
class InputGridField:
    """
    Validated input bundle for a shared-grid field dataset.

    This object describes only the common spatial grid. Physical data such as
    Q tensors, velocity, active force, and concentration should be bound later
    to the owning dataset, where their leading grid shape can be checked against
    this metadata.

    Parameters
    ----------
    shape
        Optional lattice shape ``(Nx, Ny, Nz)``. If omitted, a dataset may infer
        it from the first field that is bound.
    box_periodic_flag
        Periodic-boundary-condition flags for the three lattice directions.
    grid_offset
        Translation offset that maps lattice indices to real-space coordinates.
    grid_transform
        3x3 linear transform that maps lattice indices to real-space
        coordinates, or ``GRID_TRANSFORM_IDENTITY`` for the identity map.
    mask
        Optional per-voxel validity mask of shape ``(Nx, Ny, Nz)``. True marks
        voxels whose data is physically meaningful. The mask can only be
        supplied here, at dataset construction; once bound it is immutable, and
        a dataset built without a mask can never gain one later.
    """

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
        "shape": lambda v, d: as_grid_shape(v, name=d),
        "box_periodic_flag": lambda v, d: as_dimension_info(
            v,
            name=d,
            is_bool=True,
        ),
        "grid_offset": lambda v, d: as_grid_offset(v, name=d),
        "grid_transform": lambda v, d: as_grid_transform(v, name=d),
        "mask": lambda v, d: as_lattice_mask(v, name=d),
    }

    # Keep the InputQ-style assignment contract: validation runs both during
    # dataclass initialization and during later interactive edits.
    def __setattr__(self, key, value):
        if key in self._validators and value is not UNSET:
            desc = f"{key!r}: {self.__class__.__attrs__[key]}"
            value = self._validators[key](value, desc)
        object.__setattr__(self, key, value)
