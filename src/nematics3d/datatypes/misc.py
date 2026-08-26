"""
Miscellaneous semantic data aliases and runtime conversion helpers.

This module contains helpers that do not yet have a dedicated datatype module.
Dedicated validators are re-exported here temporarily for compatibility.
"""

import numpy as np

from .axes import as_axes
from .color_rgb import ColorRGB, as_ColorRGB, as_ColorRGB_array
from .list import as_list
from .number import Number, as_number, as_value_range
from .string import as_str
from .tensor import Tensor, as_tensor
from .vector import Vect, as_vector


def as_readonly_array(input_data, *, dtype=float, copy: bool = True) -> np.ndarray:
    """Return one NumPy array view/copy with write access disabled."""
    values = np.asarray(input_data, dtype=dtype)
    if copy:
        values = values.copy()
    values.setflags(write=False)
    return values


# -------------------------
# Physical field types
# -------------------------

# All fields are NumPy arrays defined over a 3D grid of shape (Nx, Ny, Nz).
#
# `GeneralField` is the abstract base type of all physical fields, where each voxel
# may hold scalar, vector, tensor, or feature-vector data.

# -------------------------
# Base type
# -------------------------

# General field type defined over a 3D grid (Nx, Ny, Nz), with arbitrary per-voxel data shape.
# This serves as the base type for all derived fields (scalar, vector, tensor, etc).
#
# Examples:
# - Scalar field: shape (Nx, Ny, Nz)
# - Vector field: shape (Nx, Ny, Nz, 3)
# - Tensor field: shape (Nx, Ny, Nz, 3, 3)
# - Custom feature vector per voxel: shape (Nx, Ny, Nz, D)
GeneralField = np.ndarray


def as_real_lattice_field(
    input_data,
    name: str = "field values",
    *,
    extra_ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
    is_finite: bool = True,
    value_range=None,
    bounded: bool = False,
) -> GeneralField:
    """Convert input into a real-valued NumPy lattice field.

    A lattice field must contain at least three lattice axes, contain numeric
    real-valued data, and may optionally be constrained to an exact trailing
    component rank through ``extra_ndim``. Integer-like data is accepted and
    converted to floating point. Optionally require finite values, enforce or
    clip a global numeric interval, or require one exact array shape.
    """
    values = np.asarray(input_data)
    if values.ndim < 3:
        raise ValueError(
            f"{name!r} must have at least 3 lattice axes. "
            f"Got shape {values.shape} instead."
        )
    if extra_ndim is not None:
        extra_ndim = int(
            as_number(extra_ndim, name=f"{name} extra_ndim", is_integer=True)
        )
        if extra_ndim < 0:
            raise ValueError(
                f"{name!r} extra_ndim must be non-negative. Got {extra_ndim}."
            )
        expected_ndim = 3 + extra_ndim
        if values.ndim != expected_ndim:
            raise ValueError(
                f"{name!r} must have shape (Nx, Ny, Nz)"
                f"{', ...' if extra_ndim > 0 else ''} with exactly {extra_ndim} "
                f"extra dimension{'s' if extra_ndim != 1 else ''}. "
                f"Got shape {values.shape} instead."
            )
    if shape is not None:
        shape = tuple(int(v) for v in shape)
        if values.shape != shape:
            raise ValueError(
                f"{name!r} must have shape {shape}. Got shape {values.shape} instead."
            )
    if any(int(dim) <= 0 for dim in values.shape):
        raise ValueError(
            f"{name!r} must not contain empty axes. Got shape {values.shape} instead."
        )
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"{name!r} must contain numeric values. Got {values.dtype}.")
    if np.iscomplexobj(values):
        raise TypeError(
            f"{name!r} must be real-valued; complex fields are unsupported."
        )
    values = values.astype(float, copy=False)

    if is_finite and not np.all(np.isfinite(values)):
        raise ValueError(f"{name!r} must be finite everywhere.")

    if value_range is not None:
        lo, hi = as_value_range(value_range, name=f"{name} value_range")
        below = values < lo
        above = values > hi
        if np.any(below) or np.any(above):
            if not bounded:
                value_min = float(np.nanmin(values))
                value_max = float(np.nanmax(values))
                raise ValueError(
                    f"{name!r} must stay within [{lo}, {hi}]. "
                    f"Got value range [{value_min}, {value_max}]."
                )
            values = np.clip(values, lo, hi)

    return values


# -------------------------
# Specialized field types (all are subtypes of GeneralField)
# -------------------------

# Validity mask field, shape: (Nx, Ny, Nz), dtype bool
# Subtype of GeneralField
# True marks voxels where the field data is physically meaningful;
# False marks voxels whose values must not enter any derived analysis.
MaskField = np.ndarray


def as_lattice_mask(
    input_data,
    name: str = "lattice mask",
    *,
    shape: tuple[int, ...] | None = None,
) -> MaskField:
    """Convert input into a boolean 3D lattice validity mask.

    Accepts boolean arrays or numeric arrays containing only 0/1 values,
    with exactly three lattice axes. Optionally require one exact shape.
    """
    values = np.asarray(input_data)
    if values.dtype == bool:
        values = values.astype(np.uint8)
    values = as_real_lattice_field(
        values,
        name=name,
        extra_ndim=0,
        shape=shape,
        is_finite=True,
        value_range=(0.0, 1.0),
    )
    if not np.all((values == 0.0) | (values == 1.0)):
        raise ValueError(
            f"{name!r} must contain only boolean-like values (True/False or 0/1)."
        )
    return values.astype(bool)


def as_points(coords, name="input data", dim=3, *, is_unique=False, min_num=None):
    try:
        coords = np.asarray(coords, dtype=float)
        if coords.ndim == 1:
            coords = np.asarray([coords], dtype=float)
        if coords.ndim != 2:
            raise ValueError(
                f"{name!r} must be a 2D array of shape (N, D). Got shape={coords.shape}."
            )
        if dim is not None and coords.shape[1] != dim:
            raise ValueError(
                f"{name!r} must be an (N, {dim}) array. Got shape={coords.shape}."
            )
        if is_unique:
            coords = np.unique(coords, axis=0)
        if min_num is not None and len(coords) < min_num:
            raise ValueError(
                f"{name!r} must contain at least {min_num} point(s). "
                f"Got {len(coords)}."
            )
        return coords.copy()
    except (ValueError, TypeError) as e:
        raise TypeError(f"Invalid `coords` input: {e}")


class _UnsetType:
    """
    Internal sentinel type representing an explicit "unset" state.

    This type is used to distinguish between:
    - a value that has not been provided by the user (UNSET), and
    - a value that is explicitly provided as None or another valid value.

    It is intentionally designed to be:
    - state-less (no attributes),
    - identity-based (checked via `is UNSET`),
    - type-identifiable (usable in type annotations),
    - and safe against accidental mutation.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


UNSET = _UnsetType()
Unset = _UnsetType
