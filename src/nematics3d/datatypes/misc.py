"""
Miscellaneous semantic data aliases and runtime conversion helpers.

This file defines semantic type aliases used throughout the package for clarity in
function signatures, documentation, and interface contracts.

All types defined here are **semantic aliases**: they describe the *intended meaning*
of data (e.g., a 3D vector or per-dimension metadata), but they do **not** enforce
structural constraints (e.g., shape or dtype) at the type level. That is:

    - We do NOT statically enforce shapes like (3,) or numeric-only elements.
    - Type checkers (e.g., mypy, Pyright) will treat these as np.ndarray or general unions.
    - Shape and value validation must be performed at runtime, if needed.

This file is intended to:
    - Serve as the centralized definition for commonly used input/output types.
    - Provide self-documenting names for inputs like DimensionInfo, Vector3, etc.
    - Allow future migration to stronger typing (e.g., with Pydantic, beartype) if desired.

Example usage:

    from datatypes import DimensionInfo

    def func(info: DimensionInfo):
        ...
"""

from typing import Union, Sequence, Literal, Tuple
import numpy as np
import numbers

from ..logging_decorator import logging_and_warning_decorator
from .dimension_info import DimensionInfo, as_dimension_info
from .number import Number, as_number, as_value_range
from .tensor import Tensor, as_tensor
from .vector import Vect, as_vector


def as_readonly_array(input_data, *, dtype=float, copy: bool = True) -> np.ndarray:
    """Return one NumPy array view/copy with write access disabled."""
    values = np.asarray(input_data, dtype=dtype)
    if copy:
        values = values.copy()
    values.setflags(write=False)
    return values


# Axes is a 3D orthonormal frame stored as columns.
Axes = np.ndarray
AxesInput = Union[Sequence[Sequence[Number]], np.ndarray]


def as_axes(
    input_data: AxesInput,
    name: str = "axes",
    *,
    atol: float = 1e-8,
    is_right_handed: bool = True,
) -> Axes:
    """Validate a 3D orthonormal axes frame stored as column vectors."""

    axes = as_tensor(input_data, (3, 3), name=name).copy()
    if not np.allclose(axes.T @ axes, np.eye(3), atol=atol):
        raise ValueError(f"{name!r} must be an orthonormal axes frame.")

    det = float(np.linalg.det(axes))
    if np.isclose(det, 0.0, atol=atol):
        raise ValueError(f"{name!r} must be a non-degenerate axes frame.")

    if is_right_handed and det < 0:
        axes[:, -1] = -axes[:, -1]

    return axes


# ColorRGB represents a color in RGB expression. It must be a tuple
ColorRGB = Tuple[float, float, float]


@logging_and_warning_decorator(start_finish_level=5)
def as_ColorRGB(
    input_data,
    name="input data",
    is_norm=False,
    norm_order=2,
    replace=None,
    logger=None,
):
    """
    Convert input into an RGB color tuple with optional normalization.

    This function ensures that the input is a 3-element vector representing
    RGB color values, each within the range [0, 1]. Optionally, the vector
    can be normalized according to a specified norm order.

    Parameters
    ----------
    input_data : array-like
        A sequence of 3 numeric values representing the Red, Green, and Blue
        components of the color. Each value should be in the range [0, 1].
        Accepts list, tuple, or NumPy array.

    is_norm : bool, optional
        Whether to normalize the RGB vector. If True, each component is divided
        by the sum of its components raised to the power of `norm_order`.
        Defaults to False.

    norm_order : int or float, optional
        The exponent to which each component is raised before summing in the
        normalization step. For example:
            - norm_order=2 : Euclidean-like normalization (sum of squares)
            - norm_order=1 : L1 normalization (sum of absolute values)
        Defaults to 2.

    replace : ColorRGB, optional
        Recover the input_data to this replace value if the input_data is illegal.
        Defaults to None, no recovery.


    Returns
    -------
    tuple of float
        The processed RGB color as a tuple of 3 floats, each in [0, 1]
        after validation and optional normalization.

    Raises
    ------
    ValueError
        If `input_data` does not have exactly 3 elements.
        If any component is outside the range [0, 1].

    Examples
    --------
    >>> as_ColorRGB([0.2, 0.5, 0.8])
    (0.2, 0.5, 0.8)

    >>> as_ColorRGB([0.2, 0.5, 0.8], is_norm=True, norm_order=2)
    (0.19245008972987526, 0.480, 0.7698001794597505)
    """

    try:
        if (
            not isinstance(input_data, (tuple, list, np.ndarray))
            or len(input_data) != 3
            or not all(isinstance(x, numbers.Real) for x in input_data)
        ):
            raise ValueError(
                f"{name} should be ColorRGB, which must be a tuple with 3 numbers. Got {input_data} instead."
            )

        input_data = np.asarray(input_data)

        if np.max(input_data) > 1 or np.min(input_data) < 0:
            raise ValueError(
                f"{name} should be ColorRGB, where each number should be in [0,1]. Got {input_data} instead."
            )
    except:
        if replace:
            logger.exception("Please check data type.")
            logger.recovery(f"Set color={replace} in the following.")
            input_data = replace
        else:
            raise

        if is_norm:
            if np.sum(input_data) < 1e-3:
                return (0, 0, 0)
            input_data = input_data / np.sum(input_data**norm_order)

    return tuple(map(float, input_data))


@logging_and_warning_decorator(start_finish_level=5)
def as_ColorRGB_array(
    input_data,
    name="input data",
    is_norm=False,
    norm_order=2,
    replace=None,
    logger=None,
):
    """
    Validate and process an (N, 3) array of RGB colors with optional row-wise normalization.

    Recovery is ONLY applied when:
      - input_data is array-like with shape (N, 3)
      - but contains illegal values (dtype or range)

    Any other structural error is not recoverable.
    """

    # ---------- Structural check (NOT recoverable) ----------
    if not isinstance(input_data, (list, tuple, np.ndarray)):
        raise ValueError(
            f"{name} must be an array-like with shape (N, 3). Got {type(input_data).__name__}."
        )

    input_data = np.asarray(input_data)

    if input_data.ndim != 2 or input_data.shape[1] != 3:
        raise ValueError(
            f"{name} must have shape (N, 3). Got shape {input_data.shape}."
        )

    N = input_data.shape[0]

    # ---------- Content check (recoverable) ----------
    try:
        if not np.issubdtype(input_data.dtype, np.number):
            raise ValueError(
                f"{name} must contain numeric values. Got dtype {input_data.dtype}."
            )

        if np.max(input_data) > 1 or np.min(input_data) < 0:
            raise ValueError(
                f"{name} is ColorRGB array, where each number should be in [0,1]."
            )

        input_data = input_data.astype(float)

    except Exception:
        if replace is None:
            raise
        else:
            logger.exception("Please check color array values.")
            logger.recovery(f"Set color array to {replace} in the following.")

            # --- Recovery: broadcast replace to (N, 3) ---
            replace_arr = np.asarray(replace, dtype=float)

            if replace_arr.ndim == 1:
                if replace_arr.shape != (3,):
                    raise ValueError(
                        f"replace must be shape (3,) or (N, 3). Got shape {replace_arr.shape}."
                    )
                # (3,) -> (N, 3)
                input_data = np.tile(replace_arr, (N, 1))

            elif replace_arr.ndim == 2:
                if replace_arr.shape != (N, 3):
                    raise ValueError(
                        f"replace must be shape (3,) or (N, 3). Got shape {replace_arr.shape}, "
                        f"expected (N, 3) with N={N}."
                    )
                input_data = replace_arr

            else:
                raise ValueError(
                    f"replace must be shape (3,) or (N, 3). Got array with ndim={replace_arr.ndim}."
                )

    # ---------- Row-wise normalization ----------
    if is_norm:
        norms = np.sum(input_data**norm_order, axis=1)

        safe = norms >= 1e-3
        input_data[safe] /= norms[safe][:, None]
        input_data[~safe] = 0.0

    return input_data


@logging_and_warning_decorator(start_finish_level=5)
def as_str(input_data, name="input_data", pool=None, replace=None, logger=None):
    """
    Validate an input value as a string, with optional membership check and
    user-provided fallback replacement.

    Parameters
    ----------
    input_data : Any
        The value to be validated. It is expected to be of type ``str`` under
        normal usage.
    name : str, optional
        A human-readable name used in error messages.
    pool : iterable, optional
        A collection of allowed string values. The membership check is applied
        only when ``pool`` is truthy. It is the caller's responsibility to ensure
        that ``pool`` itself is a valid iterable of acceptable values.
    replace : Any, optional
        A fallback value used when validation fails. When provided, validation
        errors will be suppressed and the return value will be forcibly replaced
        by ``replace``.
        **Note:** ``replace`` is not validated and may be of any type, including
        non-string values. The caller must ensure its semantic correctness.

    Returns
    -------
    Any
        Returns ``input_data`` if validation succeeds. Otherwise returns
        ``replace`` when it is provided. No guarantee is made that the return
        value is of type ``str`` when the replacement path is taken.

    Raises
    ------
    TypeError
        If ``input_data`` is not a string and ``replace`` is not provided.
    ValueError
        If ``input_data`` is not contained in ``pool`` and ``replace`` is not
        provided.
    """

    try:
        if not isinstance(input_data, str):
            raise TypeError(
                f"{name!r} should be str. Got {input_data} with type {type(input_data).__name__} instead"
            )
        elif pool and input_data not in pool:
            raise ValueError(f"{name!r} must be in {pool}. Got {input_data} instead.")
    except (TypeError, ValueError):
        if replace is None:
            raise
        else:
            logger.exception("Please check data type")
            logger.recovery(f"Changed it into {replace!r} in the following.")
            input_data = replace

    return input_data


@logging_and_warning_decorator(start_finish_level=5)
def as_list(input_data, name="input_data", replace=None, logger=None):
    """
    Normalize input into a list.

    If ``input_data`` is already a list, it is returned unchanged. If it is a
    tuple or set, it is converted to a list of its elements. Otherwise the value
    is treated as a single item and wrapped into a one-element list.
    """

    try:
        if isinstance(input_data, list):
            return input_data
        if isinstance(input_data, (tuple, set)):
            return list(input_data)
        return [input_data]
    except Exception:
        if replace is None:
            raise

        logger.exception(f"Failed to normalize {name!r} into a list.")
        logger.recovery(f"Change {name!r} into {replace!r} in the following.")
        return replace


# -------------------------
# Dimension periodicity types
# -------------------------

# DimensionPeriodic is a **specific form of DimensionInfo** that encodes boundary condition per dimension.
# - np.inf -> non-periodic
# - int -> periodic, with value as the boundary size
# Like DimensionInfo, it is a NumPy array of shape (3,).
DimensionPeriodic = DimensionInfo

# Input type for DimensionPeriodic
# - scalar -> broadcasted to all 3 dimensions
# - list/tuple/array of 3 values -> used directly
DimensionPeriodicInput = DimensionInfo


def boundary_periodic_size_to_flag(arr: DimensionPeriodicInput) -> np.ndarray:
    """
    Return a boolean mask indicating which spatial dimensions are periodic.

    Each output element is ``True`` when the corresponding box size is finite
    and therefore periodic, and ``False`` when it is infinite and non-periodic.

    Examples
    --------
    >>> boundary_periodic_flag(np.array([np.inf, 10, np.inf]))
    array([ False, True,  False])
    """

    arr = as_dimension_info(arr, name="periodic boundary size")

    return arr != np.inf


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


@logging_and_warning_decorator(start_finish_level=5)
def as_bool(input_data, name="input data", replace=None, logger=None) -> bool:

    try:
        # --- Case 1: already boolean ---
        if isinstance(input_data, (bool, np.bool_)):
            return input_data

        # --- Case 2: numeric 0 / 1 ---
        if isinstance(input_data, Number):
            if input_data in (0, 1):
                return bool(input_data)
            else:
                raise TypeError(
                    f"{name} must contain only 0/1 when numeric. Got {input_data}."
                )

        # --- Everything else is invalid ---
        raise TypeError(
            f"{name} must be boolean or in (0,1). "
            f"Got {input_data} with dtype={getattr(input_data, 'dtype', type(input_data).__name__)}"
        )

    except TypeError:
        # --- No recovery allowed ---
        if replace is None:
            raise

        logger.exception("Please check data type")
        logger.recovery(f"set {name} to be {replace} in the following.")

        return replace


def check_bool_flags(d: dict, prefix: str = "is_"):
    for name, value in d.items():
        if name.startswith(prefix):
            if not isinstance(value, (bool, np.bool_, Number)) or (
                isinstance(value, Number) and value not in (0, 1)
            ):
                raise TypeError(f"{name} must be a bool, got {type(value)}")


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
        # Provide a concise and readable representation for debugging,
        # logging, and error messages.
        return "UNSET"


# The single, canonical instance used throughout the codebase to denote
# an "unset" value. Identity comparison (`is UNSET`) should always be used.
UNSET = _UnsetType()

# Public alias for the sentinel's type, intended for use in type annotations,
# e.g. `float | Unset`. Users should not instantiate this type directly.
Unset = _UnsetType
