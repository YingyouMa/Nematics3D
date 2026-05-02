"""
datatypes.py

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

    from datatypes import DimensionInfo, DimensionInfoInput

    def func(info: DimensionInfoInput) -> DimensionPeriodic:
        ...
"""

from typing import Union, Sequence, Literal, Tuple
import numpy as np
import numbers

from nematics3d.logging_decorator import logging_and_warning_decorator

# Number includes int, float, np.interger, np.floating and so on.
# Notably, Number includes np.inf
Number = numbers.Real

# - scalar -> broadcasted to all 3 dimensions
# - list/tuple/array of 3 values -> used directly
NumericInput = Union[Number, Sequence[Number]]


@logging_and_warning_decorator(start_finish_level=5)
def as_Number(
    input_data,
    name="input data",
    is_int=False,
    value_range=None,
    bounded=False,
    replace=None,
    logger=None,
):

    try:
        # --- Type checks ---
        if is_int:
            if isinstance(input_data, numbers.Integral):
                input_data = int(input_data)
            elif (
                isinstance(input_data, numbers.Real) and float(input_data).is_integer()
            ):
                input_data = int(input_data)
            else:
                raise TypeError(
                    f"{name!r} must be an integer-valued number. Got {input_data} instead."
                )
        else:
            if not isinstance(input_data, numbers.Real):
                raise TypeError(f"{name!r} must be a number. Got {input_data} instead.")

        lo = hi = None

        # --- Validate value_range itself (recover by ignoring range if malformed) ---
        if value_range is not None:
            try:
                value_range_arr = np.asarray(value_range)

                if (
                    len(value_range_arr) != 2
                    or not isinstance(value_range_arr[0], numbers.Real)
                    or not isinstance(value_range_arr[1], numbers.Real)
                ):
                    raise TypeError(
                        f"value_range must be a tuple/list of two numbers, got {value_range!r}"
                    )

                lo, hi = value_range_arr
                if hi <= lo:
                    raise ValueError(
                        f"value_range must be strictly increasing, got {value_range!r}"
                    )

            except Exception as e:
                logger.exception(
                    f"Invalid value_range for {name!r}: {value_range!r}. Reason: {e}"
                )
                logger.recovery("Ignore value_range in the following.")
                value_range = None

        # --- Enforce range if applicable ---
        if value_range is not None:
            if not (lo <= input_data <= hi):
                msg = f"{name!r} must be in [{lo}, {hi}], got {input_data}."

                if not bounded:
                    raise ValueError(msg)

                # bounded=True: clip and warn
                if input_data < lo:
                    input_data = lo
                elif input_data > hi:
                    input_data = hi

                msg += f"\nSet {name!r} to be {input_data} in the following."
                logger.warning(msg)

        return input_data

    except (TypeError, ValueError) as e:
        # --- Recovery ---
        if replace is None:
            raise

        logger.exception(f"Validation failed for {name!r}: {e}")
        logger.recovery(f"Set {name!r} to be {replace!r} in the following.")

        return replace


# Vect(d) is simply vector in d-dimensions
def Vect(d):
    return Union[Sequence[Union[int, float]], np.ndarray]


@logging_and_warning_decorator(start_finish_level=5)
def as_Vect(
    input_data,
    dim=3,
    name="input data",
    is_norm=False,
    is_permit_zero=True,
    replace=None,
    logger=None,
):
    if is_norm:
        is_permit_zero = False

    try:
        if (
            not isinstance(input_data, (tuple, list, np.ndarray))
            or len(input_data) != dim
            or not all(isinstance(x, numbers.Real) for x in input_data)
        ):
            raise ValueError(
                f"{name!r} must be a vector with {dim} numbers. Got {input_data} instead."
            )
    except ValueError:
        if replace is None:
            raise

        logger.exception("Check input data.")
        logger.recovery(f"Change {name!r} into {replace} in the following.")
        input_data = replace
        if not isinstance(input_data, (tuple, list, np.ndarray)):
            return input_data

    input_data = np.asarray(input_data, dtype=float)
    norm = float(np.linalg.norm(input_data))

    if (not np.isfinite(norm)) or ((not is_permit_zero) and norm <= 1e-12):
        raise ValueError(
            f"{name!r} must be a {'non-zero ' if not is_permit_zero else ''}vector. Got {input_data} instead."
        )

    if is_norm:
        input_data = input_data / norm

    return input_data


# Tensor(shape) is simply matrix with given shape
def Tensor(shape):
    return Union[Sequence[Union[int, float]], np.ndarray]


def as_Tensor(input_data, shape, name="input data"):

    if (
        not isinstance(input_data, (tuple, list, np.ndarray))
        or np.shape((input_data)) != shape
    ):
        raise ValueError(
            f"{name} must be a matrix with shape {shape}. Got {input_data} instead."
        )
    else:
        input_data = np.asarray(input_data, dtype=float)

    return input_data


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

    axes = as_Tensor(input_data, (3, 3), name=name).copy()
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
# Dimension info types
# -------------------------

# DimensionInfo represents general per-dimension numeric metadata.
# It is a NumPy array of shape (3,), where each element corresponds to a spatial dimension.
# Example: number of grid points per dimension.
DimensionInfo = np.ndarray

# Input type for DimensionInfo:
# - scalar -> broadcasted to all 3 dimensions
# - list/tuple/array of 3 values -> used directly
DimensionInfoInput = NumericInput


def as_dimension_info(
    input_data: DimensionInfoInput, name: str = "input_data", is_bool: bool = False
) -> DimensionInfo:
    """
    Convert flexible user input into a standardized DimensionInfo array of shape (3,).

    Parameters
    ----------
    input_data : DimensionInfoInput
        Can be:
        - a scalar (int or float): will be broadcasted to all 3 dimensions;
        - a list, tuple, or ndarray of exactly 3 numeric values.

    Returns
    -------
    DimensionInfo
        A NumPy array of shape (3,) representing per-dimension numeric metadata.

    Raises
    ------
    ValueError
        If input is not a scalar or not a 3-element structure.
    ValueError
        If is_bool is True and non-boolian data is input.
    """

    if isinstance(input_data, (int, float)):
        result = np.array([input_data] * 3)
    elif isinstance(input_data, (list, tuple, np.ndarray)) and len(input_data) == 3:
        result = np.array(input_data)
    else:
        raise ValueError(
            f"{name} must be either a single number or a list, tuple, or NumPy array of exactly three elements."
        )

    if is_bool and not all(isinstance(x, (bool, np.bool_)) for x in result):
        raise ValueError(f"The elements in {name} must be bool. Got {result} instead.")

    return result


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
DimensionPeriodicInput = NumericInput

# -------------------------
# Dimension flag types
# -------------------------


# DimensionFlag is a **specific form of DimensionInfo** each element is boolean.
# Like DimensionInfo, it is a NumPy array of shape (3,).
DimensionFlag = DimensionInfo  # conceptually a specialized DimensionInfo

# Input type for DimensionFlag
# - bool -> broadcasted to all 3 dimensions
# - list/tuple/array of 3 boolean values -> used directly
DimensionFlagInput = NumericInput


def boundary_periodic_size_to_flag(arr: DimensionPeriodicInput) -> DimensionFlag:
    """
    Return a boolean mask indicating which spatial dimensions are periodic.

    This function converts a DimensionPeriodic array into a DimensionFlag,
    where each element is:
        - True  -> the corresponding dimension is non-periodic (value is np.inf)
        - False -> the dimension is periodic (value is an integer)

    Examples
    --------
    >>> boundary_periodic_flag(np.array([np.inf, 10, np.inf]))
    array([ False, True,  False])
    """

    arr = as_dimension_info(arr)
    if arr.shape != (3,):
        raise ValueError("Input must be a NumPy array of shape (3,)")

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

# -------------------------
# Specialized field types (all are subtypes of GeneralField)
# -------------------------

# Director field (unit vector), shape: (Nx, Ny, Nz, 3)
# Subtype of GeneralField
# It may relax to shape (..., 3)
nField = np.ndarray


def check_Sn(
    data, datatype: Literal["n", "S"], is_3d_strict: bool = True, is_norm=True
):

    data = np.asarray(data, dtype=np.float64)
    shape = np.shape(data)

    if datatype == "n":
        if shape[-1] != 3:
            raise ValueError(
                f"Director field must end with shape (..., 3), got {shape}"
            )
        if is_3d_strict and len(shape) != 4:
            raise ValueError(
                f"Strict 3D director field must have shape (Nx, Ny, Nz, 3), got {shape}"
            )
        if is_norm:
            norms = np.linalg.norm(data, axis=-1, keepdims=True)
            data = data / norms
    elif datatype == "S":
        if is_3d_strict and len(shape) != 3:
            raise ValueError(
                f"Strict 3D scalar field must have shape (Nx, Ny, Nz), got {shape}"
            )

    else:
        raise TypeError(f"Unsupported datatype '{datatype}': expected 'S' or 'n'")

    return data


# Scalar order parameter field, shape: (Nx, Ny, Nz)
# Subtype of GeneralField
# In the perfect ordered state, S is defined to be 1.
SField = np.ndarray

# Tensor order parameter in 5-component representation, shape: (Nx, Ny, Nz, 5)
# Subtype of GeneralField
# Components: [Q_xx, Q_xy, Q_xz, Q_yy, Q_yz]
QField5 = np.ndarray

# Tensor order parameter in full 3x3 matrix form, shape: (Nx, Ny, Nz, 3, 3)
# Subtype of GeneralField
# Symmetric traceless tensor Q_ij with:
# Q[..., 0,0] = Q_xx, Q[..., 0,1] = Q_xy, Q[..., 1,0] = Q_xy, etc.
QField9 = np.ndarray


def _validate_qfield_single_shape(
    shape: tuple[int, ...],
    *,
    name: str,
    expected_ndim: int,
    expected_label: str,
) -> None:
    if len(shape) != expected_ndim:
        raise ValueError(
            f"{name!r} must be a single 3D Q field in one of the supported "
            "representations: (Nx, Ny, Nz, 5) or (Nx, Ny, Nz, 3, 3). "
            f"This input matches the {expected_label} representation by its "
            f"trailing dimensions, but has shape {shape}."
        )


def as_qfield9(
    qtensor: Union[QField5, QField9],
    name="QField",
    is_single_field: bool = True,
) -> QField9:
    #! strict3d
    """
    Convert a Q-tensor field into full 3Ãƒâ€”3 matrix form (QField9).

    Accepts either:
    - a 5-component representation (QField5), shape (Nx, Ny, Nz, 5), or
    - a full matrix representation (QField9), shape (Nx, Ny, Nz, 3, 3)

    Set ``is_single_field=False`` to allow the more general legacy shapes
    ``(..., 5)`` and ``(..., 3, 3)`` for point sets, slices, or batched tensors.

    Parameters
    ----------
    qtensor : QField
        Input Q-tensor field, either in 5-component or 3Ãƒâ€”3 form.

    Returns
    -------
    QField9
        Converted Q-tensor field in full 3Ãƒâ€”3 matrix form.

    Raises
    ------
    TypeError
        If the input is not numerical
    ValueError
        If the input is not a valid QField5 or QField9 structure.
    """
    qtensor = np.asarray(qtensor)

    if not np.issubdtype(qtensor.dtype, np.floating):
        raise TypeError(
            f"QField must be a float-type NumPy array, got dtype {qtensor.dtype}. Name of QField: {name}"
        )

    shape = qtensor.shape

    if len(shape) >= 1 and shape[-1] == 5:
        if is_single_field:
            _validate_qfield_single_shape(
                shape,
                name=name,
                expected_ndim=4,
                expected_label="(Nx, Ny, Nz, 5)",
            )
        # Convert from 5-component representation to full 3x3 tensor
        Q = np.zeros((*shape[:-1], 3, 3), dtype=qtensor.dtype)
        Q[..., 0, 0] = qtensor[..., 0]  # Q_xx
        Q[..., 0, 1] = qtensor[..., 1]  # Q_xy
        Q[..., 0, 2] = qtensor[..., 2]  # Q_xz
        Q[..., 1, 0] = qtensor[..., 1]  # Q_yx = Q_xy
        Q[..., 1, 1] = qtensor[..., 3]  # Q_yy
        Q[..., 1, 2] = qtensor[..., 4]  # Q_yz
        Q[..., 2, 0] = qtensor[..., 2]  # Q_zx = Q_xz
        Q[..., 2, 1] = qtensor[..., 4]  # Q_zy = Q_yz
        Q[..., 2, 2] = -Q[..., 0, 0] - Q[..., 1, 1]  # Q_zz from traceless condition
        return Q

    if len(shape) >= 2 and shape[-2:] == (3, 3):
        if is_single_field:
            _validate_qfield_single_shape(
                shape,
                name=name,
                expected_ndim=5,
                expected_label="(Nx, Ny, Nz, 3, 3)",
            )
        Q = qtensor
        return Q  # Already in QField9 form

    raise ValueError(
        "Invalid QField shape: expected (Nx, Ny, Nz, 5) or "
        f"(Nx, Ny, Nz, 3, 3), but got shape {shape}. "
        f"Name of QField: {name}"
    )


def as_qfield5(
    qtensor: Union[QField5, QField9],
    name="QField",
    is_single_field: bool = True,
) -> QField5:
    """
    Convert a Q-tensor field into full 3Ãƒâ€”3 matrix form (QField9).

    Accepts either:
    - a 5-component representation (QField5), shape (Nx, Ny, Nz, 5), or
    - a full matrix representation (QField9), shape (Nx, Ny, Nz, 3, 3)

    Set ``is_single_field=False`` to allow the more general legacy shapes
    ``(..., 5)`` and ``(..., 3, 3)`` for point sets, slices, or batched tensors.

    Assumes the input is a symmetric, traceless 3Ãƒâ€”3 tensor field.

    Parameters
    ----------
    qtensor : QField
        Input Q-tensor field, either in 5-component or 3Ãƒâ€”3 form.

    Returns
    -------
    QField5
        5-component vector form of Q-tensor with shape (..., 5)

    Raises
    ------
    TypeError
        If the input is not a float-type NumPy array.
    ValueError
        If the input shape is not (..., 3, 3)
    """
    qtensor = np.asarray(qtensor)

    if not np.issubdtype(qtensor.dtype, np.floating):
        raise TypeError(
            f"QField must be a float-type NumPy array, got dtype {qtensor.dtype}. Name of QField: {name}"
        )

    shape = qtensor.shape

    if len(shape) >= 2 and shape[-2:] == (3, 3):
        if is_single_field:
            _validate_qfield_single_shape(
                shape,
                name=name,
                expected_ndim=5,
                expected_label="(Nx, Ny, Nz, 3, 3)",
            )

        Q5 = np.empty(shape[:-2] + (5,), dtype=qtensor.dtype)

        Q5[..., 0] = qtensor[..., 0, 0]  # Q_xx
        Q5[..., 1] = qtensor[..., 0, 1]  # Q_xy
        Q5[..., 2] = qtensor[..., 0, 2]  # Q_xz
        Q5[..., 3] = qtensor[..., 1, 1]  # Q_yy
        Q5[..., 4] = qtensor[..., 1, 2]  # Q_yz

        return Q5

    if len(shape) >= 1 and shape[-1] == 5:
        if is_single_field:
            _validate_qfield_single_shape(
                shape,
                name=name,
                expected_ndim=4,
                expected_label="(Nx, Ny, Nz, 5)",
            )
        Q5 = qtensor
        return Q5

    raise ValueError(
        "Invalid QField shape: expected (Nx, Ny, Nz, 5) or "
        f"(Nx, Ny, Nz, 3, 3), but got shape {shape}. "
        f"Name of QField: {name}"
    )


# -------------------------
# Disclination points type
# -------------------------

# DefectIndex represents the index-based location of a topological defect
# in a 3D discrete lattice of nematic directors. This is a **grid coordinate**, NOT a spatial position.
#
# The coordinate identifies the center of a 2Ãƒâ€”2 square loop of neighboring sites,
# where the winding number is computed.
#
# Format: (i, j+0.5, k+0.5), represented as (int, float, float)
# - The first entry (i) is an integer index along one lattice axis (e.g. x)
# - The second and third entries (j+0.5, k+0.5) are half-integer values, indicating that
#   the defect is located **between grid points** along those two directions (e.g. y and z)
#
# These half-integer values mean that the defect is not associated with a single lattice point,
# but rather with a 2Ãƒâ€”2 square loop. The defect coordinate is assumped to correspond to the **center** of that loop.
# The integer could be in any dimension.
#
# Example:
#   A defect at (3, 4.5, 7.5) lies in the yz-face centered between:
#     grid points (3, 4, 7), (3, 4, 8), (3, 5, 7), and (3, 5, 8)
#   This defines a 2Ãƒâ€”2 loop over which the director field forms a closed path.
DefectIndex = np.ndarray


def as_DefectIndex(arr: np.ndarray, tol=1e-8, is_return_row=False) -> DefectIndex:

    arr = np.asarray(arr)

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"Input must be (N,3) array for defect_indices, got shape {arr.shape}"
        )

    is_int = np.abs(arr - np.round(arr)) < tol
    is_half = (np.abs(arr * 2 - np.round(arr * 2)) < tol) & (~is_int)
    valid_rows = (is_int.sum(axis=1) == 1) & (is_half.sum(axis=1) == 2)

    valid_rows = (is_int.sum(axis=1) == 1) & (is_half.sum(axis=1) == 2)

    if not np.all(valid_rows):
        msg = "DefectIndex is not valid. For each defect there must be one integer and two half-intergers.\n"
        if is_return_row:
            bad_idx = np.where(~valid_rows)[0]
            msg += f"Invalid DefectIndex rows detected at indices {bad_idx.tolist()} "
        raise ValueError(msg)

    return arr


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
