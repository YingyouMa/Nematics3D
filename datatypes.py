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

# __all__ = [
#     "NumericInput",
#     "Vect3D",
#     "DimensionInfo",
#     "DimensionInfoInput",
#     "DimensionPeriodic",
#     "DimensionPeriodicInput",
#     "as_dimension_info",
#     "DimensionFlag",
#     "DimensionFlagInput",
#     "boundary_periodic_size_to_flag",
#     "GeneralField",
#     "nField",
#     "SField",
#     "QField5",
#     "QField9",
#     "QField",
#     "as_QField5",
#     "as_QField9",
# ]

# Number includes int, float, np.interger, np.floating and so on.
# Notably, Number includes np.inf
Number = numbers.Real

# - scalar → broadcasted to all 3 dimensions
# - list/tuple/array of 3 values → used directly
NumericInput = Union[Number, Sequence[Number]]

def as_Number(input_data, name='input data'):
    
    if not isinstance(input_data, numbers.Real):
        raise TypeError(f"{name} must be a number. Got {input_data} instead.")
        
    return input_data

# Vect(d) is simply vector in d-dimensions
def Vect(d):
    return Union[Sequence[Union[int, float]], np.ndarray]

def as_Vect(input_data, dim=3, name='input data', is_norm=False):
    """
    Convert the given input into a dim-D NumPy vector, with optional normalization.

    This function ensures that the input is interpreted as a NumPy array with shape (dim,).
    If `is_norm` is True, the vector will be normalized according to the sum of the squares
    of its components. The normalization is done in-place after the shape check.

    Parameters
    ----------
    input_data : array-like
        The input vector to be converted. Can be a list, tuple, or NumPy array.
        Must contain exactly dim elements.

    is_norm : bool, optional
        If True, normalize the vector so that its magnitude (based on sum of squares)
        becomes 1. Defaults to False.

    Returns
    -------
    numpy.ndarray
        A NumPy array of shape (dim,) representing the (optionally normalized) vector.

    Raises
    ------
    ValueError
        If the input cannot be reshaped into a vector of exactly 3 elements.

    Examples
    --------
    >>> as_Vect3D([1, 2, 3])
    array([1, 2, 3])

    >>> as_Vect3D((3, 4, 0), is_norm=True)
    array([0.6, 0.8, 0.0])
    """
    
    if (
       not isinstance(input_data, (tuple, list, np.ndarray))
       or len(input_data) != dim     
       or not any(isinstance(x, numbers.Real) for x in input_data)
            ):
        raise ValueError(
            f"{name} must be a vector with {dim} numbers. Got {input_data} instead."
        )
    else:
        input_data = np.asarray(input_data)

    if is_norm:
        input_data = input_data / np.linalg.norm(input_data)

    return input_data

# Tensor(shape) is simply matrix with given shape
def Tensor(shape):
    return Union[Sequence[Union[int, float]], np.ndarray]


def as_Tensor(input_data, shape, name='input data'):
  
    if (
       not isinstance(input_data, (tuple, list, np.ndarray))
       or np.shape((input_data)) != shape     
       ):
        raise ValueError(
            f"{name} must be a matrix with shape {shape}. Got {input_data} instead."
        )
    else:
        input_data = np.asarray(input_data)

    return input_data



# ColorRGB represents a color in RGB expression. It must be a tuple
ColorRGB = Tuple[float, float, float]


def as_ColorRGB(input_data, is_norm=False, norm_order=2):
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
    
    if (
       not isinstance(input_data, (tuple, list, np.ndarray))
       or len(input_data) != 3     
       or not any(isinstance(x, numbers.Real) for x in input_data)
            ):
        raise ValueError(
            f"For ColorRGB, input_data must be a vector with 3 numbers. Got {input_data} instead."
        )
        
    input_data = np.asarray(input_data)
        
    if np.max(input_data) > 1 or np.min(input_data) < 0:
        raise ValueError(
            f"For ColorRGB, each number should be in [0,1]. Got {input_data} instead."
        )

    if is_norm:
        if np.sum(input_data) < 1e-3:
            return (0, 0, 0)
        input_data = input_data / np.sum(input_data**norm_order)

    return tuple(input_data)


# -------------------------
# Dimension info types
# -------------------------

# DimensionInfo represents general per-dimension numeric metadata.
# It is a NumPy array of shape (3,), where each element corresponds to a spatial dimension.
# Example: number of grid points per dimension.
DimensionInfo = np.ndarray

# Input type for DimensionInfo:
# - scalar → broadcasted to all 3 dimensions
# - list/tuple/array of 3 values → used directly
DimensionInfoInput = NumericInput


def as_dimension_info(input_data: DimensionInfoInput) -> DimensionInfo:
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
    """

    if isinstance(input_data, (int, float)):
        return np.array([input_data] * 3)

    if isinstance(input_data, (list, tuple, np.ndarray)) and len(input_data) == 3:
        return np.array(input_data)

    raise ValueError(
        "Input must be either a single number or a list, tuple, or NumPy array of exactly three elements."
    )


# -------------------------
# Dimension periodicity types
# -------------------------

# DimensionPeriodic is a **specific form of DimensionInfo** that encodes boundary condition per dimension.
# - np.inf → non-periodic
# - int → periodic, with value as the boundary size
# Like DimensionInfo, it is a NumPy array of shape (3,).
DimensionPeriodic = DimensionInfo

# Input type for DimensionPeriodic
# - scalar → broadcasted to all 3 dimensions
# - list/tuple/array of 3 values → used directly
DimensionPeriodicInput = NumericInput

# -------------------------
# Dimension flag types
# -------------------------


# DimensionFlag is a **specific form of DimensionInfo** each element is boolean.
# Like DimensionInfo, it is a NumPy array of shape (3,).
DimensionFlag = DimensionInfo  # conceptually a specialized DimensionInfo

# Input type for DimensionFlag
# - bool → broadcasted to all 3 dimensions
# - list/tuple/array of 3 boolean values → used directly
DimensionFlagInput = NumericInput


def boundary_periodic_size_to_flag(arr: DimensionPeriodicInput) -> DimensionFlag:
    """
    Return a boolean mask indicating which spatial dimensions are periodic.

    This function converts a DimensionPeriodic array into a DimensionFlag,
    where each element is:
        - True  → the corresponding dimension is non-periodic (value is np.inf)
        - False → the dimension is periodic (value is an integer)

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


def as_QField9(qtensor: Union[QField5, QField9]) -> QField9:
    #! strict3d
    """
    Convert a Q-tensor field into full 3×3 matrix form (QField9).

    Accepts either:
    - a 5-component representation (QField5), shape (..., 5), or
    - a full matrix representation (QField9), shape (..., 3, 3)

    Parameters
    ----------
    qtensor : QField
        Input Q-tensor field, either in 5-component or 3×3 form.

    Returns
    -------
    QField9
        Converted Q-tensor field in full 3×3 matrix form.

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
            f"QField must be a float-type NumPy array, got dtype {qtensor.dtype}"
        )

    shape = qtensor.shape

    if len(shape) >= 2 and shape[-1] == 5:
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

    if len(shape) >= 3 and shape[-2:] == (3, 3):
        Q = qtensor
        return Q  # Already in QField9 form

    raise ValueError(
        "Invalid QField shape: expected (..., 5) or (..., 3, 3), "
        f"but got shape {shape}"
    )


def as_QField5(qtensor: Union[QField5, QField9]) -> QField5:
    """
    Convert a Q-tensor field into full 3×3 matrix form (QField9).

    Accepts either:
    - a 5-component representation (QField5), shape (..., 5), or
    - a full matrix representation (QField9), shape (..., 3, 3)

    Assumes the input is a symmetric, traceless 3×3 tensor field.

    Parameters
    ----------
    qtensor : QField
        Input Q-tensor field, either in 5-component or 3×3 form.

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
            f"QField must be a float-type NumPy array, got dtype {qtensor.dtype}"
        )

    shape = qtensor.shape

    if len(shape) >= 2 and shape[-2:] == (3, 3):

        Q5 = np.empty(shape[:-2] + (5,), dtype=qtensor.dtype)

        Q5[..., 0] = qtensor[..., 0, 0]  # Q_xx
        Q5[..., 1] = qtensor[..., 0, 1]  # Q_xy
        Q5[..., 2] = qtensor[..., 0, 2]  # Q_xz
        Q5[..., 3] = qtensor[..., 1, 1]  # Q_yy
        Q5[..., 4] = qtensor[..., 1, 2]  # Q_yz

        return Q5

    if len(shape) >= 3 and shape[-1] == 5:
        Q5 = qtensor
        return Q5

    raise ValueError(
        "Invalid QField shape: expected (..., 5) or (..., 3, 3), "
        f"but got shape {shape}"
    )


# -------------------------
# Disclination points type
# -------------------------

# DefectIndex represents the index-based location of a topological defect
# in a 3D discrete lattice of nematic directors. This is a **grid coordinate**, NOT a spatial position.
#
# The coordinate identifies the center of a 2×2 square loop of neighboring sites,
# where the winding number is computed.
#
# Format: (i, j+0.5, k+0.5), represented as (int, float, float)
# - The first entry (i) is an integer index along one lattice axis (e.g. x)
# - The second and third entries (j+0.5, k+0.5) are half-integer values, indicating that
#   the defect is located **between grid points** along those two directions (e.g. y and z)
#
# These half-integer values mean that the defect is not associated with a single lattice point,
# but rather with a 2×2 square loop. The defect coordinate is assumped to correspond to the **center** of that loop.
# The integer could be in any dimension.
#
# Example:
#   A defect at (3, 4.5, 7.5) lies in the yz-face centered between:
#     grid points (3, 4, 7), (3, 4, 8), (3, 5, 7), and (3, 5, 8)
#   This defines a 2×2 loop over which the director field forms a closed path.
DefectIndex = np.ndarray


def check_bool_flags(d: dict, prefix: str = "is_"):
    for name, value in d.items():
        if name.startswith(prefix) and not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool, got {type(value)}")
