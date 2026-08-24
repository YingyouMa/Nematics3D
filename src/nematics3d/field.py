"""
Field-level utilities for structured Q-tensor and director data.
"""

from typing import List, Tuple

import numpy as np

# from .general import *
from .datatypes import (
    QField9,
    nField,
    SField,
    as_director_field,
    as_scalar_field,
    GeneralField,
    DimensionFlagInput,
    as_dimension_info,
)
from .logging_decorator import logging_and_warning_decorator


@logging_and_warning_decorator()
def getQ(n: nField, S: SField = None, logger=None) -> QField9:
    #! biaxial
    """
    Compute the Q-tensor field from a given director field and optional scalar order parameter.

    This function constructs a symmetric, traceless, uniaxial Q-tensor of the form:
        Q_ij = S * (n_i n_j - δ_ij / 3)

    If `S` is not provided, the tensor is computed assuming S = 1.

    Parameters
    ----------
    n : nField
        Director field of shape (..., 3).

    S : SField, optional
        Scalar order parameter field of shape (...,). If provided, scales the Q-tensor accordingly.

    Returns
    -------
    Q : QField9
        The computed Q-tensor field of shape (..., 3, 3), symmetric and traceless.
    """

    n = as_director_field(n, name="n")

    Q = np.einsum("...i, ...j -> ...ij", n, n) - np.eye(3) / 3
    if S is not None:
        S = as_scalar_field(S, name="S")
        Q = np.einsum("..., ...ij -> ...ij", S, Q)
    else:
        logger.warning(">>> No S input. Set to be 1.")

    return Q


def add_periodic_boundary(
    data: GeneralField, is_boundary_periodic: DimensionFlagInput = 0
) -> GeneralField:
    #! loop
    """
    Extend a physical field with periodic boundary slices in specified dimensions.

    This function appends one extra grid slice along each of the periodic dimensions.
    The added slice is a copy of the first slice along that axis, ensuring periodic continuity.
    If a dimension is non-periodic, it is left unchanged.

    Parameters
    ----------
    data : GeneralField
        Input physical field of shape (Nx, Ny, Nz, ...), where (Nx, Ny, Nz) are spatial dimensions,
        and the remaining axes represent vector/tensor components or other per-voxel data.

    is_boundary_periodic : DimensionFlagInput, optional
        A 3-element flag indicating which spatial dimensions are periodic.
        - Can be a scalar (broadcasted), or
        - A list/tuple/array of booleans with shape (3,)
        - Default is 0 (all dimensions non-periodic)

    Returns
    -------
    output : GeneralField
        Extended field with one additional slice along each periodic dimension.
        Shape becomes:
            (Nx + is_periodic[0], Ny + is_periodic[1], Nz + is_periodic[2], ...)
    """
    is_boundary_periodic = as_dimension_info(is_boundary_periodic)

    if np.any(is_boundary_periodic):
        Nx, Ny, Nz, *rest_shape = data.shape  # Extract the first three dimensions
        output = np.empty(
            (
                Nx + is_boundary_periodic[0],
                Ny + is_boundary_periodic[1],
                Nz + is_boundary_periodic[2],
                *rest_shape,
            ),
            dtype=data.dtype,
        )
        output[:Nx, :Ny, :Nz] = data  # Copy original data into the new array

        # Copy first slices to last.
        if is_boundary_periodic[0]:
            output[Nx] = output[0]
        if is_boundary_periodic[1]:
            output[:, Ny] = output[:, 0]
        if is_boundary_periodic[2]:
            output[:, :, Nz] = output[:, :, 0]
    else:
        output = data

    return output


def align_directors(n_reference: nField, n_target: nField) -> nField:
    """
    Align target director to have similar orientation as reference.
    This is used to handle the nematic symmetry of directors.
    """
    n_reference = as_director_field(n_reference, name="n_reference")
    n_target = as_director_field(n_target, name="n_target")
    dots = np.einsum("...i,...i->...", n_reference, n_target)
    signs = np.where(dots < 0, -1, 1)
    return np.einsum("...,...i->...i", signs, n_target)


def align_stack(stack):

    dots = np.einsum("...i,...i->...", stack[:-1], stack[1:])

    flips = np.ones(stack.shape[:-1], dtype=np.int8)
    flips[1:] = np.where(dots < 0, -1, 1).astype(np.int8, copy=False)

    acc_flips = np.cumprod(flips, axis=0)

    stack *= acc_flips[..., np.newaxis].astype(stack.dtype, copy=False)
    return stack


def n_color_immerse(n: nField) -> List[Tuple]:
    """
    Map a nematic director field to RGB colors for visualization.

    This function encodes the orientation of a unit director vector `n`
    into RGB color values using a nonlinear polynomial mapping followed by
    a fixed linear transformation and scaling. The colormap is an immersion
    from RP^2 to R^3. This ensures that similar orientatioin of n refer to
    similar colors but different orientations might refer to the same color.

    The colormap is modified from boy's surface.

    The color is specifically desined to be distince on white background,
    and to get x, y, z direction closed to red, blue and green colors.

    x: [0.90535893, 0.22874911, 0.22062688]
    y: [0.05416607, 0.27934554, 0.48937438]
    z: [0.30416607, 0.90434554, 0.22687438]

    Parameters
    ----------
    n : array_like, shape (..., 3)
        Nematic director field.
        Can be of arbitrary leading dimensions.

    Returns
    -------
    colors : list of tuples. Each tuple has 3 elements
        RGB color with values typically in [0, 1], suitable for plotting.

    Examples
    --------
    >>> n = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    >>> colors = n_color_immerse(n)
    >>> colors
    [(0.39345357, 0.1364875 , 0.60187625),
     (0.09285714, 0.47428571, 0.605     ),
     (0.51845357, 0.4489875 , 0.47062625)]
    """

    n = as_director_field(n, name="n", is_normalized=True)

    RGB = np.zeros((*(np.shape(n)[:-1]), 3))

    x = n[..., 0]
    y = n[..., 1]
    z = n[..., 2]

    x2 = x**2
    y2 = y**2
    z2 = z**2

    RGB[..., 0] = (
        (2 * x2 - y2 - z2)
        + 2 * y * z * (y2 - z2)
        + z * x * (x2 - z2)
        + x * y * (y2 - x2)
    )
    RGB[..., 1] = (y2 - z2) + z * x * (z2 - x2) + x * y * (y2 - x2)
    RGB[..., 2] = (x + y + z) * ((x + y + z) ** 3 + 4 * (y - x) * (z - y) * (x - z))

    RGB[..., 0] = RGB[..., 0] / 2
    RGB[..., 1] = RGB[..., 1] * 7 / 8
    RGB[..., 2] = RGB[..., 2] / 8

    M = np.array(
        [
            [1.01667, -0.3, -0.48333],
            [-1.01667, -1.5, -1.31667],
            [-0.18333, 0.3, 1.31667],
        ]
    )

    result = np.einsum("...i, ji -> ...j", RGB, M)

    scales = np.array([2.1, 4.2, 2.0])
    offsets = np.array([0.45, 0.51, 0.23])
    result = result / scales + offsets

    colors = []
    for color in result:
        colors.append(tuple(color))

    return colors
