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
    check_Sn,
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

    n = check_Sn(n, "n", is_3d_strict=False)

    Q = np.einsum("...i, ...j -> ...ij", n, n) - np.eye(3) / 3
    if S is not None:
        S = check_Sn(S, "S", is_3d_strict=False)
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
    n_reference = check_Sn(n_reference, "n", is_3d_strict=False)
    n_target = check_Sn(n_target, "n", is_3d_strict=False)
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

    The mapping combines a Boy-surface polynomial immersion of RP^2 with a
    vividness-optimized affine transform. The selected map maximizes mean
    OKLab chroma subject to ``J_loc <= 0.55``, calibrated red/green/blue axis
    tolerances, and the sRGB gamut constraint.

    Parameters
    ----------
    n : array_like, shape (..., 3)
        Nematic director field.

    Returns
    -------
    colors : list of tuple
        RGB colors in [0, 1], suitable for plotting.
    """
    n = check_Sn(n, "n", is_3d_strict=False, is_norm=True)

    boy = np.empty(n.shape, dtype=float)
    x = n[..., 0]
    y = n[..., 1]
    z = n[..., 2]
    x2 = x**2
    y2 = y**2
    z2 = z**2

    boy[..., 0] = 0.5 * (
        (2.0 * x2 - y2 - z2)
        + 2.0 * y * z * (y2 - z2)
        + z * x * (x2 - z2)
        + x * y * (y2 - x2)
    )
    boy[..., 1] = (7.0 / 8.0) * (
        (y2 - z2) + z * x * (z2 - x2) + x * y * (y2 - x2)
    )
    boy[..., 2] = (
        (1.0 / 8.0)
        * (x + y + z)
        * ((x + y + z) ** 3 + 4.0 * (y - x) * (z - y) * (x - z))
    )

    transform = np.array(
        [
            [0.5022508927293965, 0.0814191819777772, 0.4278817282953531],
            [-0.2622468169155294, 0.4198664552518698, 0.2843783905850694],
            [-0.2603273418569074, -0.3829942092529955, 0.3705024138608909],
        ],
        dtype=float,
    )
    offset = np.array(
        [0.3810134662659256, 0.4051244318995519, 0.4114207201942865],
        dtype=float,
    )

    result = np.einsum("...i,ji->...j", boy, transform) + offset
    return [tuple(color) for color in np.clip(result, 0.0, 1.0)]
