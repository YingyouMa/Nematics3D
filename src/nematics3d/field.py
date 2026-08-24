"""
Field-level utilities for structured Q-tensor and director data.
"""

from typing import List, Tuple

import numpy as np

# from .general import *
from .datatypes import (
    DimensionFlagInput,
    GeneralField,
    QField9,
    SField,
    as_dimension_info,
    as_director_field,
    as_scalar_field,
    nField,
)


def get_q(
    n: nField,
    S: SField | None = None,  # noqa: N803 - conventional scalar-order symbol
    *,
    m: nField | None = None,
    P: SField | None = None,  # noqa: N803 - conventional biaxial-order symbol
) -> QField9:
    """Construct a symmetric traceless Q-tensor field.

    The uniaxial convention is ``Q = S * (n n - I / 3)``. When ``m`` and
    ``P`` are supplied, the biaxial contribution is ``P * (m m - l l)``,
    where ``l = cross(n, m)``. Directors are normalized during validation.

    Parameters
    ----------
    n : nField
        Primary director data with trailing shape ``(..., 3)``.
    S : SField or None, optional
        Uniaxial scalar-order data. ``None`` is equivalent to ``1``.
    m : nField or None, optional
        Secondary director data with trailing shape ``(..., 3)``. It must be
        supplied together with ``P``.
    P : SField or None, optional
        Signed biaxial-order data. It must be supplied together with ``m``.

    Returns
    -------
    QField9
        Symmetric traceless tensors with trailing shape ``(..., 3, 3)``.

    Raises
    ------
    ValueError
        If a director is zero, ``m`` and ``P`` are not supplied together,
        field shapes cannot broadcast, or biaxial directors are not
        orthogonal.
    """
    n = as_director_field(n, name="n", is_zero_allowed=False)
    scalar_order = as_scalar_field(1.0 if S is None else S, name="S")

    is_m_given = m is not None
    is_p_given = P is not None
    if is_m_given != is_p_given:
        raise ValueError("'m' and 'P' must be supplied together.")

    leading_shapes = [n.shape[:-1], scalar_order.shape]
    if is_m_given:
        m = as_director_field(m, name="m", is_zero_allowed=False)
        biaxial_order = as_scalar_field(P, name="P")
        leading_shapes.extend((m.shape[:-1], biaxial_order.shape))

    try:
        field_shape = np.broadcast_shapes(*leading_shapes)
    except ValueError as error:
        raise ValueError(
            "The leading shapes of n, S, m, and P must be broadcastable. "
            f"Got {leading_shapes}."
        ) from error

    n = np.broadcast_to(n, field_shape + (3,))
    scalar_order = np.broadcast_to(scalar_order, field_shape)
    identity = np.eye(3, dtype=float)
    q_tensor = scalar_order[..., None, None] * (
        np.einsum("...i,...j->...ij", n, n) - identity / 3.0
    )

    if is_m_given:
        m = np.broadcast_to(m, field_shape + (3,))
        biaxial_order = np.broadcast_to(biaxial_order, field_shape)
        dot_products = np.einsum("...i,...i->...", n, m)
        if not np.all(np.isclose(dot_products, 0.0, rtol=0.0, atol=1e-8)):
            max_abs_dot = float(np.max(np.abs(dot_products)))
            raise ValueError(
                "'n' and 'm' must be orthogonal at every field point. "
                f"Maximum absolute dot product: {max_abs_dot:.6g}."
            )

        third_director = np.cross(n, m)
        q_tensor += biaxial_order[..., None, None] * (
            np.einsum("...i,...j->...ij", m, m)
            - np.einsum("...i,...j->...ij", third_director, third_director)
        )

    return q_tensor


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
