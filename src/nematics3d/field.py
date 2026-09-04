"""
Field-level utilities for structured Q-tensor and director data.
"""

from typing import List, Tuple

import numpy as np

from .datatypes import (
    DimensionInfo,
    GeneralField,
    QField5,
    QField9,
    SField,
    as_dimension_info,
    as_director_field,
    as_scalar_field,
    as_str,
    nField,
)


def _get_q_from_validated(
    n: nField,
    scalar_order: SField,
    *,
    m: nField | None = None,
    biaxial_order: SField | None = None,
    output: str = "q9",
) -> QField5 | QField9:
    """Build Q5 or Q9 from already validated, normalized field inputs."""
    leading_shapes = [n.shape[:-1], scalar_order.shape]
    if m is not None:
        leading_shapes.extend((m.shape[:-1], biaxial_order.shape))
    field_shape = np.broadcast_shapes(*leading_shapes)

    calculation_dtype = np.result_type(
        n.dtype,
        scalar_order.dtype,
        *(value.dtype for value in (m, biaxial_order) if value is not None),
    )
    n = np.asarray(np.broadcast_to(n, field_shape + (3,)), dtype=calculation_dtype)
    scalar_order = np.asarray(
        np.broadcast_to(scalar_order, field_shape),
        dtype=calculation_dtype,
    )

    component_indices = (
        ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2))
        if output == "q5"
        else ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    )
    component_shape = (5,) if output == "q5" else (3, 3)
    q_tensor = np.empty(field_shape + component_shape, dtype=calculation_dtype)

    for index_component, (index_i, index_j) in enumerate(component_indices):
        target = (
            q_tensor[..., index_component]
            if output == "q5"
            else q_tensor[..., index_i, index_j]
        )
        np.multiply(n[..., index_i], n[..., index_j], out=target)
        if index_i == index_j:
            target -= calculation_dtype.type(1.0 / 3.0)
        np.multiply(target, scalar_order, out=target)

    if m is not None:
        m = np.asarray(
            np.broadcast_to(m, field_shape + (3,)),
            dtype=calculation_dtype,
        )
        biaxial_order = np.asarray(
            np.broadcast_to(biaxial_order, field_shape),
            dtype=calculation_dtype,
        )
        third_director = np.cross(n, m)
        contribution = np.empty(field_shape, dtype=calculation_dtype)
        third_product = np.empty(field_shape, dtype=calculation_dtype)

        for index_component, (index_i, index_j) in enumerate(component_indices):
            target = (
                q_tensor[..., index_component]
                if output == "q5"
                else q_tensor[..., index_i, index_j]
            )
            np.multiply(m[..., index_i], m[..., index_j], out=contribution)
            np.multiply(
                third_director[..., index_i],
                third_director[..., index_j],
                out=third_product,
            )
            contribution -= third_product
            np.multiply(contribution, biaxial_order, out=contribution)
            target += contribution

    if output == "q9":
        q_tensor[..., 1, 0] = q_tensor[..., 0, 1]
        q_tensor[..., 2, 0] = q_tensor[..., 0, 2]
        q_tensor[..., 2, 1] = q_tensor[..., 1, 2]

    return q_tensor


def get_q(
    n: nField,
    S: SField | None = None,  # noqa: N803 - conventional scalar-order symbol
    *,
    m: nField | None = None,
    P: SField | None = None,  # noqa: N803 - conventional biaxial-order symbol
    output: str = "q9",
) -> QField5 | QField9:
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

    output : {"q5", "q9"}, optional
        Output representation. ``"q9"`` returns full symmetric 3x3 tensors;
        ``"q5"`` returns compact ``(Qxx, Qxy, Qxz, Qyy, Qyz)`` components.

    Returns
    -------
    QField5 or QField9
        Symmetric traceless tensors in the requested representation.

    Raises
    ------
    ValueError
        If a director is zero, ``m`` and ``P`` are not supplied together,
        field shapes cannot broadcast, or biaxial directors are not
        orthogonal.
    """
    output = as_str(output, name="Q output representation", pool=("q5", "q9"))
    n = as_director_field(n, name="n", is_zero_allowed=False)
    scalar_order = as_scalar_field(1.0 if S is None else S, name="S")

    is_m_given = m is not None
    is_p_given = P is not None
    if is_m_given != is_p_given:
        raise ValueError("'m' and 'P' must be supplied together.")

    if is_m_given:
        m = as_director_field(m, name="m", is_zero_allowed=False)
        biaxial_order = as_scalar_field(P, name="P")

    try:
        leading_shapes = [n.shape[:-1], scalar_order.shape]
        if is_m_given:
            leading_shapes.extend((m.shape[:-1], biaxial_order.shape))
        np.broadcast_shapes(*leading_shapes)
    except ValueError as error:
        raise ValueError(
            "The leading shapes of n, S, m, and P must be broadcastable. "
            f"Got {leading_shapes}."
        ) from error

    if is_m_given:
        dot_products = np.einsum("...i,...i->...", n, m)
        if not np.all(np.isclose(dot_products, 0.0, rtol=0.0, atol=1e-8)):
            max_abs_dot = float(np.max(np.abs(dot_products)))
            raise ValueError(
                "'n' and 'm' must be orthogonal at every field point. "
                f"Maximum absolute dot product: {max_abs_dot:.6g}."
            )

    return _get_q_from_validated(
        n,
        scalar_order,
        m=m,
        biaxial_order=biaxial_order if is_m_given else None,
        output=output,
    )


def add_periodic_boundary(
    data: GeneralField, is_boundary_periodic: DimensionInfo = 0
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

    is_boundary_periodic : DimensionInfo, optional
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
    is_boundary_periodic = as_dimension_info(
        is_boundary_periodic,
        name="is_boundary_periodic",
        is_bool=True,
    )

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

    The mapping combines a Boy-surface polynomial immersion of RP^2 with the
    selected OKLab-optimized affine transform at the Pareto knee
    ``J_loc ~= 0.43``. The optimization improves perceptual separation and
    local color uniformity while keeping the output inside the sRGB gamut.

    Parameters
    ----------
    n : array_like, shape (..., 3)
        Nematic director field.

    Returns
    -------
    colors : list of tuple
        RGB colors in [0, 1], suitable for plotting.
    """
    n = as_director_field(n, name="n", is_normalized=True)
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
    boy[..., 1] = (7.0 / 8.0) * ((y2 - z2) + z * x * (z2 - x2) + x * y * (y2 - x2))
    boy[..., 2] = (
        (1.0 / 8.0)
        * (x + y + z)
        * ((x + y + z) ** 3 + 4.0 * (y - x) * (z - y) * (x - z))
    )

    transform = np.array(
        [
            [0.5015275525743265, 0.0814208604228778, 0.4289041134674454],
            [-0.2426021621724047, 0.3331598986797062, 0.2349957727825471],
            [-0.2761140886939694, -0.3089204069097675, 0.4083459067954693],
        ],
        dtype=float,
    )
    offset = np.array(
        [0.3805938025338775, 0.4480306832558079, 0.4044714907877873],
        dtype=float,
    )
    result = np.einsum("...i,ji->...j", boy, transform) + offset
    return [tuple(color) for color in np.clip(result, 0.0, 1.0)]
