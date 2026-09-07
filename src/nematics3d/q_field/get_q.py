"""Construction helpers for symmetric traceless Q-tensor fields."""

import numpy as np

from ..datatypes import (
    QField5,
    QField9,
    SField,
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


__all__ = ["get_q"]
