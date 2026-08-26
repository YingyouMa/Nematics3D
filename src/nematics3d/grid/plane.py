"""Plane-axis resolution helpers."""

import numpy as np

from ..datatypes import as_vector
from ..geometry import rotation_matrix_from_vectors
from ..logging_decorator import logging_and_warning_decorator


@logging_and_warning_decorator(start_finish_level=5)
def resolve_plane_physical_axes(
    normal,
    axis1=None,
    *,
    is_warn=True,
    logger=None,
):
    """Return a valid orthonormal in-plane basis from ``normal`` and ``axis1``.

    ``normal`` is assumed to already be a unit normal. If ``axis1`` is missing,
    collinear with ``normal``, or not perfectly perpendicular, this helper
    repairs the basis and returns the final physical ``axis1`` and derived
    ``axis2 = cross(normal, axis1)``.
    """
    normal = as_vector(
        normal,
        name="normal used to resolve the plane physical axes",
        is_normalized=True,
    )
    axis1_use = (
        None
        if axis1 is None
        else as_vector(
            axis1,
            name="axis1 used to resolve the plane physical axes",
            is_normalized=True,
        )
    )
    logger.debug(
        f"Resolve plane physical axes from normalized normal={normal} "
        f"and normalized axis1={axis1_use}."
    )

    if axis1_use is not None:
        dot_product = normal @ axis1_use
        if np.isclose(abs(dot_product), 1.0, atol=1e-8):
            old_axis1 = axis1_use.copy()
            axis1_use = None
            if is_warn:
                logger.warning(
                    f"Invalid geometry: axis1 is collinear with normal "
                    f"(dot product: {dot_product:.4e}). Ignore original axis1 "
                    f"{old_axis1} and fall back to the automatic reference "
                    f"axis for normal {normal}."
                )
        elif not np.isclose(dot_product, 0, atol=1e-8):
            old_axis1 = axis1_use.copy()
            axis1_use = axis1_use - dot_product * normal
            axis1_use = as_vector(
                axis1_use,
                name="projected axis1 used to resolve the plane physical axes",
                is_normalized=True,
            )
            if is_warn:
                logger.warning(
                    f"Invalid geometry: axis1 is not perpendicular to normal "
                    f"(dot product: {dot_product:.4e}). Projecting original "
                    f"axis1 {old_axis1} onto the plane defined by normal "
                    f"{normal}. New orthonormal axis1: {axis1_use}."
                )

    if axis1_use is None:
        rotation_matrix = rotation_matrix_from_vectors((0, 0, 1), normal)
        axis1_use = as_vector(
            rotation_matrix @ np.array([1.0, 0.0, 0.0]),
            name="auto-generated axis1 used to resolve the plane physical axes",
            is_normalized=True,
        )
        logger.debug(
            f"axis1 not provided. Automatically generated a reference axis1 "
            f"{axis1_use} perpendicular to normal {normal}."
        )

    axis2 = as_vector(
        np.cross(normal, axis1_use),
        name="axis2 derived while resolving the plane physical axes",
        is_normalized=True,
    )
    logger.debug(
        f"Resolved plane physical axes: normal={normal}, axis1={axis1_use}, "
        f"axis2={axis2}."
    )
    return axis1_use, axis2
