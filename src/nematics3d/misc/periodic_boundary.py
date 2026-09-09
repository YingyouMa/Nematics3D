"""Temporary home for field periodic-boundary extension helpers."""

import numpy as np

from ..datatypes import DimensionInfo, GeneralField, as_dimension_info


def add_periodic_boundary(
    data: GeneralField, is_boundary_periodic: DimensionInfo = 0
) -> GeneralField:
    """Append one periodic image slice along selected spatial axes."""
    data = np.asarray(data)
    if data.ndim < 3:
        raise ValueError(
            "`data` must have at least three spatial dimensions; "
            f"got shape {data.shape}."
        )

    is_boundary_periodic = as_dimension_info(
        is_boundary_periodic,
        name="is_boundary_periodic",
        is_bool=True,
    )

    output = data.copy()
    for axis, is_periodic in enumerate(is_boundary_periodic):
        if is_periodic:
            output = np.concatenate(
                (output, np.take(output, [0], axis=axis)), axis=axis
            )
    return output


__all__ = ["add_periodic_boundary"]
