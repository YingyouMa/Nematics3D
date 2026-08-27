"""Grid-shape runtime converter."""

from collections.abc import Sequence

import numpy as np

from .bool import as_bool


def as_grid_shape(
    input_data,
    name: str = "grid shape",
    *,
    is_strict_3d: bool = False,
) -> tuple[int, ...]:
    """Validate and normalize a positive-integer grid shape.

    Parameters
    ----------
    input_data
        Sequence of positive integer dimensions.
    name : str, optional
        Reader-facing name used in validation errors.
    is_strict_3d : bool, optional
        If True, require exactly three dimensions. Otherwise any non-empty
        dimensionality is accepted.
    """
    is_strict_3d = as_bool(is_strict_3d, name="is_strict_3d")

    if isinstance(input_data, (str, bytes)):
        raise TypeError(f"{name!r} must be a sequence of positive integers.")

    try:
        values = tuple(input_data)
    except TypeError as exc:
        raise TypeError(f"{name!r} must be a sequence of positive integers.") from exc

    if not values:
        raise ValueError(f"{name!r} must contain at least one dimension.")
    if is_strict_3d and len(values) != 3:
        raise ValueError(
            f"{name!r} must contain exactly three dimensions. Got {values!r}."
        )

    result = []
    for i, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(
                f"{name!r}[{i}] must be an integer. Got {value!r}."
            )
        value = int(value)
        if value <= 0:
            raise ValueError(
                f"{name!r}[{i}] must be positive. Got {value!r}."
            )
        result.append(value)

    return tuple(result)
