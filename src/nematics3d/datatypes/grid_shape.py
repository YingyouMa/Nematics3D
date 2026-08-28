"""Grid-shape runtime converter."""

from collections.abc import Iterable, Mapping, Set

import numpy as np

from .bool import as_bool


def as_grid_shape(
    input_data: Iterable[int],
    name: str = "grid shape",
    *,
    is_strict_3d: bool = False,
) -> tuple[int, ...]:
    """Validate and normalize a positive-integer grid shape.

    Parameters
    ----------
    input_data : iterable of int
        Ordered iterable of positive integer dimensions. Mappings, sets,
        strings, and bytes are rejected because they do not provide a valid
        ordered shape representation.
    name : str, optional
        Reader-facing name used in validation errors.
    is_strict_3d : bool, optional
        If True, require exactly three dimensions. Otherwise any non-empty
        dimensionality is accepted.

    Returns
    -------
    tuple of int
        Validated dimensions normalized to Python integers.

    Raises
    ------
    TypeError
        If the input is not an ordered iterable of integer dimensions.
    ValueError
        If the shape is empty, contains a non-positive dimension, or does not
        have three dimensions when ``is_strict_3d=True``.
    """
    is_strict_3d = as_bool(is_strict_3d, name="is_strict_3d")

    if isinstance(input_data, (str, bytes, Mapping, Set)):
        raise TypeError(f"{name!r} must be an ordered iterable of positive integers.")

    try:
        values = tuple(input_data)
    except TypeError as exc:
        raise TypeError(
            f"{name!r} must be an ordered iterable of positive integers."
        ) from exc

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
            raise TypeError(f"{name!r}[{i}] must be an integer. Got {value!r}.")
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name!r}[{i}] must be positive. Got {value!r}.")
        result.append(value)

    return tuple(result)
