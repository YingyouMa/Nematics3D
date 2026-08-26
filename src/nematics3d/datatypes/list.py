"""Lightweight normalization helper for single-or-multiple inputs."""


def as_list(input_data) -> list:
    """Return *input_data* as a list without interpreting scalars as iterables.

    Lists are returned unchanged, tuples and sets are converted element-wise,
    and every other object is treated as one item and wrapped in a one-element
    list. In particular, strings, ranges, generators, and NumPy arrays are
    treated as single objects rather than expanded implicitly.
    """
    if isinstance(input_data, list):
        return input_data
    if isinstance(input_data, (tuple, set)):
        return list(input_data)
    return [input_data]
