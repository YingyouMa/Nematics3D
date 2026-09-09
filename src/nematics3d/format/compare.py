"""Small comparison and selector predicates."""

import numpy as np


def is_equal_array(v1, v2):
    """Compare array-like values as ndarrays, treating NaNs as equal."""
    try:
        arr1 = np.asarray(v1)
        arr2 = np.asarray(v2)
    except Exception as exc:
        raise TypeError(
            f"Input value cannot be converted to numpy array: {exc}"
        ) from exc
    return np.array_equal(arr1, arr2, equal_nan=True)


def is_equal(v1, v2):
    """Safely compare scalar or array-like values."""
    try:
        return is_equal_array(v1, v2)
    except (TypeError, ValueError):
        try:
            result = v1 == v2
        except Exception:
            return False
        return bool(result) if np.isscalar(result) else False


def is_given_str(value, expected):
    """Return whether ``value`` is exactly the expected string."""
    return isinstance(value, str) and value == expected


__all__ = ["is_equal", "is_equal_array", "is_given_str"]
