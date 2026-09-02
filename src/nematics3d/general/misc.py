"""General helpers that do not yet have a more specific module home."""

import numpy as np


__all__ = ["mark_points_membership"]


def mark_points_membership(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """Return whether each row in ``points1`` appears exactly in ``points2``."""
    a = np.ascontiguousarray(points1)
    b = np.ascontiguousarray(points2)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"points1 and points2 must be 2D arrays. Got {a.ndim=} and {b.ndim=}."
        )
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            "points1 and points2 must have the same number of columns. "
            f"Got {a.shape[1]=} vs {b.shape[1]=}."
        )

    row_dtype = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    a_view = a.view(row_dtype).ravel()
    b_view = b.view(row_dtype).ravel()

    return np.isin(a_view, b_view).reshape(-1)
