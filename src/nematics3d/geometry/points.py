"""Point-set geometry helpers."""

import numpy as np

from ..datatypes import as_points


__all__ = ["points_membership_mask"]


def points_membership_mask(points, candidates) -> np.ndarray:
    """Return whether each point appears exactly in a candidate point set.

    Both inputs must be finite real point collections with shape ``(N, D)``
    and the same coordinate dimension ``D``. Numerical dtypes may differ;
    both arrays are promoted to their common NumPy dtype before comparison.
    Floating-point coordinates are compared exactly, without a tolerance.

    Duplicate rows do not change the result: each returned entry is simply
    whether the corresponding row of ``points`` occurs at least once in
    ``candidates``.
    """
    points_raw = np.asarray(points)
    candidates_raw = np.asarray(candidates)
    if points_raw.ndim != 2 or candidates_raw.ndim != 2:
        raise ValueError(
            "points and candidates must both be two-dimensional arrays. "
            f"Got shapes {points_raw.shape} and {candidates_raw.shape}."
        )

    points_array = as_points(
        points,
        d=None,
        name="points",
        is_empty=True,
    )
    candidates_array = as_points(
        candidates,
        d=None,
        name="candidates",
        is_empty=True,
    )
    if points_array.shape[1] != candidates_array.shape[1]:
        raise ValueError(
            "points and candidates must have the same coordinate dimension. "
            f"Got {points_array.shape[1]} and {candidates_array.shape[1]}."
        )

    if len(points_array) == 0:
        return np.empty(0, dtype=bool)
    if len(candidates_array) == 0:
        return np.zeros(len(points_array), dtype=bool)

    common_dtype = np.result_type(points_array.dtype, candidates_array.dtype)
    points_array = np.ascontiguousarray(points_array, dtype=common_dtype)
    candidates_array = np.ascontiguousarray(candidates_array, dtype=common_dtype)

    row_dtype = np.dtype(
        [(f"coord_{i}", common_dtype) for i in range(points_array.shape[1])]
    )
    points_rows = points_array.view(row_dtype).reshape(-1)
    candidate_rows = candidates_array.view(row_dtype).reshape(-1)

    return np.isin(points_rows, candidate_rows)
