"""Nearest-point geometry queries."""

import numpy as np


__all__ = ["find_nearest_point"]


def find_nearest_point(query_pt, coords, is_return_idx=False):
    """Find the nearest point in ``coords`` to ``query_pt`` in Euclidean distance."""
    q = np.asarray(query_pt, dtype=float).reshape(-1)
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != len(q):
        raise ValueError(
            f"`coords` shape is {pts.shape},"
            f"while normalized `query_pt` shape is {q.shape}"
        )

    delta = pts - q
    distance_squared = np.einsum("ij,ij->i", delta, delta)
    idx = int(np.argmin(distance_squared))
    return (pts[idx], idx) if is_return_idx else pts[idx]
