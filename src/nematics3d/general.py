"""Legacy helpers that have not yet moved to dedicated modules."""

import numpy as np


__all__ = [
    "closest_point_on_polyline",
    "find_nearest_point",
    "mark_points_membership",
    "select_grid_in_box",
]


def __getattr__(name):
    """Temporarily resolve migrated helpers for legacy internal imports."""
    if name in {
        "find_rotation_axis",
        "get_box_corners",
        "rotation_matrix_from_vectors",
    }:
        from . import geometry

        return getattr(geometry, name)
    if name in {"get_square", "get_square_each"}:
        from .analysis.disclination.line import get_square, get_square_each

        return {"get_square": get_square, "get_square_each": get_square_each}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def select_grid_in_box(
    grid: np.ndarray,
    corners_limit: np.ndarray | None,
    is_return_mask: bool = False,
    logger=None,
):
    """Filter 3D points by an oriented rectangular box."""
    grid = np.asarray(grid)
    n = grid.shape[0] if grid.ndim >= 1 else 0

    if n == 0:
        if logger is not None:
            logger.warning("Input `grid` is empty; returning an empty selection.")
        mask_empty = np.zeros((0,), dtype=bool)
        return (grid, mask_empty) if is_return_mask else grid

    if corners_limit is None:
        mask_all = np.ones((n,), dtype=bool)
        return (grid, mask_all) if is_return_mask else grid

    corners_limit = np.asarray(corners_limit)
    if (
        corners_limit.ndim != 2
        or corners_limit.shape[1] != 3
        or corners_limit.shape[0] < 4
    ):
        raise ValueError(
            f"`corners_limit` must have shape (>=4, 3). Got {corners_limit.shape} instead."
        )

    axes = [corners_limit[i] - corners_limit[0] for i in range(1, 4)]
    lengths = [np.linalg.norm(axis) for axis in axes]

    if any(length <= 0.0 for length in lengths):
        raise ValueError(
            "Degenerate `corners_limit`: box edge length(s) must be positive. "
            f"Got lengths={lengths}."
        )

    unit_axes = [axis / length for axis, length in zip(axes, lengths)]
    rel = grid - corners_limit[0]
    coords = np.stack([rel @ axis for axis in unit_axes], axis=1)

    tol = 1e-9
    bounds = np.array(lengths, dtype=coords.dtype)
    mask = np.all((coords >= -tol) & (coords <= bounds + tol), axis=1)

    grid_selected = grid[mask]
    if grid_selected.shape[0] == 0 and logger is not None:
        logger.warning(
            "No points from `grid` fall inside the specified box defined by "
            f"`corners_limit`:\n{corners_limit}"
        )

    return (grid_selected, mask) if is_return_mask else grid_selected


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


def closest_point_on_polyline(query_pt: np.ndarray, poly_pts: np.ndarray) -> np.ndarray:
    """Compute the closest point on a polyline to a query point in 3D."""
    q = np.asarray(query_pt, dtype=float)
    pts = np.asarray(poly_pts, dtype=float)

    if pts.shape[0] == 1:
        return pts[0].copy()

    a = pts[:-1]
    b = pts[1:]
    ab = b - a
    aq = q - a

    ab2 = np.einsum("ij,ij->i", ab, ab)
    ab2 = np.where(ab2 <= 1e-30, 1e-30, ab2)

    t = np.einsum("ij,ij->i", aq, ab) / ab2
    t = np.clip(t, 0.0, 1.0)

    proj = a + ab * t[:, None]
    diff = proj - q
    d2 = np.einsum("ij,ij->i", diff, diff)

    idx = int(np.argmin(d2))
    return proj[idx]
