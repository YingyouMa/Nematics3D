"""Approximate oriented bounding-box fitting for 3D point clouds."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation as R

from ..core.result_base import ResultBase
from ..datatypes import as_axes, as_dimension_info, as_points, as_vector

__all__ = [
    "OBBFit",
    "box_corners_from_center_axes_radii",
    "canonicalize_axes",
    "compute_convex_hull_points",
    "obb_fit_approx",
    "obb_fit_pca",
    "obb_refine_random_search",
]


def box_corners_from_center_axes_radii(center, axes, radii) -> np.ndarray:
    """Return oriented-box corners from center, column-wise axes, and radii."""
    center = as_vector(center, name="box center", d=3)
    axes = as_axes(axes, name="box axes")
    radii = as_dimension_info(radii, name="box radii").astype(float)
    if np.any(radii <= 0):
        raise ValueError("`radii` must contain only positive values.")

    local_corners = np.array(
        [
            [-radii[0], -radii[1], -radii[2]],
            [radii[0], -radii[1], -radii[2]],
            [-radii[0], radii[1], -radii[2]],
            [-radii[0], -radii[1], radii[2]],
            [radii[0], radii[1], -radii[2]],
            [radii[0], -radii[1], radii[2]],
            [-radii[0], radii[1], radii[2]],
            [radii[0], radii[1], radii[2]],
        ],
        dtype=float,
    )
    return center + local_corners @ axes.T


@dataclass(slots=True, frozen=True, repr=False)
class OBBFit(ResultBase):
    """Pure-geometry result of an oriented bounding-box fit."""

    __result_name__: ClassVar[str] = "The parameters of an oriented bounding-box fit"

    axes: np.ndarray
    center: np.ndarray
    lengths: np.ndarray
    local_min: np.ndarray
    local_max: np.ndarray
    volume: float


def compute_convex_hull_points(points):
    """Return unique convex-hull vertices, or unique input points if degenerate."""
    points = as_points(
        points,
        name="points used to compute a convex hull",
        d=3,
        is_unique=True,
        min_num=1,
    )
    if len(points) <= 3:
        return points

    try:
        hull = ConvexHull(points)
    except QhullError:
        return points

    return points[np.unique(hull.vertices)]


def canonicalize_axes(axes):
    """Canonicalize a 3D column-wise frame and make it right-handed."""
    axes = np.asarray(axes, dtype=float).copy()
    if axes.shape != (3, 3):
        raise ValueError(f"Expected axes to have shape (3, 3), got {axes.shape}.")

    for axis_index in range(3):
        axis = axes[:, axis_index]
        pivot = int(np.argmax(np.abs(axis)))
        if axis[pivot] < 0:
            axes[:, axis_index] = -axis

    if np.linalg.det(axes) < 0:
        axes[:, -1] = -axes[:, -1]

    return axes


def obb_fit_pca(points):
    """Fit a deterministic PCA-oriented bounding box to 3D points."""
    points = as_points(
        points,
        name="points used to fit a PCA oriented bounding box",
        d=3,
        is_unique=True,
        min_num=1,
    )

    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    covariance = centered_points.T @ centered_points / len(points)

    if np.allclose(covariance, 0.0):
        axes = np.eye(3)
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        axes = eigenvectors[:, order]

    return _obb_fit_in_axes(points, canonicalize_axes(axes))


def obb_refine_random_search(
    points,
    initial_fit: OBBFit,
    *,
    angle_scales_deg=(15.0, 5.0, 1.0, 0.2),
    trials_per_scale=64,
    seed=None,
):
    """Refine an OBB fit with approximate multi-scale random rotation search."""
    points = as_points(
        points,
        name="points used to refine an oriented bounding box",
        d=3,
        is_unique=True,
        min_num=1,
    )
    if not isinstance(initial_fit, OBBFit):
        raise TypeError("`initial_fit` must be an OBBFit.")

    trials_per_scale = int(trials_per_scale)
    if trials_per_scale < 1:
        raise ValueError("`trials_per_scale` must be at least 1.")

    angle_scales_deg = np.asarray(angle_scales_deg, dtype=float)
    if angle_scales_deg.ndim != 1 or len(angle_scales_deg) == 0:
        raise ValueError("`angle_scales_deg` must be a non-empty 1D sequence.")
    if np.any(angle_scales_deg < 0):
        raise ValueError("`angle_scales_deg` cannot contain negative values.")

    rng = np.random.default_rng(seed)
    best_fit = initial_fit

    for angle_scale_deg in angle_scales_deg:
        angle_scale_rad = np.deg2rad(angle_scale_deg)
        for _ in range(trials_per_scale):
            rotvec = rng.normal(size=3)
            rotvec_norm = float(np.linalg.norm(rotvec))
            if rotvec_norm == 0.0 or angle_scale_rad == 0.0:
                continue

            rotvec *= rng.normal(scale=angle_scale_rad) / rotvec_norm
            rotation = R.from_rotvec(rotvec).as_matrix()
            candidate_fit = _obb_fit_in_axes(points, rotation @ best_fit.axes)
            if candidate_fit.volume < best_fit.volume:
                best_fit = candidate_fit

    return best_fit


def obb_fit_approx(
    points,
    *,
    angle_scales_deg=(15.0, 5.0, 1.0, 0.2),
    trials_per_scale=64,
    seed=None,
):
    """Fit an approximate minimum-volume OBB to 3D points."""
    hull_points = compute_convex_hull_points(points)
    initial_fit = obb_fit_pca(hull_points)
    return obb_refine_random_search(
        hull_points,
        initial_fit,
        angle_scales_deg=angle_scales_deg,
        trials_per_scale=trials_per_scale,
        seed=seed,
    )


def _obb_fit_in_axes(points, axes):
    """Return the smallest OBB wrapping points in the supplied frame."""
    points = as_points(
        points,
        name="points used to fit an oriented bounding box",
        d=3,
        min_num=1,
    )
    axes = canonicalize_axes(axes)

    local_points = points @ axes
    local_min = np.min(local_points, axis=0)
    local_max = np.max(local_points, axis=0)
    lengths = local_max - local_min
    local_center = 0.5 * (local_min + local_max)
    center = local_center @ axes.T

    return OBBFit(
        axes=axes,
        center=center,
        lengths=lengths,
        local_min=local_min,
        local_max=local_max,
        volume=float(np.prod(lengths)),
    )
