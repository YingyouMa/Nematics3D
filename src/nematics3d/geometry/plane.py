"""Plane-fitting helpers for 3D point clouds."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..core.result_base import ResultBase
from ..datatypes import as_bool, as_points


__all__ = ["PlaneNormalResult", "find_plane_normal"]


@dataclass(slots=True, frozen=True, repr=False)
class PlaneNormalResult(ResultBase):
    """Result of fitting one average plane to a 3D point cloud."""

    __result_name__: ClassVar[str] = "point-cloud plane fit"
    __field_docs__: ClassVar[dict[str, str]] = {
        "normal": (
            "Unit normal of the least-squares best-fit plane. Its sign is "
            "intrinsically ambiguous."
        ),
        "centroid": (
            "Centroid of the input points; the fitted plane passes through it."
        ),
        "planarity_score": (
            "Dimensionless score in [0, 1] approaching 1 when the point cloud "
            "has little variance normal to the fitted plane."
        ),
        "thickness_rms": (
            "Root-mean-square point-cloud thickness along the fitted normal."
        ),
        "linearity_risk": (
            "Ratio of the two smallest eigenvalues; values near 1 indicate that "
            "the fitted normal is poorly determined because the cloud is close "
            "to one-dimensional."
        ),
        "eigenvalues": (
            "Ascending eigenvalues of the centered point-cloud second-moment "
            "matrix."
        ),
    }

    normal: np.ndarray
    centroid: np.ndarray
    planarity_score: float
    thickness_rms: float
    linearity_risk: float
    eigenvalues: np.ndarray

    @property
    def metric(self) -> dict[str, object]:
        """Return the fit diagnostics as one shallow dictionary."""
        return {
            "centroid": self.centroid,
            "planarity_score": self.planarity_score,
            "thickness_rms": self.thickness_rms,
            "eigenvalues": self.eigenvalues,
            "linearity_risk": self.linearity_risk,
        }


class _LegacyPlaneNormalResult(PlaneNormalResult):
    """Temporary private adapter for one unmigrated internal tuple-unpack caller."""

    def __iter__(self):
        yield self.normal
        yield self.metric


def find_plane_normal(points, is_return_metric=False) -> PlaneNormalResult:
    """Fit a least-squares plane to finite 3D points.

    The fitted plane passes through the point-cloud centroid. Its normal is the
    eigenvector associated with the smallest eigenvalue of the centered
    second-moment matrix, equivalently the unit vector minimizing the summed
    squared perpendicular distances from the input points to the plane.

    Parameters
    ----------
    points : array-like, shape (N, 3)
        Finite 3D points. At least three points are required.
    is_return_metric : bool, default=False
        Temporary compatibility flag for the current internal
        ``DisclinationLine.act_calc_norm()`` caller. New code should omit this
        argument and consume the returned ``PlaneNormalResult`` directly.

    Returns
    -------
    PlaneNormalResult
        Fitted unit normal, centroid, and fixed diagnostics describing
        planarity, thickness, and normal-direction degeneracy.
    """
    is_return_metric = as_bool(is_return_metric, name="is_return_metric")
    points = as_points(
        points,
        d=3,
        name="points used to fit a plane",
        min_num=3,
    )

    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    moment = centered_points.T @ centered_points
    eigenvalues, eigenvectors = np.linalg.eigh(moment)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    normal = eigenvectors[:, 0]

    total_variance = float(np.sum(eigenvalues))
    if total_variance > 0.0:
        planarity_score = 1.0 - 3.0 * float(eigenvalues[0]) / total_variance
        planarity_score = float(np.clip(planarity_score, 0.0, 1.0))
    else:
        planarity_score = 1.0

    thickness_rms = float(np.sqrt(float(eigenvalues[0]) / len(points)))
    second_eigenvalue = float(eigenvalues[1])
    linearity_risk = (
        float(eigenvalues[0]) / second_eigenvalue
        if second_eigenvalue > 0.0
        else 1.0
    )

    result_type = _LegacyPlaneNormalResult if is_return_metric else PlaneNormalResult
    return result_type(
        normal=normal,
        centroid=centroid,
        planarity_score=planarity_score,
        thickness_rms=thickness_rms,
        linearity_risk=float(np.clip(linearity_risk, 0.0, 1.0)),
        eigenvalues=eigenvalues,
    )
