"""Plane-fitting helpers for 3D point clouds."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..core.result_base import ResultBase
from ..datatypes import as_points


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


def find_plane_normal(points) -> PlaneNormalResult:
    """Fit a least-squares plane to finite 3D points.

    The fitted plane passes through the point-cloud centroid. Its normal is the
    eigenvector associated with the smallest eigenvalue of the centered
    second-moment matrix, equivalently the unit vector minimizing the summed
    squared perpendicular distances from the input points to the plane.

    Parameters
    ----------
    points : array-like, shape (N, 3)
        Finite 3D points. At least three points are required.

    Returns
    -------
    PlaneNormalResult
        Fitted unit normal, centroid, and fixed diagnostics describing
        planarity, thickness, and normal-direction degeneracy.
    """
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
    # ``eigh`` can return a tiny positive eigenvalue for an exactly rank-2
    # point cloud. Treat values at the matrix roundoff scale as numerical zero
    # so exact planes report zero thickness without masking real finite noise.
    eigenvalue_tol = (
        np.finfo(eigenvalues.dtype).eps
        * moment.shape[0]
        * max(float(eigenvalues[-1]), 1.0)
    )
    eigenvalues[eigenvalues <= eigenvalue_tol] = 0.0
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

    return PlaneNormalResult(
        normal=normal,
        centroid=centroid,
        planarity_score=planarity_score,
        thickness_rms=thickness_rms,
        linearity_risk=float(np.clip(linearity_risk, 0.0, 1.0)),
        eigenvalues=eigenvalues,
    )
