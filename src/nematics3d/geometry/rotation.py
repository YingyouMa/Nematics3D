"""Rotation helpers for 3D vectors and ordered director sequences."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..core.result_base import ResultBase
from ..datatypes import as_points, as_vector


__all__ = [
    "RotationAxisResult",
    "find_rotation_axis",
    "rotation_matrix_from_vectors",
]


@dataclass(slots=True, frozen=True, repr=False)
class RotationAxisResult(ResultBase):
    """Result of fitting a common rotation axis to an ordered director sequence."""

    __result_name__: ClassVar[str] = "director rotation-axis fit"
    __field_docs__: ClassVar[dict[str, str]] = {
        "axis": (
            "Best-fit common rotation axis, oriented by the net ordered rotation "
            "of the directors."
        ),
        "orthogonality_score": (
            "Fractional score approaching 1 when directors lie close to a plane "
            "normal to the fitted axis."
        ),
        "rms_sin_theta": (
            "RMS magnitude of the director component along the fitted axis for "
            "normalized directors."
        ),
        "tilt_angle_degrees": "Angle in degrees corresponding to rms_sin_theta.",
        "rotation_consistency": (
            "Magnitude of net signed rotation divided by the total absolute "
            "signed rotation along the ordered sequence."
        ),
        "eigenvalues": (
            "Ascending eigenvalues of the director second-moment matrix used for "
            "the axis fit."
        ),
    }

    axis: np.ndarray
    orthogonality_score: float
    rms_sin_theta: float
    tilt_angle_degrees: float
    rotation_consistency: float
    eigenvalues: np.ndarray

    @property
    def metric(self) -> dict[str, object]:
        """Return the fit diagnostics as one shallow dictionary."""
        return {
            "orthogonality_score": self.orthogonality_score,
            "rms_sin_theta": self.rms_sin_theta,
            "tilt_angle_degrees": self.tilt_angle_degrees,
            "rotation_consistency": self.rotation_consistency,
            "eigenvalues": self.eigenvalues,
        }


def find_rotation_axis(directors) -> RotationAxisResult:
    """Fit a common rotation axis to an ordered sequence of 3D unit directors.

    The fitted axis is the eigenvector associated with the smallest eigenvalue
    of ``directors.T @ directors``. Its sign is then oriented by the summed
    cross products of consecutive directors, so the returned direction follows
    the net ordered rotation when that rotation is nonzero.

    Parameters
    ----------
    directors : array-like, shape (N, 3)
        Ordered finite 3D unit directors. At least two directors are required.

    Returns
    -------
    RotationAxisResult
        The fitted axis together with fixed fit diagnostics.
    """
    directors = as_points(
        directors,
        d=3,
        name="directors used to fit a rotation axis",
        min_num=2,
    )

    norms = np.linalg.norm(directors, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-7, atol=1e-10):
        raise ValueError("All directors must be normalized unit vectors.")

    moment = directors.T @ directors
    eigenvalues, eigenvectors = np.linalg.eigh(moment)
    axis = eigenvectors[:, 0]

    cross_products = np.cross(directors[:-1], directors[1:])
    average_cross = np.sum(cross_products, axis=0)
    if np.dot(axis, average_cross) < 0:
        axis = -axis

    total_variance = float(np.sum(eigenvalues))
    orthogonality_score = (
        1.0 - float(eigenvalues[0]) / total_variance if total_variance > 0.0 else 0.0
    )
    rms_sin_theta = float(np.sqrt(max(float(eigenvalues[0]), 0.0) / len(directors)))
    tilt_angle_degrees = float(np.degrees(np.arcsin(np.clip(rms_sin_theta, -1.0, 1.0))))

    signed_rotation_steps = cross_products @ axis
    total_signed_rotation = float(np.sum(signed_rotation_steps))
    total_rotation_magnitude = float(np.sum(np.abs(signed_rotation_steps)))
    rotation_consistency = (
        abs(total_signed_rotation) / total_rotation_magnitude
        if total_rotation_magnitude > 1e-12
        else 0.0
    )

    return RotationAxisResult(
        axis=axis,
        orthogonality_score=float(orthogonality_score),
        rms_sin_theta=rms_sin_theta,
        tilt_angle_degrees=tilt_angle_degrees,
        rotation_consistency=float(rotation_consistency),
        eigenvalues=eigenvalues,
    )


def rotation_matrix_from_vectors(source_vector, target_vector) -> np.ndarray:
    """Return the minimal 3D rotation mapping one direction onto another.

    Parameters
    ----------
    source_vector, target_vector : array-like, shape (3,)
        Finite non-zero 3D vectors. Their magnitudes are ignored; each input is
        normalized internally before the rotation is constructed.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Proper orthogonal rotation matrix ``R`` satisfying
        ``R`` maps the direction of ``source_vector`` onto the direction of
        ``target_vector`` up to floating-point error.

    Notes
    -----
    Parallel vectors return the identity matrix. Antiparallel vectors admit
    infinitely many 180-degree rotations; this implementation chooses a
    deterministic axis by crossing the source vector with the Cartesian basis
    direction least aligned with it.
    """
    source = as_vector(
        source_vector,
        d=3,
        name="source_vector",
        is_normalized=True,
    )
    target = as_vector(
        target_vector,
        d=3,
        name="target_vector",
        is_normalized=True,
    )

    cosine = float(np.clip(source @ target, -1.0, 1.0))
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))

    if sine <= 1e-12:
        if cosine > 0.0:
            return np.eye(3)

        basis = np.zeros(3)
        basis[int(np.argmin(np.abs(source)))] = 1.0
        axis = np.cross(source, basis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    axis = cross / sine
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)
