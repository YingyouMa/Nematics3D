"""Experimental projection of vertex directors onto a surface tangent field.

This module is intentionally outside the public :mod:`nematics3d` package.
Its API may change while the surface-streamline workflow is being developed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pyvista as pv


@dataclass(slots=True, frozen=True)
class SurfaceDirectorProjectionResult:
    """Named outputs from :func:`project_surface_directors`."""

    projected_directors: np.ndarray
    surface_normals: np.ndarray
    tilt_angles_degrees: np.ndarray
    normal_fractions: np.ndarray
    tangent_fractions: np.ndarray
    is_projectable: np.ndarray
    exceeded_indices: np.ndarray
    max_tilt_degrees: float | None


def _readonly(values: np.ndarray, *, dtype) -> np.ndarray:
    """Return an independent read-only result array."""
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _as_bounded_float(value, *, name: str, lower: float, upper: float) -> float:
    """Validate one finite real scalar within a closed interval."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    result = float(value)
    if not np.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}].")
    return result


def _as_directors(values) -> np.ndarray:
    """Validate and normalize an ``(N, 3)`` real director array."""
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f"surface directors must have shape (N, 3), got {raw.shape}.")
    if raw.dtype.kind not in "biuf":
        raise TypeError(
            "surface directors must contain real numeric values, got dtype "
            f"{raw.dtype}."
        )

    directors = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(directors)):
        raise ValueError("surface directors must contain only finite values.")

    norms = np.linalg.norm(directors, axis=1, keepdims=True)
    normalized = np.zeros_like(directors)
    np.divide(directors, norms, out=normalized, where=norms > 1.0e-12)
    return normalized


def _surface_point_normals(surface: pv.PolyData) -> np.ndarray:
    """Compute smooth unit point normals without changing vertex identities."""
    if surface.n_points == 0 or surface.n_cells == 0:
        raise ValueError("surface must contain points and surface cells.")
    if float(surface.area) <= 0.0:
        raise ValueError("surface must have positive area.")

    surface_with_normals = surface.compute_normals(
        cell_normals=False,
        point_normals=True,
        split_vertices=False,
        consistent_normals=True,
        auto_orient_normals=False,
        inplace=False,
    )
    if surface_with_normals.n_points != surface.n_points:
        raise RuntimeError("Computing normals unexpectedly changed the vertex count.")

    normals = np.asarray(surface_with_normals.point_data["Normals"], dtype=float)
    if normals.shape != (surface.n_points, 3) or not np.all(np.isfinite(normals)):
        raise ValueError("surface point normals must be finite with shape (N, 3).")

    normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(normal_norms <= 1.0e-12):
        invalid_indices = np.flatnonzero(normal_norms[:, 0] <= 1.0e-12)
        raise ValueError(
            "surface contains undefined point normals at vertex indices including "
            f"{invalid_indices[:10].tolist()}."
        )
    return normals / normal_norms


def project_surface_directors(
    surface,
    directors,
    *,
    max_tilt_degrees=None,
    tangent_tolerance=1.0e-10,
) -> SurfaceDirectorProjectionResult:
    """Project one director per surface vertex onto its local tangent plane.

    Parameters
    ----------
    surface
        Surface-like PyVista/VTK data. Vertex order is preserved, and no point
        cleaning, resampling, or point-cloud reconstruction is performed.
    directors
        Director array with shape ``(surface.n_points, 3)``. Values are
        normalized internally. A zero director is accepted but marked as not
        projectable.
    max_tilt_degrees
        Optional warning threshold in the closed interval ``[0, 90]``. Tilt is
        the unsigned angle away from the reconstructed local tangent plane.
    tangent_tolerance
        A projection is valid only when its tangent fraction is strictly
        greater than this non-negative tolerance.

    Returns
    -------
    SurfaceDirectorProjectionResult
        Read-only projected directors, normals, diagnostics, and validity data
        aligned with the original surface vertices.

    Notes
    -----
    The output remains a nematic line field: ``n`` and ``-n`` are equivalent.
    This function deliberately does not orient neighboring directors or
    integrate streamlines.
    """
    if not isinstance(surface, pv.PolyData):
        raise TypeError(
            "surface must be a pyvista.PolyData in this experimental API, got "
            f"{type(surface).__name__}."
        )
    normalized_directors = _as_directors(directors)
    if normalized_directors.shape[0] != surface.n_points:
        raise ValueError(
            "surface directors must contain exactly one director per surface "
            f"vertex: got {normalized_directors.shape[0]} directors for "
            f"{surface.n_points} vertices."
        )

    tangent_tolerance = _as_bounded_float(
        tangent_tolerance,
        name="tangent_tolerance",
        lower=0.0,
        upper=np.inf,
    )
    if max_tilt_degrees is not None:
        max_tilt_degrees = _as_bounded_float(
            max_tilt_degrees,
            name="max_tilt_degrees",
            lower=0.0,
            upper=90.0,
        )

    surface_normals = _surface_point_normals(surface)
    director_norms = np.linalg.norm(normalized_directors, axis=1)
    is_nonzero = director_norms > 1.0e-12

    signed_normal_components = np.einsum(
        "ij,ij->i", normalized_directors, surface_normals
    )
    tangent_vectors = (
        normalized_directors - signed_normal_components[:, None] * surface_normals
    )
    tangent_fractions = np.linalg.norm(tangent_vectors, axis=1)
    is_projectable = is_nonzero & (tangent_fractions > tangent_tolerance)

    projected_directors = np.zeros_like(tangent_vectors)
    np.divide(
        tangent_vectors,
        tangent_fractions[:, None],
        out=projected_directors,
        where=is_projectable[:, None],
    )

    normal_fractions = np.abs(signed_normal_components)
    tilt_angles_degrees = np.degrees(np.arcsin(np.clip(normal_fractions, 0.0, 1.0)))
    normal_fractions[~is_nonzero] = np.nan
    tangent_fractions[~is_nonzero] = np.nan
    tilt_angles_degrees[~is_nonzero] = np.nan

    if max_tilt_degrees is None:
        exceeded_indices = np.empty(0, dtype=int)
    else:
        exceeded_indices = np.flatnonzero(
            is_nonzero & (tilt_angles_degrees > max_tilt_degrees)
        )
        if exceeded_indices.size:
            warnings.warn(
                f"{exceeded_indices.size} director(s) exceed the "
                f"{max_tilt_degrees:g}-degree tilt threshold relative to the "
                "reconstructed surface; vertex indices include "
                f"{exceeded_indices[:10].tolist()}.",
                RuntimeWarning,
                stacklevel=2,
            )

    return SurfaceDirectorProjectionResult(
        projected_directors=_readonly(projected_directors, dtype=float),
        surface_normals=_readonly(surface_normals, dtype=float),
        tilt_angles_degrees=_readonly(tilt_angles_degrees, dtype=float),
        normal_fractions=_readonly(normal_fractions, dtype=float),
        tangent_fractions=_readonly(tangent_fractions, dtype=float),
        is_projectable=_readonly(is_projectable, dtype=bool),
        exceeded_indices=_readonly(exceeded_indices, dtype=int),
        max_tilt_degrees=max_tilt_degrees,
    )


__all__ = [
    "SurfaceDirectorProjectionResult",
    "project_surface_directors",
]
