"""Analysis of directors defined on triangulated surfaces."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from numbers import Real
from typing import ClassVar

import numpy as np
import pyvista as pv

from ..core.result_base import ResultBase
from ..geometry.surface import surface_triangle_coordinates


@dataclass(slots=True, frozen=True, repr=False)
class SurfaceDirectorProjectionResult(ResultBase):
    """Projection of one director per surface vertex onto local tangent planes."""

    __result_name__: ClassVar[str] = "surface director tangent projection"

    projected_directors: np.ndarray
    surface_normals: np.ndarray
    tilt_angles_degrees: np.ndarray
    normal_fractions: np.ndarray
    tangent_fractions: np.ndarray
    is_projectable: np.ndarray
    exceeded_indices: np.ndarray
    max_tilt_degrees: float | None


@dataclass(slots=True, frozen=True, repr=False)
class SurfaceDirectorInterpolationResult(ResultBase):
    """Nematic director interpolation results at surface query positions."""

    __result_name__: ClassVar[str] = "surface director interpolation"

    directors: np.ndarray
    surface_positions: np.ndarray
    surface_distances: np.ndarray
    cell_indices: np.ndarray
    barycentric_coordinates: np.ndarray
    is_interpolable: np.ndarray


def _readonly(values, *, dtype) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _as_bounded_float(value, *, name: str, lower: float, upper: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    result = float(value)
    if not np.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}].")
    return result


def _as_directors(values) -> np.ndarray:
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


def _as_points(values, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim == 1 and raw.shape == (3,):
        raw = raw[None, :]
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {raw.shape}.")
    if raw.dtype.kind not in "biuf":
        raise TypeError(f"{name} must contain real numeric values, got dtype {raw.dtype}.")
    points = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values.")
    return points


def _normalized(values: np.ndarray, *, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(values, axis=-1)
    is_valid = norms > tolerance
    result = np.zeros_like(values, dtype=float)
    np.divide(values, norms[..., None], out=result, where=is_valid[..., None])
    return result, is_valid


def _as_reference_directors(
    values,
    *,
    query_count: int,
    tolerance: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if values is None:
        return None, None
    raw = np.asarray(values)
    if raw.shape == (3,):
        raw = np.broadcast_to(raw, (query_count, 3))
    references = _as_points(raw, name="reference_directors")
    if len(references) != query_count:
        raise ValueError(
            "reference_directors must contain one vector per query position."
        )
    return _normalized(references, tolerance=tolerance)


def _surface_point_normals(surface: pv.PolyData) -> np.ndarray:
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

    ``surface`` must be a :class:`pyvista.PolyData`. No cleaning, resampling, or
    reconstruction is performed, so the input vertex identities and ordering
    remain the reference for ``directors``.

    Tilt is reported as the unsigned nematic angle away from the local tangent
    plane. Zero directors and directors parallel to the local normal cannot be
    normalized into tangent directions and are marked by ``is_projectable``.
    """
    if not isinstance(surface, pv.PolyData):
        raise TypeError(
            "surface must be a pyvista.PolyData, got "
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
                "surface; vertex indices include "
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


def interpolate_surface_directors(
    surface,
    vertex_directors,
    positions,
    *,
    reference_directors=None,
    norm_tolerance=1.0e-10,
) -> SurfaceDirectorInterpolationResult:
    """Interpolate a head-tail-symmetric director field on a triangle surface.

    Query positions are mapped to their closest locations on ``surface``. The
    three vertex directors of each containing triangle are sign-aligned before
    barycentric interpolation, preventing equivalent ``n`` and ``-n`` values
    from cancelling. ``reference_directors`` may be supplied to select the
    returned sign continuously along a trajectory such as a streamline.
    """
    if not isinstance(surface, pv.PolyData):
        raise TypeError("surface must be a pyvista.PolyData.")

    tolerance = _as_bounded_float(
        norm_tolerance,
        name="norm_tolerance",
        lower=0.0,
        upper=np.inf,
    )
    queries = _as_points(positions, name="positions")
    directors = _as_points(vertex_directors, name="vertex_directors")
    if len(directors) != surface.n_points:
        raise ValueError(
            "vertex_directors must contain exactly one vector per surface vertex."
        )
    directors, is_vertex_valid = _normalized(directors, tolerance=tolerance)
    references, is_reference_valid = _as_reference_directors(
        reference_directors,
        query_count=len(queries),
        tolerance=tolerance,
    )

    cell_indices, surface_positions, triangle_indices, barycentric = (
        surface_triangle_coordinates(
            surface,
            queries,
            project_to_surface=True,
        )
    )
    triangle_directors = directors[triangle_indices].copy()
    triangle_is_valid = is_vertex_valid[triangle_indices]

    if references is None:
        anchor_slots = np.argmax(triangle_is_valid, axis=1)
        anchors = triangle_directors[np.arange(len(queries)), anchor_slots]
        is_anchor_valid = np.any(triangle_is_valid, axis=1)
    else:
        anchors = references.copy()
        is_anchor_valid = is_reference_valid.copy()
        fallback_slots = np.argmax(triangle_is_valid, axis=1)
        fallbacks = triangle_directors[np.arange(len(queries)), fallback_slots]
        use_fallback = ~is_anchor_valid & np.any(triangle_is_valid, axis=1)
        anchors[use_fallback] = fallbacks[use_fallback]
        is_anchor_valid |= use_fallback

    alignments = np.einsum("nij,nj->ni", triangle_directors, anchors)
    is_flip = triangle_is_valid & is_anchor_valid[:, None] & (alignments < 0.0)
    triangle_directors[is_flip] *= -1.0

    interpolated = np.einsum("ni,nij->nj", barycentric, triangle_directors)
    interpolated, is_interpolable = _normalized(interpolated, tolerance=tolerance)
    if references is not None:
        is_flip_result = (
            is_interpolable
            & is_reference_valid
            & (np.einsum("ij,ij->i", interpolated, references) < 0.0)
        )
        interpolated[is_flip_result] *= -1.0

    distances = np.linalg.norm(queries - surface_positions, axis=1)
    return SurfaceDirectorInterpolationResult(
        directors=_readonly(interpolated, dtype=float),
        surface_positions=_readonly(surface_positions, dtype=float),
        surface_distances=_readonly(distances, dtype=float),
        cell_indices=_readonly(cell_indices, dtype=np.int64),
        barycentric_coordinates=_readonly(barycentric, dtype=float),
        is_interpolable=_readonly(is_interpolable, dtype=bool),
    )


__all__ = [
    "SurfaceDirectorInterpolationResult",
    "SurfaceDirectorProjectionResult",
    "interpolate_surface_directors",
    "project_surface_directors",
]
