"""Experimental interpolation of a nematic director field on a triangle mesh."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pyvista as pv


@dataclass(slots=True, frozen=True)
class SurfaceDirectorInterpolationResult:
    """Outputs aligned with the supplied surface query positions."""

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


def _as_points(values, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim == 1 and raw.shape == (3,):
        raw = raw[None, :]
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {raw.shape}.")
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values, got {raw.dtype}.")
    result = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _normalized(
    values: np.ndarray, *, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(values, axis=-1)
    is_valid = norms > tolerance
    result = np.zeros_like(values, dtype=float)
    np.divide(values, norms[..., None], out=result, where=is_valid[..., None])
    return result, is_valid


def _as_positive_tolerance(value, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _triangle_faces(surface: pv.PolyData) -> np.ndarray:
    if surface.n_points == 0 or surface.n_cells == 0:
        raise ValueError("surface must contain points and cells.")
    if not surface.is_all_triangles:
        raise ValueError("surface must contain only triangle cells.")
    faces_flat = np.asarray(surface.faces, dtype=np.int64)
    if faces_flat.size != 4 * surface.n_cells:
        raise ValueError("surface has an unexpected triangle-connectivity layout.")
    faces = faces_flat.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("surface has an unexpected non-triangle cell.")
    return faces[:, 1:]


def _barycentric_coordinates(
    positions: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """Resolve barycentrics for positions already projected onto triangles."""
    a = triangles[:, 0]
    edge_0 = triangles[:, 1] - a
    edge_1 = triangles[:, 2] - a
    relative = positions - a

    dot_00 = np.einsum("ij,ij->i", edge_0, edge_0)
    dot_01 = np.einsum("ij,ij->i", edge_0, edge_1)
    dot_11 = np.einsum("ij,ij->i", edge_1, edge_1)
    dot_20 = np.einsum("ij,ij->i", relative, edge_0)
    dot_21 = np.einsum("ij,ij->i", relative, edge_1)
    denominator = dot_00 * dot_11 - dot_01 * dot_01
    scale = np.maximum(dot_00 * dot_11, 1.0)
    is_degenerate = np.abs(denominator) <= np.finfo(float).eps * scale
    if np.any(is_degenerate):
        indices = np.flatnonzero(is_degenerate)
        raise ValueError(
            "queried surface cells contain degenerate triangles at query indices "
            f"including {indices[:10].tolist()}."
        )

    weight_1 = (dot_11 * dot_20 - dot_01 * dot_21) / denominator
    weight_2 = (dot_00 * dot_21 - dot_01 * dot_20) / denominator
    barycentric = np.column_stack((1.0 - weight_1 - weight_2, weight_1, weight_2))
    barycentric = np.clip(barycentric, 0.0, 1.0)
    barycentric /= np.sum(barycentric, axis=1, keepdims=True)
    return barycentric


def _as_references(
    references,
    *,
    query_count: int,
    tolerance: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if references is None:
        return None, None
    raw = np.asarray(references)
    if raw.shape == (3,):
        raw = np.broadcast_to(raw, (query_count, 3))
    values = _as_points(raw, name="reference_directors")
    if len(values) != query_count:
        raise ValueError(
            "reference_directors must contain one vector per query position."
        )
    return _normalized(values, tolerance=tolerance)


def interpolate_surface_directors(
    surface,
    vertex_directors,
    positions,
    *,
    reference_directors=None,
    norm_tolerance=1.0e-10,
) -> SurfaceDirectorInterpolationResult:
    """Interpolate a head-tail-symmetric vertex field at surface positions.

    Query positions are first mapped to their closest triangle locations. The
    three vertex directors are locally sign-aligned before barycentric
    interpolation, so equivalent ``n`` and ``-n`` inputs reinforce rather than
    cancel. Supplying ``reference_directors`` selects the returned sign and is
    recommended during streamline integration to maintain step continuity.
    """
    if not isinstance(surface, pv.PolyData):
        raise TypeError("surface must be a pyvista.PolyData.")
    faces = _triangle_faces(surface)
    queries = _as_points(positions, name="positions")
    tolerance = _as_positive_tolerance(norm_tolerance, name="norm_tolerance")

    directors = _as_points(vertex_directors, name="vertex_directors")
    if len(directors) != surface.n_points:
        raise ValueError(
            "vertex_directors must contain exactly one vector per surface vertex."
        )
    directors, is_vertex_valid = _normalized(directors, tolerance=tolerance)
    references, is_reference_valid = _as_references(
        reference_directors,
        query_count=len(queries),
        tolerance=tolerance,
    )

    cell_indices, surface_positions = surface.find_closest_cell(
        queries,
        return_closest_point=True,
    )
    cell_indices = np.atleast_1d(np.asarray(cell_indices, dtype=np.int64))
    surface_positions = np.atleast_2d(np.asarray(surface_positions, dtype=float))
    triangle_indices = faces[cell_indices]
    triangle_points = np.asarray(surface.points, dtype=float)[triangle_indices]
    barycentric = _barycentric_coordinates(surface_positions, triangle_points)

    triangle_directors = directors[triangle_indices].copy()
    triangle_is_valid = is_vertex_valid[triangle_indices]

    if references is None:
        anchor_slots = np.argmax(triangle_is_valid, axis=1)
        anchors = triangle_directors[np.arange(len(queries)), anchor_slots]
        is_anchor_valid = np.any(triangle_is_valid, axis=1)
    else:
        anchors = references
        is_anchor_valid = is_reference_valid
        fallback_slots = np.argmax(triangle_is_valid, axis=1)
        fallbacks = triangle_directors[np.arange(len(queries)), fallback_slots]
        use_fallback = ~is_anchor_valid & np.any(triangle_is_valid, axis=1)
        anchors = anchors.copy()
        anchors[use_fallback] = fallbacks[use_fallback]
        is_anchor_valid = is_anchor_valid | use_fallback

    alignments = np.einsum("nij,nj->ni", triangle_directors, anchors)
    is_flip = triangle_is_valid & is_anchor_valid[:, None] & (alignments < 0.0)
    triangle_directors[is_flip] *= -1.0

    interpolated = np.einsum("ni,nij->nj", barycentric, triangle_directors)
    interpolated, is_interpolable = _normalized(
        interpolated,
        tolerance=tolerance,
    )
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
    "interpolate_surface_directors",
]
