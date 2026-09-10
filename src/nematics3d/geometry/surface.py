"""Low-level geometry helpers for triangulated surfaces."""

from __future__ import annotations

import numpy as np
import pyvista as pv


def _as_surface_points(values, *, name: str) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.ndim == 1 and points.shape == (3,):
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {points.shape}.")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values.")
    return points


def surface_triangle_coordinates(
    surface: pv.PolyData,
    points,
    *,
    project_to_surface: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve triangle membership and barycentric coordinates for surface points.

    Parameters
    ----------
    surface
        Triangulated ``pyvista.PolyData`` surface.
    points
        Query positions with shape ``(N, 3)`` or one position with shape ``(3,)``.
    project_to_surface
        If true, first replace every query by its closest point on ``surface``.
        If false, barycentric coordinates are evaluated at the supplied positions.

    Returns
    -------
    tuple
        ``(cell_indices, surface_positions, triangle_point_indices, barycentric)``.
    """
    if not isinstance(surface, pv.PolyData):
        raise TypeError("surface must be a pyvista.PolyData.")
    if surface.n_points == 0 or surface.n_cells == 0:
        raise ValueError("surface must contain points and cells.")
    if not surface.is_all_triangles:
        raise ValueError("surface must contain only triangle cells.")

    queries = _as_surface_points(points, name="points")
    if project_to_surface:
        cell_indices, surface_positions = surface.find_closest_cell(
            queries,
            return_closest_point=True,
        )
        surface_positions = np.atleast_2d(
            np.asarray(surface_positions, dtype=float)
        )
    else:
        cell_indices = surface.find_closest_cell(queries)
        surface_positions = queries

    cell_indices = np.atleast_1d(np.asarray(cell_indices, dtype=np.int64))
    faces_flat = np.asarray(surface.faces, dtype=np.int64)
    if faces_flat.size != 4 * surface.n_cells:
        raise ValueError("surface has an unexpected triangle-connectivity layout.")
    faces = faces_flat.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("surface has an unexpected non-triangle cell.")

    triangle_point_indices = faces[cell_indices, 1:]
    triangles = np.asarray(surface.points, dtype=float)[triangle_point_indices]
    a = triangles[:, 0]
    edge_0 = triangles[:, 1] - a
    edge_1 = triangles[:, 2] - a
    relative = surface_positions - a

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
            "surface coordinate resolution encountered degenerate triangles at "
            f"query indices including {indices[:10].tolist()}."
        )

    weight_1 = (dot_11 * dot_20 - dot_01 * dot_21) / denominator
    weight_2 = (dot_00 * dot_21 - dot_01 * dot_20) / denominator
    barycentric = np.column_stack((1.0 - weight_1 - weight_2, weight_1, weight_2))
    barycentric = np.clip(barycentric, 0.0, 1.0)
    barycentric /= np.sum(barycentric, axis=1, keepdims=True)
    return cell_indices, surface_positions, triangle_point_indices, barycentric


__all__ = ["surface_triangle_coordinates"]
