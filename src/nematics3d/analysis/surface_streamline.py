"""Integration of nematic line fields on triangulated surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import ClassVar

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from ..core.result_base import ResultBase
from .surface_director import interpolate_surface_directors


@dataclass(slots=True, frozen=True, repr=False)
class SurfaceStreamlineResult(ResultBase):
    """One bidirectional surface streamline and its termination diagnostics."""

    __result_name__: ClassVar[str] = "surface streamline"

    positions: np.ndarray
    seed_position: np.ndarray
    forward_status: str
    backward_status: str
    length: float


def _readonly(values) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _as_positive_real(value, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _as_positive_integer(value, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _surface_point_normals(surface: pv.PolyData) -> np.ndarray:
    """Return smooth unit point normals without changing surface topology."""
    surface_with_normals = surface.compute_normals(
        cell_normals=False,
        point_normals=True,
        split_vertices=False,
        consistent_normals=True,
        auto_orient_normals=False,
        inplace=False,
    )
    normals = np.asarray(surface_with_normals.point_data["Normals"], dtype=float)
    if normals.shape != (surface.n_points, 3) or not np.all(np.isfinite(normals)):
        raise ValueError("surface must have finite point normals with shape (N, 3).")
    magnitudes = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(magnitudes <= 1.0e-12):
        raise ValueError("surface must have finite, nonzero point normals.")
    return normals / magnitudes


def _triangle_point_indices(surface: pv.PolyData) -> np.ndarray:
    if surface.n_points == 0 or surface.n_cells == 0:
        raise ValueError("surface must contain points and cells.")
    if not surface.is_all_triangles:
        raise ValueError("surface must contain only triangle cells.")
    faces = np.asarray(surface.faces, dtype=np.int64)
    if faces.size != 4 * surface.n_cells:
        raise ValueError("surface has an unexpected triangle-connectivity layout.")
    faces = faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("surface has an unexpected non-triangle cell.")
    return faces[:, 1:]


def _tangent_direction(
    interpolation,
    *,
    point_normals: np.ndarray,
    triangle_indices: np.ndarray,
    reference: np.ndarray | None,
    tolerance: float,
) -> np.ndarray | None:
    """Project an interpolated nematic director onto the local smooth tangent plane."""
    cell_id = int(interpolation.cell_indices[0])
    barycentric = interpolation.barycentric_coordinates[0]
    vertex_ids = triangle_indices[cell_id]
    normal = np.einsum("i,ij->j", barycentric, point_normals[vertex_ids])
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= tolerance:
        return None
    normal /= normal_norm

    direction = np.array(interpolation.directors[0], dtype=float, copy=True)
    direction -= np.dot(direction, normal) * normal
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= tolerance:
        return None
    direction /= direction_norm

    if reference is not None and np.dot(direction, reference) < 0.0:
        direction *= -1.0
    return direction


def _trace_one_direction(
    surface: pv.PolyData,
    vertex_directors: np.ndarray,
    point_normals: np.ndarray,
    triangle_indices: np.ndarray,
    seed_position: np.ndarray,
    initial_direction: np.ndarray,
    *,
    step_size: float,
    max_length: float,
    max_steps: int,
    min_step_fraction: float,
    closure_tolerance: float,
    minimum_closure_length: float,
    tangent_tolerance: float,
    stop_tree: cKDTree | None,
    minimum_separation: float | None,
) -> tuple[np.ndarray, str]:
    """Trace one oriented branch with tangent-projected midpoint steps."""
    points = [np.asarray(seed_position, dtype=float)]
    current = points[0]
    reference = np.asarray(initial_direction, dtype=float)
    accumulated_length = 0.0

    for _ in range(max_steps):
        midpoint_query = current + 0.5 * step_size * reference
        midpoint = interpolate_surface_directors(
            surface,
            vertex_directors,
            midpoint_query,
            reference_directors=reference,
        )
        if not midpoint.is_interpolable[0]:
            return np.asarray(points), "non-interpolable midpoint"
        midpoint_direction = _tangent_direction(
            midpoint,
            point_normals=point_normals,
            triangle_indices=triangle_indices,
            reference=reference,
            tolerance=tangent_tolerance,
        )
        if midpoint_direction is None:
            return np.asarray(points), "undefined tangent direction"

        endpoint_query = current + step_size * midpoint_direction
        endpoint = interpolate_surface_directors(
            surface,
            vertex_directors,
            endpoint_query,
            reference_directors=midpoint_direction,
        )
        if not endpoint.is_interpolable[0]:
            return np.asarray(points), "non-interpolable endpoint"

        next_position = endpoint.surface_positions[0]
        displacement = float(np.linalg.norm(next_position - current))
        if displacement < min_step_fraction * step_size:
            return np.asarray(points), "stagnated at surface constraint"
        if stop_tree is not None and len(points) > 2:
            distance, _ = stop_tree.query(next_position, k=1)
            if distance < minimum_separation:
                return np.asarray(points), "minimum separation"

        accumulated_length += displacement
        points.append(next_position)
        current = next_position

        endpoint_direction = _tangent_direction(
            endpoint,
            point_normals=point_normals,
            triangle_indices=triangle_indices,
            reference=midpoint_direction,
            tolerance=tangent_tolerance,
        )
        if endpoint_direction is None:
            return np.asarray(points), "undefined tangent direction"
        reference = endpoint_direction

        if accumulated_length >= max_length:
            return np.asarray(points), "maximum length"
        if (
            accumulated_length >= minimum_closure_length
            and np.linalg.norm(current - points[0]) <= closure_tolerance
        ):
            points[-1] = points[0]
            return np.asarray(points), "closed loop"

    return np.asarray(points), "maximum steps"


def integrate_surface_streamline(
    surface,
    vertex_directors,
    seed_position,
    *,
    step_size=0.35,
    max_length=24.0,
    max_steps=300,
    min_step_fraction=1.0e-3,
    closure_tolerance=None,
    minimum_closure_length=None,
    tangent_tolerance=1.0e-10,
    stop_positions=None,
    minimum_separation=None,
) -> SurfaceStreamlineResult:
    """Integrate one bidirectional streamline of a nematic field on a surface.

    The vertex line field is barycentrically interpolated on the closest
    triangle. At every midpoint and endpoint, the interpolated director is
    explicitly reprojected onto the local tangent plane obtained from smooth
    interpolated point normals. The previous oriented direction resolves the
    otherwise equivalent ``n`` and ``-n`` signs along each branch.

    ``max_length`` is the maximum arc length allowed for each traced branch.
    For an open line the combined bidirectional result may therefore approach
    ``2 * max_length``. A closed line is returned as soon as one branch closes
    and is never duplicated by tracing the opposite branch around the same loop.

    ``closure_tolerance`` defaults to ``0.75 * step_size`` and
    ``minimum_closure_length`` defaults to ``4 * step_size``. They are exposed
    explicitly because closed-loop recognition is a geometric stopping policy,
    not part of the tangent-field definition.
    """
    if not isinstance(surface, pv.PolyData):
        raise TypeError("surface must be a pyvista.PolyData.")
    step_size = _as_positive_real(step_size, name="step_size")
    max_length = _as_positive_real(max_length, name="max_length")
    max_steps = _as_positive_integer(max_steps, name="max_steps")
    min_step_fraction = _as_positive_real(min_step_fraction, name="min_step_fraction")
    tangent_tolerance = _as_positive_real(tangent_tolerance, name="tangent_tolerance")
    if closure_tolerance is None:
        closure_tolerance = 0.75 * step_size
    closure_tolerance = _as_positive_real(closure_tolerance, name="closure_tolerance")
    if minimum_closure_length is None:
        minimum_closure_length = 4.0 * step_size
    minimum_closure_length = _as_positive_real(
        minimum_closure_length,
        name="minimum_closure_length",
    )

    triangle_indices = _triangle_point_indices(surface)
    point_normals = _surface_point_normals(surface)

    if (stop_positions is None) != (minimum_separation is None):
        raise ValueError(
            "stop_positions and minimum_separation must be provided together."
        )
    if stop_positions is None:
        stop_tree = None
    else:
        stop_positions = np.asarray(stop_positions, dtype=float)
        if stop_positions.ndim != 2 or stop_positions.shape[1] != 3:
            raise ValueError("stop_positions must have shape (N, 3).")
        if not np.all(np.isfinite(stop_positions)):
            raise ValueError("stop_positions must contain only finite values.")
        minimum_separation = _as_positive_real(
            minimum_separation,
            name="minimum_separation",
        )
        stop_tree = cKDTree(stop_positions) if len(stop_positions) else None

    initial = interpolate_surface_directors(surface, vertex_directors, seed_position)
    if not initial.is_interpolable[0]:
        raise ValueError("seed position does not have an interpolable director.")
    seed = initial.surface_positions[0]
    direction = _tangent_direction(
        initial,
        point_normals=point_normals,
        triangle_indices=triangle_indices,
        reference=None,
        tolerance=tangent_tolerance,
    )
    if direction is None:
        raise ValueError("seed position does not have a nonzero tangent director.")

    trace_kwargs = dict(
        step_size=step_size,
        max_length=max_length,
        max_steps=max_steps,
        min_step_fraction=min_step_fraction,
        closure_tolerance=closure_tolerance,
        minimum_closure_length=minimum_closure_length,
        tangent_tolerance=tangent_tolerance,
        stop_tree=stop_tree,
        minimum_separation=minimum_separation,
    )
    forward, forward_status = _trace_one_direction(
        surface,
        vertex_directors,
        point_normals,
        triangle_indices,
        seed,
        direction,
        **trace_kwargs,
    )
    if forward_status == "closed loop":
        positions = forward
        backward_status = "not traced: forward branch closed loop"
        length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
        return SurfaceStreamlineResult(
            positions=_readonly(positions),
            seed_position=_readonly(seed),
            forward_status=forward_status,
            backward_status=backward_status,
            length=length,
        )

    backward, backward_status = _trace_one_direction(
        surface,
        vertex_directors,
        point_normals,
        triangle_indices,
        seed,
        -direction,
        **trace_kwargs,
    )
    if backward_status == "closed loop":
        positions = backward
        forward_status = "not used: backward branch closed loop"
        length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
        return SurfaceStreamlineResult(
            positions=_readonly(positions),
            seed_position=_readonly(seed),
            forward_status=forward_status,
            backward_status=backward_status,
            length=length,
        )

    positions = np.vstack((backward[:0:-1], forward))
    length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    return SurfaceStreamlineResult(
        positions=_readonly(positions),
        seed_position=_readonly(seed),
        forward_status=forward_status,
        backward_status=backward_status,
        length=length,
    )


__all__ = ["SurfaceStreamlineResult", "integrate_surface_streamline"]
