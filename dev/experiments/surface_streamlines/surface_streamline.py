"""Experimental integration of nematic streamlines on triangle surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from director_interpolation import interpolate_surface_directors


@dataclass(slots=True, frozen=True)
class SurfaceStreamlineResult:
    """One bidirectional surface streamline and termination diagnostics."""

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


def _trace_one_direction(
    surface: pv.PolyData,
    vertex_directors: np.ndarray,
    seed_position: np.ndarray,
    initial_direction: np.ndarray,
    *,
    step_size: float,
    max_length: float,
    max_steps: int,
    min_step_fraction: float,
    stop_tree: cKDTree | None,
    minimum_separation: float | None,
) -> tuple[np.ndarray, str]:
    """Trace one oriented branch with projected midpoint steps."""
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

        midpoint_direction = midpoint.directors[0]
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

        points.append(next_position)
        accumulated_length += displacement
        current = next_position
        reference = endpoint.directors[0]

        if accumulated_length >= max_length:
            return np.asarray(points), "maximum length"
        if len(points) > 8 and np.linalg.norm(current - points[0]) < 0.75 * step_size:
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
    stop_positions=None,
    minimum_separation=None,
) -> SurfaceStreamlineResult:
    """Integrate one bidirectional streamline on a nematic surface field.

    A midpoint step is taken in ambient coordinates and each intermediate
    position is projected back to its closest triangle location. The previous
    direction is passed to the nematic interpolator so ``n``/``-n`` choices
    remain continuous along each branch.
    """
    if not isinstance(surface, pv.PolyData):
        raise TypeError("surface must be a pyvista.PolyData.")
    step_size = _as_positive_real(step_size, name="step_size")
    max_length = _as_positive_real(max_length, name="max_length")
    max_steps = _as_positive_integer(max_steps, name="max_steps")
    min_step_fraction = _as_positive_real(
        min_step_fraction,
        name="min_step_fraction",
    )
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

    initial = interpolate_surface_directors(
        surface,
        vertex_directors,
        seed_position,
    )
    if not initial.is_interpolable[0]:
        raise ValueError("seed position does not have an interpolable director.")
    seed = initial.surface_positions[0]
    direction = initial.directors[0]

    forward, forward_status = _trace_one_direction(
        surface,
        vertex_directors,
        seed,
        direction,
        step_size=step_size,
        max_length=0.5 * max_length,
        max_steps=max_steps,
        min_step_fraction=min_step_fraction,
        stop_tree=stop_tree,
        minimum_separation=minimum_separation,
    )
    backward, backward_status = _trace_one_direction(
        surface,
        vertex_directors,
        seed,
        -direction,
        step_size=step_size,
        max_length=0.5 * max_length,
        max_steps=max_steps,
        min_step_fraction=min_step_fraction,
        stop_tree=stop_tree,
        minimum_separation=minimum_separation,
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
