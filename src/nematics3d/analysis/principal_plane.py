"""Self-consistent N/M/L principal-plane analysis for local Q textures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..core.result_base import ResultBase
from ..datatypes import as_axes, as_dimension_info, as_points
from ..geometry import align_axes_to_reference, axes_angle_changes_deg
from ..q_field import get_q, q_diagonalize
from .bounds import (
    Bounds,
    bounds_expanded,
    bounds_minimal_wrapping_points,
    bounds_sample_points,
)

__all__ = [
    "NMLIterationResult",
    "NMLPrincipalPlaneResult",
    "nml_principal_plane_analysis",
]


@dataclass(slots=True, frozen=True, repr=False)
class NMLIterationResult(ResultBase):
    """One self-consistent N/M/L principal-frame iteration."""

    __result_name__: ClassVar[str] = "one NML principal-plane iteration"

    iteration: int
    minimal_bounds: Bounds
    expanded_bounds: Bounds
    sample_count: int
    mean_q: np.ndarray
    eigenvalues: np.ndarray
    axes: np.ndarray
    angle_changes_deg: np.ndarray
    max_axis_angle_deg: float


@dataclass(slots=True, frozen=True, repr=False)
class NMLPrincipalPlaneResult(ResultBase):
    """Full self-consistent N/M/L principal-plane analysis result."""

    __result_name__: ClassVar[str] = "NML principal-plane analysis"

    seed_bounds: Bounds | None
    required_points: np.ndarray
    initial_axes: np.ndarray
    iterations: tuple[NMLIterationResult, ...]
    minimal_bounds: Bounds
    expanded_bounds: Bounds
    axes: np.ndarray
    plane_center: np.ndarray
    plane_axes: np.ndarray
    plane_normal: np.ndarray
    converged: bool


def nml_principal_plane_analysis(
    q_obj,
    required_points=None,
    *,
    seed_bounds: Bounds | None = None,
    initial_axes=None,
    expand_factors=1.5,
    min_lengths=0.0,
    spacing=1.0,
    angle_tol_deg=1.0,
    max_iterations=20,
    min_sample_points=1,
    is_index=True,
    origin=None,
) -> NMLPrincipalPlaneResult:
    """Find the local N/M/L frame and return its N-M principal plane.

    ``required_points`` or ``seed_bounds`` constrain only the sampling region.
    The frame itself is texture-derived: Q is sampled locally, reduced to
    directors, rebuilt as S=1 Q tensors, averaged, and diagonalized.  The new
    eigenframe is then used to rebuild the sampling box until convergence.

    This is not a PCA/OBB principal plane.  An OBB may be supplied through
    ``seed_bounds`` as a geometric seed without changing that distinction.
    """

    if seed_bounds is not None and not isinstance(seed_bounds, Bounds):
        raise TypeError("`seed_bounds` must be a Bounds instance or None.")

    if required_points is None:
        if seed_bounds is None:
            raise ValueError("Pass `required_points` or `seed_bounds`.")
        required_points = seed_bounds.corners
    required_points = as_points(
        required_points,
        name="required points used for NML principal-plane analysis",
        d=3,
        min_num=1,
    )

    axes = np.eye(3, dtype=float) if initial_axes is None else as_axes(initial_axes)
    initial_axes_use = axes.copy()

    expand_factors = as_dimension_info(expand_factors, name="expand_factors").astype(
        float
    )
    if np.any(expand_factors <= 0):
        raise ValueError("`expand_factors` must contain only positive values.")

    min_lengths = as_dimension_info(min_lengths, name="min_lengths").astype(float)
    if np.any(min_lengths < 0):
        raise ValueError("`min_lengths` cannot contain negative values.")

    spacing = as_dimension_info(spacing, name="spacing").astype(float)
    if np.any(spacing <= 0):
        raise ValueError("`spacing` must contain only positive values.")

    angle_tol_deg = float(angle_tol_deg)
    if angle_tol_deg < 0:
        raise ValueError("`angle_tol_deg` cannot be negative.")
    max_iterations = int(max_iterations)
    if max_iterations < 1:
        raise ValueError("`max_iterations` must be at least 1.")
    min_sample_points = int(min_sample_points)
    if min_sample_points < 1:
        raise ValueError("`min_sample_points` must be at least 1.")

    projection_origin = (
        np.mean(required_points, axis=0)
        if origin is None
        else np.asarray(origin, dtype=float)
    )
    if projection_origin.shape != (3,):
        raise ValueError(
            f"`origin` must have shape (3,), got {projection_origin.shape}."
        )

    iterations: list[NMLIterationResult] = []
    converged = False

    for iteration in range(max_iterations):
        minimal_bounds = bounds_minimal_wrapping_points(
            required_points,
            axes,
            origin=projection_origin,
            name=f"NML minimal bounds {iteration}",
            min_lengths=min_lengths,
        )
        expanded_bounds = bounds_expanded(
            minimal_bounds,
            expand_factors,
            min_lengths=min_lengths,
            name=f"NML expanded bounds {iteration}",
        )
        sample_points = bounds_sample_points(expanded_bounds, spacing=spacing)
        sample_count = int(len(sample_points))
        if sample_count < min_sample_points:
            raise RuntimeError(
                f"Only {sample_count} interpolation sample points at iteration "
                f"{iteration}; check expand_factors, min_lengths, and spacing."
            )

        q_values = np.asarray(
            q_obj.act_interpolate(sample_points, is_index=is_index), dtype=float
        )
        if q_values.size == 0:
            raise ValueError("Interpolated Q values cannot be empty.")

        directors = q_diagonalize(q_values).n
        mean_q = np.mean(
            get_q(np.asarray(directors, dtype=float).reshape(-1, 3), S=1), axis=0
        )
        diagonalization = q_diagonalize(
            mean_q,
            is_biaxial=True,
            is_right_handed=True,
        )
        new_axes = align_axes_to_reference(diagonalization.eigenvectors, axes)
        angle_changes = axes_angle_changes_deg(new_axes, axes)
        max_axis_angle = float(np.max(angle_changes))

        iterations.append(
            NMLIterationResult(
                iteration=iteration,
                minimal_bounds=minimal_bounds,
                expanded_bounds=expanded_bounds,
                sample_count=sample_count,
                mean_q=mean_q,
                eigenvalues=diagonalization.eigenvalues,
                axes=new_axes,
                angle_changes_deg=angle_changes,
                max_axis_angle_deg=max_axis_angle,
            )
        )

        axes = new_axes
        if max_axis_angle <= angle_tol_deg:
            converged = True
            break

    final_minimal_bounds = bounds_minimal_wrapping_points(
        required_points,
        axes,
        origin=projection_origin,
        name="NML final minimal bounds",
        min_lengths=min_lengths,
    )
    final_expanded_bounds = bounds_expanded(
        final_minimal_bounds,
        expand_factors,
        min_lengths=min_lengths,
        name="NML final expanded bounds",
    )

    return NMLPrincipalPlaneResult(
        seed_bounds=seed_bounds,
        required_points=required_points,
        initial_axes=initial_axes_use,
        iterations=tuple(iterations),
        minimal_bounds=final_minimal_bounds,
        expanded_bounds=final_expanded_bounds,
        axes=axes,
        plane_center=final_minimal_bounds.opts.origin,
        plane_axes=axes[:, :2],
        plane_normal=axes[:, 2],
        converged=converged,
    )
