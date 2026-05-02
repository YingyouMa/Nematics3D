"""NML principal-plane analysis helpers.

Here "principal" refers to the eigenframe of the local mean S=1 Q tensor,
not to PCA or OBB geometric principal axes.  The N-M plane is the 2D plane
that retains the largest local director-orientation variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .classes.bounds import (
    Bounds,
    bounds_expanded,
    bounds_minimal_wrapping_points,
    bounds_sample_points,
    obb_bounds_from_fit,
)
from .classes.result_base import ResultBase
from .datatypes import as_dimension_info, as_points
from .field import Q_diagonalize, align_directors, getQ
from .geometry import obb_fit_approx


@dataclass(slots=True, frozen=True, repr=False)
class NMLIterationResult(ResultBase):
    """One self-consistent NML principal-plane iteration."""

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
    """Full self-consistent NML principal-plane analysis result."""

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


def nml_mean_s_equals_one_q(directors: np.ndarray) -> np.ndarray:
    """Average S=1 Q tensors reconstructed from local director orientations."""

    directors = np.asarray(directors, dtype=float)
    if directors.ndim < 2 or directors.shape[-1] != 3:
        raise ValueError(
            "`directors` must contain director vectors with trailing shape (3,)."
        )
    if directors.size == 0:
        raise ValueError("`directors` cannot be empty.")

    q_values = getQ(directors.reshape(-1, 3), S=1)
    return np.mean(q_values, axis=0)


def nml_axes_from_mean_q(mean_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize a local mean Q tensor into descending N, M, L axes."""

    mean_q = np.asarray(mean_q, dtype=float)
    if mean_q.shape != (3, 3):
        raise ValueError(f"`mean_q` must have shape (3, 3), got {mean_q.shape}.")

    eigenvalues, eigenvectors = np.linalg.eigh(mean_q)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] = -axes[:, -1]
    return eigenvalues[order], axes


def nml_axes_from_q_values(
    q_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover local directors from Q values and return mean-Q NML axes."""

    q_values = np.asarray(q_values, dtype=float)
    if q_values.size == 0:
        raise ValueError("`q_values` cannot be empty.")

    _, directors = Q_diagonalize(q_values)
    mean_q = nml_mean_s_equals_one_q(directors)
    eigenvalues, axes = nml_axes_from_mean_q(mean_q)
    return mean_q, eigenvalues, axes


def align_nml_axes_to_reference(
    axes: np.ndarray, reference_axes: np.ndarray
) -> np.ndarray:
    """Flip N/M/L signs to match a reference frame under nematic symmetry."""

    axes = _as_axes(axes, name="axes")
    reference_axes = _as_axes(reference_axes, name="reference_axes")
    aligned_axes = align_directors(reference_axes.T, axes.T).T
    if np.linalg.det(aligned_axes) < 0:
        aligned_axes[:, -1] = -aligned_axes[:, -1]
    return aligned_axes


def nml_axis_angle_changes_deg(
    axes: np.ndarray, reference_axes: np.ndarray
) -> np.ndarray:
    """Return unsigned per-axis angle changes between two NML frames."""

    axes = _as_axes(axes, name="axes")
    reference_axes = _as_axes(reference_axes, name="reference_axes")
    cosines = np.sum(axes * reference_axes, axis=0)
    cosines = np.clip(np.abs(cosines), 0.0, 1.0)
    return np.degrees(np.arccos(cosines))


def nml_seed_bounds_from_points(
    points,
    *,
    name: str | None = "NML seed bounds",
    angle_scales_deg=(15.0, 5.0, 1.0, 0.2),
    trials_per_scale=64,
    seed=None,
) -> Bounds:
    """Build the OBB seed bounds used as the required geometry for NML analysis."""

    fit = obb_fit_approx(
        points,
        angle_scales_deg=angle_scales_deg,
        trials_per_scale=trials_per_scale,
        seed=seed,
    )
    return obb_bounds_from_fit(fit, name=name)


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
    """Run the self-consistent NML principal-plane iteration.

    Parameters
    ----------
    q_obj
        Object exposing ``act_interpolate(points, is_index=...)``.
    required_points
        Geometry that every iteration must enclose in the current axes frame.
        Pass loop points for the direct workflow, or seed bounds corners for
        the OBB-seeded workflow.
    seed_bounds
        Optional seed ``Bounds``.  When supplied and ``required_points`` is
        omitted, the iteration encloses ``seed_bounds.corners``.
    initial_axes
        Starting axes with columns as axes.  Defaults to lab-frame identity.
    expand_factors, min_lengths, spacing
        Scalar or per-axis values controlling sampling bounds construction.
    angle_tol_deg
        Stop once all N/M/L axis changes are below this tolerance.
    max_iterations
        Maximum number of self-consistency iterations.
    min_sample_points
        Minimum interpolation sample count accepted per iteration.
    is_index
        Passed through to ``q_obj.act_interpolate``.
    origin
        Optional projection origin used when wrapping ``required_points``.
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
        dim=3,
        min_num=1,
    )

    axes = np.eye(3, dtype=float) if initial_axes is None else _as_axes(initial_axes)
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

    initial_axes_use = axes.copy()
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

        q_values = q_obj.act_interpolate(sample_points, is_index=is_index)
        mean_q, eigenvalues, new_axes = nml_axes_from_q_values(q_values)
        new_axes = align_nml_axes_to_reference(new_axes, axes)
        angle_changes = nml_axis_angle_changes_deg(new_axes, axes)
        max_axis_angle = float(np.max(angle_changes))

        iterations.append(
            NMLIterationResult(
                iteration=iteration,
                minimal_bounds=minimal_bounds,
                expanded_bounds=expanded_bounds,
                sample_count=sample_count,
                mean_q=mean_q,
                eigenvalues=eigenvalues,
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


def _as_axes(axes, name: str = "axes") -> np.ndarray:
    """Validate and right-hand a 3D axes matrix stored by columns."""

    axes = np.asarray(axes, dtype=float).copy()
    if axes.shape != (3, 3):
        raise ValueError(f"`{name}` must have shape (3, 3), got {axes.shape}.")
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-8):
        raise ValueError(f"`{name}` must be an orthonormal axes frame.")
    if np.linalg.det(axes) < 0:
        axes[:, -1] = -axes[:, -1]
    return axes
