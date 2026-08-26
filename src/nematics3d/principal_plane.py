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
)
from .classes.result_base import ResultBase
from .datatypes import as_axes, as_dimension_info, as_points
from .field import get_q
from .geometry import align_axes_to_reference, axes_angle_changes_deg
from .analysis.q_diagonalization import q_diagonalize


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

    This function finds a local N/M/L frame from the director texture inside an
    oriented sampling box.  The frame is "principal" in the NML sense: it comes
    from the eigenvectors of the local mean S=1 Q tensor, not from PCA or from
    the geometric long/short axes of a loop.

    The central loop is:

    1. In the current axes frame, build the smallest ``Bounds`` that encloses
       the required geometry.
    2. Expand that minimal box so the Q sampling region includes enough local
       neighborhood around the required geometry.
    3. Sample Q on a regular grid inside the expanded oriented box.
    4. Recover the local director at each sample point, rebuild each director
       as an S=1 Q tensor, and average those tensors.
    5. Diagonalize the mean Q.  Its eigenvectors define the updated N/M/L axes.
    6. Flip eigenvector signs to stay close to the previous axes, then stop when
       the largest per-axis change is below ``angle_tol_deg``.

    ``required_points`` and ``seed_bounds`` intentionally separate geometry
    from texture analysis.  The direct workflow can pass loop points as
    ``required_points``.  The OBB-seeded workflow can pass ``seed_bounds`` and
    let this function enclose ``seed_bounds.corners`` at every iteration.  In
    both cases, N/M/L itself is computed only from sampled Q values.

    The returned principal plane is the final N-M plane:

    - ``result.plane_axes`` stores the final N and M axes as columns;
    - ``result.plane_normal`` stores the final L axis;
    - ``result.plane_center`` is the center of the final minimal bounds in the
      converged/final NML frame.

    Each per-iteration result stores the minimal and expanded bounds used for
    that iteration.  After convergence, the function rebuilds final minimal and
    expanded bounds once more using the final axes, so the top-level
    ``minimal_bounds`` and ``expanded_bounds`` match the returned plane.

    Parameters
    ----------
    q_obj
        Object exposing ``act_interpolate(points, is_index=...)``.  In normal
        repository use this is a ``QFieldObject`` or compatible object.
    required_points
        Geometry that every iteration must enclose in the current axes frame.
        Pass loop points for the direct workflow, or seed bounds corners for
        the OBB-seeded workflow.  These points constrain the sampling box only;
        they do not directly determine the NML axes.
    seed_bounds
        Optional seed ``Bounds``.  When supplied and ``required_points`` is
        omitted, the iteration encloses ``seed_bounds.corners``.  This supports
        the workflow where an initial geometric proxy is preserved while the
        texture-derived NML frame iterates.
    initial_axes
        Starting axes with columns as axes.  Defaults to lab-frame identity.
        This is only the initial sampling-frame guess, not a claim that the lab
        axes are physically principal.
    expand_factors
        Scalar or per-axis multiplicative expansion applied to the minimal box
        lengths before sampling Q.
    min_lengths
        Scalar or per-axis lower bound on box side lengths.  This prevents thin
        or nearly planar required geometry from producing too few sample points.
    spacing
        Scalar or per-axis sample spacing used to generate the regular grid
        inside each expanded bounds.
    angle_tol_deg
        Stop once all N/M/L axis changes are below this tolerance.  Axis signs
        are aligned before measuring the angle, respecting nematic ``n == -n``
        symmetry.
    max_iterations
        Maximum number of self-consistency iterations.
    min_sample_points
        Minimum interpolation sample count accepted per iteration.
    is_index
        Passed through to ``q_obj.act_interpolate``.  The current local trial
        script uses index-space sample points, so the default is ``True``.
    origin
        Optional projection origin used when wrapping ``required_points`` into
        the current axes frame.  By default the required-point centroid is used,
        which keeps the projection numerically centered for unwrapped loops.
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
        # Geometry step: in the current candidate NML frame, find the smallest
        # box that still encloses the required geometry.  This box is a
        # constraint container; it is not where the NML axes come from.
        minimal_bounds = bounds_minimal_wrapping_points(
            required_points,
            axes,
            origin=projection_origin,
            name=f"NML minimal bounds {iteration}",
            min_lengths=min_lengths,
        )

        # Sampling step: enlarge the minimal container before interpolation.
        # The expanded bounds is the actual local neighborhood whose director
        # texture will define the next N/M/L frame.
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

        # Texture step: sample Q, discard scalar-order weighting by rebuilding
        # directors as S=1 Q tensors, and diagonalize their mean.  This is the
        # operation that makes N-M the principal plane of the local director
        # distribution.
        q_values = q_obj.act_interpolate(sample_points, is_index=is_index)
        q_values = np.asarray(q_values, dtype=float)
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
        eigenvalues = diagonalization.eigenvalues
        new_axes = diagonalization.eigenvectors

        # Eigenvectors are sign-ambiguous and nematic directors satisfy
        # n == -n.  Sign-align to the previous frame before measuring rotation,
        # otherwise a harmless sign flip would look like a large update.
        new_axes = align_axes_to_reference(new_axes, axes)
        angle_changes = axes_angle_changes_deg(new_axes, axes)
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

    # The last per-iteration bounds were built from the axes before applying
    # that iteration's NML update.  Rebuild once in the final axes so the
    # top-level bounds and plane metadata describe the same final frame.
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
