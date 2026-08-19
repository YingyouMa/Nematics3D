"""Analytic Q-tensor diagonalization and named result objects."""

from dataclasses import dataclass
from typing import ClassVar, Union

import numexpr as ne
import numpy as np

from .classes.result_base import ResultBase
from .datatypes import QField5, QField9, as_qfield9
from .logging_decorator import logging_and_warning_decorator


@dataclass(slots=True, frozen=True, repr=False)
class QDiagonalizationResult(ResultBase):
    """Named outputs from :func:`q_diagonalize`."""

    __result_name__: ClassVar[str] = "Q-tensor diagonalization"

    # fmt: off
    __field_docs__: ClassVar[dict[str, str]] = {
        "S":               "Scalar order: 3/2 times the largest eigenvalue.",
        "n":               "Unit eigenvector for the largest eigenvalue.",
        "isotropic_indices": (
            "Coordinate indices of points handled as numerically isotropic."
        ),
        "uniaxial_indices": (
            "Indices where the canonical uniaxial frame was applied; its "
            "degenerate axes may be spatially discontinuous."
        ),
        "eigenvalues":     "Descending eigenspectrum when biaxial output is requested.",
        "eigenvectors":    "Eigenvector columns matching the descending eigenvalues.",
        "biaxial_order":   "Biaxial order: 3/2 |lambda_1 - lambda_2|.",
    }
    # fmt: on

    S: np.ndarray  # noqa: N815 - conventional public symbol for scalar order
    n: np.ndarray
    isotropic_indices: np.ndarray
    uniaxial_indices: np.ndarray
    eigenvalues: np.ndarray | None = None
    eigenvectors: np.ndarray | None = None
    biaxial_order: np.ndarray | None = None


@logging_and_warning_decorator()
def q_diagonalize(
    qtensor: Union[QField5, QField9],
    *,
    is_biaxial: bool = False,
    logger=None,
) -> QDiagonalizationResult:
    """Diagonalize a Q-tensor field and return named physical quantities.

    The default path computes only the scalar order and dominant director using
    the vectorized invariant-based algorithm. Set ``is_biaxial=True`` to also
    return the complete descending eigenspectrum, matching eigenvector columns,
    and the biaxial order parameter defined by
    ``b = 1.5 * abs(lambda_1 - lambda_2)``.

    Parameters
    ----------
    qtensor : QField5 or QField9
        Q-tensor data with trailing shape ``(..., 5)`` or ``(..., 3, 3)``.
    is_biaxial : bool, optional
        Whether to compute and return the complete biaxial eigensystem.

    Returns
    -------
    QDiagonalizationResult
        Named ``S`` and ``n`` arrays plus the coordinate indices of numerical
        isotropic points and points receiving canonical uniaxial treatment.
        Complete eigenvalue, eigenvector, and biaxial-order arrays are included
        only when requested.

    Notes
    -----
    Negative-``S`` oblate (flat) uniaxial states are not currently supported.
    Uniaxial recovery assumes a unique largest eigenvalue and a repeated lower
    eigenvalue pair, as occurs for positive-``S`` prolate states.
    """
    # Normalize both accepted Q representations to full (..., 3, 3) matrices.
    full_tensor = as_qfield9(
        qtensor,
        name="Q tensor to diagonalize",
        is_strict_3d_field=False,
    )
    calculation_dtype = np.result_type(full_tensor.dtype, np.float64)
    full_tensor = np.asarray(full_tensor, dtype=calculation_dtype)

    # Build scale-aware tolerances so the isotropic decision follows the input
    # dtype and remains meaningful when Q has unusually large components.
    machine_epsilon = np.finfo(full_tensor.dtype).eps
    tensor_abs_max = np.max(np.abs(full_tensor), axis=(-2, -1))
    tensor_scale = np.maximum(1.0, tensor_abs_max)
    isotropic_tolerance = 32 * machine_epsilon * tensor_scale

    # Expose the six independent symmetric components as zero-copy views. These
    # views let NumExpr avoid field-sized temporary arrays.
    tensor_xx = full_tensor[..., 0, 0]
    tensor_xy = full_tensor[..., 0, 1]
    tensor_xz = full_tensor[..., 0, 2]
    tensor_yy = full_tensor[..., 1, 1]
    tensor_yz = full_tensor[..., 1, 2]
    tensor_zz = full_tensor[..., 2, 2]

    tensor_components = {
        "tensor_xx": tensor_xx,
        "tensor_xy": tensor_xy,
        "tensor_xz": tensor_xz,
        "tensor_yy": tensor_yy,
        "tensor_yz": tensor_yz,
        "tensor_zz": tensor_zz,
    }
    # For traceless symmetric Q, p = tr(Q^2) / 2 and r = 2 sqrt(p / 3).
    tensor_quadratic_invariant = ne.evaluate(
        "0.5 * (tensor_xx**2 + tensor_yy**2 + tensor_zz**2 "
        "+ 2 * (tensor_xy**2 + tensor_xz**2 + tensor_yz**2))",
        local_dict=tensor_components,
        optimization="moderate",
    )
    spectral_radius = ne.evaluate(
        "2 * sqrt(tensor_quadratic_invariant / 3.0)",
        local_dict={
            "tensor_quadratic_invariant": tensor_quadratic_invariant,
        },
        optimization="moderate",
    )
    is_near_isotropic = spectral_radius <= isotropic_tolerance

    # Fuse the symmetric 3x3 determinant into one chunked expression instead of
    # invoking a general stacked-matrix determinant routine.
    tensor_determinant = ne.evaluate(
        "tensor_xx * tensor_yy * tensor_zz "
        "+ 2 * tensor_xy * tensor_xz * tensor_yz "
        "- tensor_xx * tensor_yz**2 "
        "- tensor_yy * tensor_xz**2 "
        "- tensor_zz * tensor_xy**2",
        local_dict=tensor_components,
        optimization="moderate",
    )
    # The cubic solution uses cos(3 theta) = 4 det(Q) / r^3. Isotropic points
    # receive a harmless zero argument because their direction is undefined.
    cosine_argument = ne.evaluate(
        "where(is_near_isotropic, 0.0, " "4 * tensor_determinant / spectral_radius**3)",
        local_dict={
            "is_near_isotropic": is_near_isotropic,
            "tensor_determinant": tensor_determinant,
            "spectral_radius": spectral_radius,
        },
        optimization="moderate",
    )
    # Roundoff may move a mathematically valid cosine slightly outside [-1, 1].
    np.clip(cosine_argument, -1.0, 1.0, out=cosine_argument)
    phase_angle = ne.evaluate(
        "arccos(cosine_argument) / 3.0",
        local_dict={"cosine_argument": cosine_argument},
        optimization="moderate",
    )

    # This mask remains empty unless complete biaxial output requests the
    # canonical positive-S uniaxial eigensystem described below.
    is_uniaxial = np.zeros(full_tensor.shape[:-2], dtype=bool)

    if is_biaxial:
        # Evaluate all three analytic roots and sort them from largest to
        # smallest so every eigenvector column has a stable meaning.
        root_offsets = np.array(
            [0.0, 2 * np.pi / 3, 4 * np.pi / 3],
            dtype=full_tensor.dtype,
        )
        eigenvalues = spectral_radius[..., None] * np.cos(
            phase_angle[..., None] + root_offsets
        )
        descending_order = np.argsort(eigenvalues, axis=-1)[..., ::-1]
        eigenvalues = np.take_along_axis(eigenvalues, descending_order, axis=-1)

        # Repeated roots of the analytic cubic lose roughly square-root
        # precision. Treat a sufficiently close lower pair as the positive-S
        # uniaxial limit and later replace it with the exact degenerate pair.
        lower_eigenvalue_gap = np.abs(eigenvalues[..., 1] - eigenvalues[..., 2])
        uniaxial_tolerance = np.maximum(
            32 * np.sqrt(machine_epsilon) * spectral_radius,
            64 * machine_epsilon * tensor_scale,
        )
        is_uniaxial = (~is_near_isotropic) & (
            lower_eigenvalue_gap <= uniaxial_tolerance
        )

        # Apply the cofactor/null-space formula to every eigenvalue at once.
        analytic_eigenvectors = np.stack(
            [
                full_tensor[..., 0, 2, None]
                * (full_tensor[..., 1, 1, None] - eigenvalues)
                - full_tensor[..., 0, 1, None] * full_tensor[..., 1, 2, None],
                full_tensor[..., 1, 2, None]
                * (full_tensor[..., 0, 0, None] - eigenvalues)
                - full_tensor[..., 0, 1, None] * full_tensor[..., 0, 2, None],
                full_tensor[..., 0, 1, None] ** 2
                - (full_tensor[..., 0, 0, None] - eigenvalues)
                * (full_tensor[..., 1, 1, None] - eigenvalues),
            ],
            axis=-2,
        )
        analytic_eigenvector_norms = np.linalg.norm(analytic_eigenvectors, axis=-2)
        analytic_eigenvector_tolerance = (
            32 * machine_epsilon * np.maximum(1.0, tensor_scale**2)
        )
        eigenvalue_tolerance = 64 * machine_epsilon * np.maximum(1.0, spectral_radius)
        eigenvalue_gaps = np.abs(np.diff(eigenvalues, axis=-1))
        is_principal_eigenvector_stable = (~is_near_isotropic) & (
            analytic_eigenvector_norms[..., 0] > analytic_eigenvector_tolerance
        )
        is_distinct_biaxial = (~is_near_isotropic) & (~is_uniaxial)
        # Only a genuinely biaxial point needs three independently stable
        # analytic eigenvectors. A uniaxial point needs only its unique director.
        is_complete_eigensystem_stable = (
            is_distinct_biaxial
            & np.all(
                analytic_eigenvector_norms > analytic_eigenvector_tolerance[..., None],
                axis=-1,
            )
            & np.all(
                eigenvalue_gaps > eigenvalue_tolerance[..., None],
                axis=-1,
            )
        )

        # Identity is the deterministic isotropic frame and initialized storage
        # for points that will be overwritten below.
        eigenvectors = np.broadcast_to(
            np.eye(3, dtype=full_tensor.dtype),
            full_tensor.shape,
        ).copy()
        if np.any(is_complete_eigensystem_stable):
            normalized_eigenvectors = (
                analytic_eigenvectors / analytic_eigenvector_norms[..., None, :]
            )
            eigenvectors[is_complete_eigensystem_stable] = normalized_eigenvectors[
                is_complete_eigensystem_stable
            ]

        is_analytic_uniaxial = is_uniaxial & is_principal_eigenvector_stable
        if np.any(is_analytic_uniaxial):
            eigenvectors[is_analytic_uniaxial, :, 0] = (
                analytic_eigenvectors[is_analytic_uniaxial, :, 0]
                / analytic_eigenvector_norms[is_analytic_uniaxial, 0, None]
            )

        is_eigensystem_fallback = (
            is_distinct_biaxial & (~is_complete_eigensystem_stable)
        ) | (is_uniaxial & (~is_principal_eigenvector_stable))
        # Only exceptional points reach the general eigensolver; the full field
        # is never passed to np.linalg.eigh.
        if np.any(is_eigensystem_fallback):
            fallback_tensor = full_tensor[is_eigensystem_fallback]
            fallback_eigenvalues, fallback_eigenvectors = np.linalg.eigh(
                fallback_tensor
            )
            fallback_order = np.argsort(fallback_eigenvalues, axis=-1)[..., ::-1]
            eigenvalues[is_eigensystem_fallback] = np.take_along_axis(
                fallback_eigenvalues,
                fallback_order,
                axis=-1,
            )
            eigenvectors[is_eigensystem_fallback] = np.take_along_axis(
                fallback_eigenvectors,
                fallback_order[..., None, :],
                axis=-1,
            )

        if np.any(is_uniaxial):
            # Canonicalize the positive-S uniaxial spectrum exactly. This also
            # prevents a spurious nonzero biaxial order from cubic-root noise.
            eigenvalues[is_uniaxial, 1] = -0.5 * eigenvalues[is_uniaxial, 0]
            eigenvalues[is_uniaxial, 2] = -0.5 * eigenvalues[is_uniaxial, 0]

            # The two degenerate eigenvectors are not physically unique. Choose
            # a deterministic right-handed orthonormal complement by crossing
            # the director with the coordinate axis least parallel to it.
            uniaxial_directors = eigenvectors[is_uniaxial, :, 0]
            reference_axis_indices = np.argmin(
                np.abs(uniaxial_directors),
                axis=-1,
            )
            reference_axes = np.eye(3, dtype=full_tensor.dtype)[reference_axis_indices]
            secondary_axes = np.cross(uniaxial_directors, reference_axes)
            secondary_axes /= np.linalg.norm(secondary_axes, axis=-1)[..., None]
            tertiary_axes = np.cross(uniaxial_directors, secondary_axes)
            eigenvectors[is_uniaxial, :, 1] = secondary_axes
            eigenvectors[is_uniaxial, :, 2] = tertiary_axes

        if np.any(is_near_isotropic):
            eigenvalues[is_near_isotropic] = 0.0
            eigenvectors[is_near_isotropic] = np.eye(3, dtype=full_tensor.dtype)

        # Repository convention: S = 3 lambda_0 / 2 and
        # b = 3 |lambda_1 - lambda_2| / 2.
        scalar_order = 1.5 * eigenvalues[..., 0]
        director = eigenvectors[..., :, 0]
        biaxial_order = 1.5 * np.abs(eigenvalues[..., 1] - eigenvalues[..., 2])
        biaxial_order = np.where(is_uniaxial, 0.0, biaxial_order)

        fallback_count = int(np.count_nonzero(is_eigensystem_fallback))
        if fallback_count:
            logger.warning(
                "q_diagonalize detected "
                f"{fallback_count} grid point(s) where the complete analytic "
                "eigensystem became degenerate or numerically unstable. "
                "Recomputed only those points with np.linalg.eigh."
            )
    else:
        # The common uniaxial path computes only the dominant eigenpair and does
        # not allocate complete eigenvalue or eigenvector arrays.
        logger.debug("Computing the largest eigenvalue from tensor invariants.")
        largest_eigenvalue = ne.evaluate(
            "where(is_near_isotropic, 0.0, " "spectral_radius * cos(phase_angle))",
            local_dict={
                "is_near_isotropic": is_near_isotropic,
                "spectral_radius": spectral_radius,
                "phase_angle": phase_angle,
            },
            optimization="moderate",
        )

        logger.debug("Computing the director associated with the largest eigenvalue.")
        # Construct the cofactor vector spanning the null space of
        # Q - lambda_max I, writing each component directly into its output.
        director_expression_inputs = {
            **tensor_components,
            "largest_eigenvalue": largest_eigenvalue,
        }
        analytic_director = np.empty(
            (3,) + largest_eigenvalue.shape,
            dtype=full_tensor.dtype,
        )
        director_numerator_expressions = (
            "tensor_xz * (tensor_yy - largest_eigenvalue) " "- tensor_xy * tensor_yz",
            "tensor_yz * (tensor_xx - largest_eigenvalue) " "- tensor_xy * tensor_xz",
            "tensor_xy**2 "
            "- (tensor_xx - largest_eigenvalue) "
            "* (tensor_yy - largest_eigenvalue)",
        )
        for component_index, expression in enumerate(director_numerator_expressions):
            # NumExpr cannot accept a NumPy scalar as `out`; a single Q tensor
            # therefore uses return-and-assign while lattice arrays write in place.
            if largest_eigenvalue.ndim:
                ne.evaluate(
                    expression,
                    local_dict=director_expression_inputs,
                    out=analytic_director[component_index],
                    optimization="moderate",
                )
            else:
                analytic_director[component_index] = ne.evaluate(
                    expression,
                    local_dict=director_expression_inputs,
                    optimization="moderate",
                )
        analytic_director_norm = ne.evaluate(
            "sqrt(director_x**2 + director_y**2 + director_z**2)",
            local_dict={
                "director_x": analytic_director[0],
                "director_y": analytic_director[1],
                "director_z": analytic_director[2],
            },
            optimization="moderate",
        )
        analytic_director_tolerance = (
            32 * machine_epsilon * np.maximum(1.0, tensor_scale**2)
        )
        # Reject vectors too small to normalize reliably, including the known
        # x-aligned degeneracy of this particular cofactor formula.
        is_analytic_director_stable = (~is_near_isotropic) & (
            analytic_director_norm > analytic_director_tolerance
        )

        director = np.empty(full_tensor.shape[:-1], dtype=full_tensor.dtype)
        normalization_inputs = {
            "is_analytic_director_stable": is_analytic_director_stable,
            "analytic_director_norm": analytic_director_norm,
        }
        # Normalize stable vectors and assign [1, 0, 0] elsewhere until any
        # non-isotropic fallback points are replaced below.
        for component_index in range(3):
            default_component = 1.0 if component_index == 0 else 0.0
            normalization_expression = (
                "where(is_analytic_director_stable, "
                "director_component / analytic_director_norm, "
                "default_component)"
            )
            expression_inputs = {
                **normalization_inputs,
                "director_component": analytic_director[component_index],
                "default_component": default_component,
            }
            if largest_eigenvalue.ndim:
                ne.evaluate(
                    normalization_expression,
                    local_dict=expression_inputs,
                    out=director[..., component_index],
                    optimization="moderate",
                )
            else:
                director[..., component_index] = ne.evaluate(
                    normalization_expression,
                    local_dict=expression_inputs,
                    optimization="moderate",
                )

        is_director_fallback = (~is_near_isotropic) & (~is_analytic_director_stable)
        # Recompute only degenerate non-isotropic points with the robust solver.
        if np.any(is_director_fallback):
            fallback_tensor = full_tensor[is_director_fallback]
            fallback_eigenvalues, fallback_eigenvectors = np.linalg.eigh(
                fallback_tensor
            )
            largest_eigenvalue[is_director_fallback] = fallback_eigenvalues[..., -1]
            director[is_director_fallback] = fallback_eigenvectors[..., :, -1]

        scalar_order = 1.5 * largest_eigenvalue
        eigenvalues = None
        eigenvectors = None
        biaxial_order = None

        fallback_count = int(np.count_nonzero(is_director_fallback))
        if fallback_count:
            logger.warning(
                "q_diagonalize detected "
                f"{fallback_count} grid point(s) where the analytic director "
                "formula became degenerate or numerically unstable. Recomputed "
                "the dominant eigenpair with np.linalg.eigh at those points."
            )

    # Emit recovery diagnostics only after all requested arrays are finalized.
    isotropic_count = int(np.count_nonzero(is_near_isotropic))
    if isotropic_count:
        logger.warning(
            "q_diagonalize detected "
            f"{isotropic_count} near-isotropic grid point(s). Set S "
            "to 0 and assigned the default director [1, 0, 0] at those points."
        )

    isotropic_indices = np.argwhere(is_near_isotropic)
    uniaxial_indices = np.argwhere(is_uniaxial)

    return QDiagonalizationResult(
        S=scalar_order,
        n=director,
        isotropic_indices=isotropic_indices,
        uniaxial_indices=uniaxial_indices,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        biaxial_order=biaxial_order,
    )
