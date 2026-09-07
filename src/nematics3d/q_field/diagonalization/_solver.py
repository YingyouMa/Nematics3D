"""Analytic Q-tensor diagonalization and named result objects."""

import time
from dataclasses import dataclass
from numbers import Integral
from typing import ClassVar, Union

import numexpr as ne
import numpy as np

from ...core.result_base import ResultBase
from ...datatypes import QField5, QField9, as_bool, as_qfield5, as_qfield9
from ...logging_decorator import logging_and_warning_decorator
from ._backend import (
    _resolve_worker_count,
    diagonalize_qfield5,
    is_c_backend_available,
)


@dataclass(slots=True, frozen=True, repr=False)
class QDiagonalizationResult(ResultBase):
    """Named outputs from :func:`q_diagonalize`."""

    __result_name__: ClassVar[str] = "Q-tensor diagonalization"

    # fmt: off
    __field_docs__: ClassVar[dict[str, str]] = {
        "S":               "Scalar order: 3/2 times the largest eigenvalue.",
        "n":               "Unit eigenvector for the largest eigenvalue.",
        "isotropic_indices": (
            "Index tuples of points handled as numerically isotropic."
        ),
        "eigenvalues":     "Descending eigenspectrum when biaxial output is requested.",
        "eigenvectors":    "Eigenvector columns matching the descending eigenvalues.",
    }
    # fmt: on

    S: np.ndarray  # noqa: N815 - conventional public symbol for scalar order
    n: np.ndarray
    isotropic_indices: list[tuple[int, ...]]
    eigenvalues: np.ndarray | None = None
    eigenvectors: np.ndarray | None = None


def _dominant_eigenpair_q_sd(qxx, qyy, qxy, qxz, qyz):
    """Return only the largest eigenvalue and its unit eigenvector."""
    arrays = np.broadcast_arrays(qxx, qyy, qxy, qxz, qyz)
    shape = arrays[0].shape
    a, d, b, c, e = (
        np.ascontiguousarray(value, dtype=np.float64).reshape(-1) for value in arrays
    )
    tensor_count = a.size

    is_zero = ne.evaluate("(a == 0) & (d == 0) & (b == 0) & (c == 0) & (e == 0)")
    p = ne.evaluate("sqrt((a*a + d*d + a*d + b*b + c*c + e*e) / 3)")
    safe_p = np.where(is_zero, 1.0, p)  # noqa: F841 - used by NumExpr
    determinant = ne.evaluate(  # noqa: F841 - used by NumExpr
        "-a*d*(a+d) + 2*b*c*e - a*e*e - d*c*c + (a+d)*b*b"
    )
    cosine = ne.evaluate("where(is_zero, 0, 0.5*determinant/safe_p**3)")
    np.clip(cosine, -1.0, 1.0, out=cosine)
    phase = ne.evaluate("arccos(cosine) / 3")  # noqa: F841 - used by NumExpr
    largest_value = ne.evaluate("where(is_zero, 0, 2*p*cos(phase))")
    del cosine, determinant, p, phase, safe_p

    r00 = ne.evaluate("a-largest_value")  # noqa: F841 - used by NumExpr
    r11 = ne.evaluate("d-largest_value")  # noqa: F841 - used by NumExpr
    r22 = ne.evaluate("-a-d-largest_value")  # noqa: F841 - used by NumExpr
    # Stream the three adjugate rows through one best-candidate buffer. This
    # preserves strongest-row selection without retaining all six cofactors.
    director = np.empty((3, tensor_count), dtype=np.float64)
    ne.evaluate("r11*r22-e*e", out=director[0])
    ne.evaluate("c*e-b*r22", out=director[1])
    ne.evaluate("b*e-c*r11", out=director[2])
    best_norm = ne.evaluate(
        "x*x+y*y+z*z", local_dict={"x": director[0], "y": director[1], "z": director[2]}
    )

    candidate = np.empty_like(director)
    ne.evaluate("c*e-b*r22", out=candidate[0])
    ne.evaluate("r00*r22-c*c", out=candidate[1])
    ne.evaluate("b*c-r00*e", out=candidate[2])
    candidate_norm = ne.evaluate(
        "x*x+y*y+z*z",
        local_dict={"x": candidate[0], "y": candidate[1], "z": candidate[2]},
    )
    is_stronger = candidate_norm > best_norm
    for component in range(3):
        np.copyto(director[component], candidate[component], where=is_stronger)
    np.copyto(best_norm, candidate_norm, where=is_stronger)

    ne.evaluate("b*e-c*r11", out=candidate[0])
    ne.evaluate("b*c-r00*e", out=candidate[1])
    ne.evaluate("r00*r11-b*b", out=candidate[2])
    ne.evaluate(
        "x*x+y*y+z*z",
        local_dict={
            "x": candidate[0],
            "y": candidate[1],
            "z": candidate[2],
        },
        out=candidate_norm,
    )
    is_stronger = candidate_norm > best_norm
    for component in range(3):
        np.copyto(director[component], candidate[component], where=is_stronger)
    del a, arrays, b, best_norm, c, candidate, candidate_norm, d, e, is_stronger
    del r00, r11, r22

    director_norm = ne.evaluate(
        "sqrt(vx*vx+vy*vy+vz*vz)",
        local_dict={"vx": director[0], "vy": director[1], "vz": director[2]},
    )
    is_bad_vector = director_norm == 0.0
    director_norm[is_bad_vector] = 1.0
    director /= director_norm
    director[:, is_bad_vector] = np.array([[1.0], [0.0], [0.0]])
    director[:, is_zero] = np.array([[1.0], [0.0], [0.0]])
    return largest_value.reshape(shape), np.moveaxis(director, 0, -1).reshape(
        shape + (3,)
    )


def _eigh3_q_sd(qxx, qyy, qxy, qxz, qyz):
    """Solve symmetric traceless 3x3 eigensystems in ascending order."""
    arrays = np.broadcast_arrays(qxx, qyy, qxy, qxz, qyz)
    shape = arrays[0].shape
    a, d, b, c, e = (
        np.ascontiguousarray(value, dtype=np.float64).reshape(-1) for value in arrays
    )
    tensor_count = a.size

    is_zero = ne.evaluate("(a == 0) & (d == 0) & (b == 0) & (c == 0) & (e == 0)")
    p = ne.evaluate("sqrt((a*a + d*d + a*d + b*b + c*c + e*e) / 3)")
    safe_p = np.where(is_zero, 1.0, p)  # noqa: F841 - used by NumExpr
    determinant = ne.evaluate(  # noqa: F841 - used by NumExpr
        "-a*d*(a+d) + 2*b*c*e - a*e*e - d*c*c + (a+d)*b*b"
    )
    cosine = ne.evaluate("where(is_zero, 0, 0.5*determinant/safe_p**3)")
    np.clip(cosine, -1.0, 1.0, out=cosine)
    phase = ne.evaluate("arccos(cosine) / 3")
    is_upper_isolated = cosine >= 0.0
    isolated_value = ne.evaluate(
        "where(is_upper_isolated, 2*p*cos(phase), "
        "2*p*cos(phase + two_pi_over_three))",
        local_dict={
            "is_upper_isolated": is_upper_isolated,
            "p": p,
            "phase": phase,
            "two_pi_over_three": 2.0 * np.pi / 3.0,
        },
    )
    del cosine, determinant, p, phase, safe_p

    r00 = ne.evaluate("a-isolated_value")  # noqa: F841 - used by NumExpr
    r11 = ne.evaluate("d-isolated_value")  # noqa: F841 - used by NumExpr
    r22 = ne.evaluate("-a-d-isolated_value")  # noqa: F841 - used by NumExpr
    adj00 = ne.evaluate("r11*r22-e*e")  # noqa: F841 - used by NumExpr
    adj11 = ne.evaluate("r00*r22-c*c")  # noqa: F841 - used by NumExpr
    adj22 = ne.evaluate("r00*r11-b*b")  # noqa: F841 - used by NumExpr
    adj01 = ne.evaluate("c*e-b*r22")  # noqa: F841 - used by NumExpr
    adj02 = ne.evaluate("b*e-c*r11")  # noqa: F841 - used by NumExpr
    adj12 = ne.evaluate("b*c-r00*e")  # noqa: F841 - used by NumExpr
    norm0 = ne.evaluate("adj00**2+adj01**2+adj02**2")
    norm1 = ne.evaluate("adj01**2+adj11**2+adj12**2")
    norm2 = ne.evaluate("adj02**2+adj12**2+adj22**2")
    choose0 = (norm0 >= norm1) & (norm0 >= norm2)
    choose1 = (~choose0) & (norm1 >= norm2)  # noqa: F841 - used by NumExpr

    isolated_vector = np.empty((3, tensor_count), dtype=np.float64)
    ne.evaluate(
        "where(choose0,adj00,where(choose1,adj01,adj02))",
        out=isolated_vector[0],
    )
    ne.evaluate(
        "where(choose0,adj01,where(choose1,adj11,adj12))",
        out=isolated_vector[1],
    )
    ne.evaluate(
        "where(choose0,adj02,where(choose1,adj12,adj22))",
        out=isolated_vector[2],
    )
    del (
        adj00,
        adj01,
        adj02,
        adj11,
        adj12,
        adj22,
        choose0,
        choose1,
        norm0,
        norm1,
        norm2,
        r00,
        r11,
        r22,
    )
    vector_norm = ne.evaluate(
        "sqrt(vx*vx+vy*vy+vz*vz)",
        local_dict={
            "vx": isolated_vector[0],
            "vy": isolated_vector[1],
            "vz": isolated_vector[2],
        },
    )
    is_bad_vector = vector_norm == 0.0
    safe_norm = np.where(is_bad_vector, 1.0, vector_norm)
    isolated_vector /= safe_norm
    is_canonical = is_zero | is_bad_vector
    isolated_vector[:, is_canonical] = np.array([[1.0], [0.0], [0.0]])
    del vector_norm
    del is_bad_vector, is_canonical, safe_norm

    # Delay the complete output allocation until the adjugate work arrays have
    # been released; otherwise both large stages overlap at peak memory.
    vectors_soa = np.empty((3, 3, tensor_count), dtype=np.float64)
    vectors_soa[0] = isolated_vector
    del isolated_vector
    isolated_vector = vectors_soa[0]
    vx, vy, vz = isolated_vector

    abs_vector = np.abs(isolated_vector)
    use_x = (abs_vector[0] <= abs_vector[1]) & (abs_vector[0] <= abs_vector[2])
    use_y = (~use_x) & (abs_vector[1] <= abs_vector[2])  # noqa: F841
    projection = ne.evaluate(  # noqa: F841 - used by NumExpr
        "where(use_x,vx,where(use_y,vy,vz))"
    )
    inverse_plane_norm = ne.evaluate(  # noqa: F841 - used by NumExpr
        "1/sqrt(1-projection*projection)"
    )
    plane_u = vectors_soa[1]
    ne.evaluate("(where(use_x,1,0)-projection*vx)*inverse_plane_norm", out=plane_u[0])
    ne.evaluate("(where(use_y,1,0)-projection*vy)*inverse_plane_norm", out=plane_u[1])
    ne.evaluate(
        "(where((~use_x)&(~use_y),1,0)-projection*vz)*inverse_plane_norm",
        out=plane_u[2],
    )
    del abs_vector, inverse_plane_norm, projection, use_x, use_y
    ux, uy, uz = plane_u
    plane_w = vectors_soa[2]
    plane_w[0] = vy * uz - vz * uy
    plane_w[1] = vz * ux - vx * uz
    plane_w[2] = vx * uy - vy * ux
    wx, wy, wz = plane_w

    block_a = ne.evaluate(  # noqa: F841 - used by NumExpr
        "a*ux**2+d*uy**2-(a+d)*uz**2+2*(b*ux*uy+c*ux*uz+e*uy*uz)"
    )
    block_b = ne.evaluate(  # noqa: F841 - used by NumExpr
        "wx*(a*ux+b*uy+c*uz)+wy*(b*ux+d*uy+e*uz)" "+wz*(c*ux+e*uy-(a+d)*uz)"
    )
    del a, arrays, b, c, d, e, wx, wy, wz
    difference = ne.evaluate("2*block_a+isolated_value")
    discriminant = ne.evaluate("sqrt(difference**2+4*block_b**2)")
    lower_value = ne.evaluate("0.5*(-isolated_value-discriminant)")
    upper_value = ne.evaluate("0.5*(-isolated_value+discriminant)")

    is_degenerate = discriminant == 0.0
    is_positive = difference >= 0.0  # noqa: F841 - used by NumExpr
    safe_discriminant = np.where(  # noqa: F841 - used by NumExpr
        is_degenerate, 1.0, discriminant
    )
    denominator = ne.evaluate(  # noqa: F841 - used by NumExpr
        "where(is_positive,safe_discriminant+difference,"
        "safe_discriminant-difference)"
    )
    tangent = ne.evaluate(  # noqa: F841 - used by NumExpr
        "where(is_degenerate,0,2*block_b/denominator)"
    )
    inverse_rotation_norm = ne.evaluate(  # noqa: F841 - used by NumExpr
        "1/sqrt(1+tangent*tangent)"
    )
    cosine_rotation = ne.evaluate(
        "where(is_positive,inverse_rotation_norm,abs(tangent)*inverse_rotation_norm)"
    )
    sine_rotation = ne.evaluate(
        "where(is_positive,tangent*inverse_rotation_norm,"
        "where(block_b>=0,inverse_rotation_norm,-inverse_rotation_norm))"
    )
    cosine_rotation[is_degenerate] = 1.0
    sine_rotation[is_degenerate] = 0.0
    del (
        block_a,
        block_b,
        denominator,
        difference,
        discriminant,
        inverse_rotation_norm,
        is_degenerate,
        is_positive,
        safe_discriminant,
        tangent,
    )

    eigenvalues = np.empty((tensor_count, 3), dtype=np.float64)
    eigenvalues[:, 0] = isolated_value
    np.copyto(eigenvalues[:, 0], lower_value, where=is_upper_isolated)
    eigenvalues[:, 1] = lower_value
    np.copyto(eigenvalues[:, 1], upper_value, where=is_upper_isolated)
    eigenvalues[:, 2] = upper_value
    np.copyto(eigenvalues[:, 2], isolated_value, where=is_upper_isolated)
    for component in range(3):
        # Preserve the upper vector temporarily, then overwrite the plane
        # buffers with the two rotated eigenvectors in place.
        upper_component = ne.evaluate(
            "cosine_rotation*u+sine_rotation*w",
            local_dict={
                "sine_rotation": sine_rotation,
                "cosine_rotation": cosine_rotation,
                "u": plane_u[component],
                "w": plane_w[component],
            },
        )
        ne.evaluate(
            "-sine_rotation*u+cosine_rotation*w",
            local_dict={
                "sine_rotation": sine_rotation,
                "cosine_rotation": cosine_rotation,
                "u": plane_u[component],
                "w": plane_w[component],
            },
            out=plane_u[component],
        )
        plane_w[component] = upper_component
        del upper_component

        # The buffers currently hold [isolated, lower, upper]. Only tensors
        # whose isolated root is the largest need the cyclic ascending reorder.
        isolated_component = isolated_vector[component].copy()
        np.copyto(
            isolated_vector[component],
            plane_u[component],
            where=is_upper_isolated,
        )
        np.copyto(
            plane_u[component],
            plane_w[component],
            where=is_upper_isolated,
        )
        np.copyto(
            plane_w[component],
            isolated_component,
            where=is_upper_isolated,
        )
        del isolated_component
    eigenvectors = np.transpose(vectors_soa, (2, 1, 0))
    eigenvalues[is_zero] = 0.0
    eigenvectors[is_zero] = np.eye(3)
    return eigenvalues.reshape(shape + (3,)), eigenvectors.reshape(shape + (3, 3))


@logging_and_warning_decorator()
def q_diagonalize(
    qtensor: Union[QField5, QField9],
    *,
    is_biaxial: bool = False,
    is_right_handed: bool = False,
    is_use_c_backend: bool | None = None,
    worker_count: int | None = None,
    logger=None,
) -> QDiagonalizationResult:
    """Diagonalize a symmetric traceless Q-tensor field robustly.

    The solver uses an isolated eigenpair followed by a symmetric 2x2 solve in
    its orthogonal plane. This keeps the complete frame orthonormal near a
    repeated eigenvalue.

    Parameters
    ----------
    qtensor : QField5 or QField9
        Q-tensor data with trailing shape ``(..., 5)`` or ``(..., 3, 3)``.
    is_biaxial : bool, optional
        Whether to return the complete descending eigensystem.
    is_right_handed : bool, optional
        Whether complete eigenvector frames must be right-handed. Requires
        ``is_biaxial=True``.
    is_use_c_backend : bool or None, optional
        Backend selection. ``None`` uses the compiled C backend when available
        and otherwise falls back to NumExpr. ``True`` requires the C backend;
        ``False`` forces NumExpr.
    worker_count : int or None, optional
        Positive worker count. The C backend uses a Python thread pool whose C
        loops release the GIL. The NumExpr backend temporarily uses this many
        NumExpr threads. ``None`` selects each backend's automatic behavior.

    Returns
    -------
    QDiagonalizationResult
        Scalar order, dominant director, recovery indices, and optionally the
        complete eigensystem.

    Raises
    ------
    TypeError
        If ``qtensor`` does not have a floating-point dtype, or a backend
        option has the wrong type.
    ValueError
        If the input is empty or invalid, or a right-handed frame is requested
        without complete biaxial output, or ``worker_count`` is invalid.
    ImportError
        If the compiled backend is required but unavailable.
    """
    input_tensor = np.asarray(qtensor)
    is_biaxial = as_bool(is_biaxial, name="is_biaxial")
    is_right_handed = as_bool(is_right_handed, name="is_right_handed")
    if is_use_c_backend is not None:
        is_use_c_backend = as_bool(
            is_use_c_backend,
            name="is_use_c_backend",
        )

    logger.debug(
        f"Received Q-tensor input: shape={input_tensor.shape}, "
        f"dtype={input_tensor.dtype}.\n"
        "Options: "
        f"is_biaxial={is_biaxial}, "
        f"is_right_handed={is_right_handed}, "
        f"is_use_c_backend={is_use_c_backend}, "
        f"worker_count={worker_count}."
    )

    if is_right_handed and not is_biaxial:
        raise ValueError("'is_right_handed=True' requires 'is_biaxial=True'.")

    if worker_count is not None:
        if isinstance(worker_count, bool) or not isinstance(worker_count, Integral):
            raise TypeError("'worker_count' must be a positive integer or None.")
        worker_count = int(worker_count)
        if worker_count < 1:
            raise ValueError("'worker_count' must be a positive integer or None.")

    is_c_available = is_c_backend_available()
    if is_use_c_backend is True and not is_c_available:
        raise ImportError(
            "'is_use_c_backend=True' requires the compiled Nematics3D "
            "Q-diagonalization extension, but it could not be imported."
        )
    is_using_c = is_c_available if is_use_c_backend is None else is_use_c_backend
    if not is_using_c and worker_count is not None and worker_count > ne.MAX_THREADS:
        raise ValueError(
            f"'worker_count' cannot exceed the NumExpr limit of {ne.MAX_THREADS} "
            "when the Python backend is selected."
        )

    logger.debug("Preparing and validating Q-tensor input.")
    stage_start = time.perf_counter()
    if input_tensor.ndim >= 1 and input_tensor.shape[-1] == 5:
        input_representation = "compact five-component"
        compact_tensor = as_qfield5(
            input_tensor,
            name="Q tensor to diagonalize",
            is_strict_3d_field=False,
        )
    else:
        input_representation = "full 3x3"
        full_tensor = as_qfield9(
            input_tensor,
            name="Q tensor to diagonalize",
            is_strict_3d_field=False,
        )
        compact_tensor = as_qfield5(
            full_tensor,
            name="Q tensor to diagonalize",
            is_strict_3d_field=False,
            is_validate_tensor=False,
        )
        del full_tensor
    if compact_tensor.size == 0:
        raise ValueError(
            "'qtensor' must contain at least one Q tensor; "
            f"got shape {input_tensor.shape}."
        )
    tensor_count = compact_tensor.size // 5
    logger.debug(
        f"Prepared {tensor_count:,} Q tensor(s) from {input_representation} "
        f"input in {time.perf_counter() - stage_start:.3f} seconds."
    )

    if is_using_c:
        actual_worker_count = _resolve_worker_count(worker_count, tensor_count)
        backend_label = "compiled C"
        worker_label = f"{actual_worker_count} worker(s)"
    else:
        actual_worker_count = (
            ne.get_num_threads() if worker_count is None else worker_count
        )
        backend_label = "NumExpr"
        worker_label = f"{actual_worker_count} thread(s)"
    calculation_label = "complete-eigensystem" if is_biaxial else "principal-eigenpair"
    logger.debug(
        f"Selected {backend_label} {calculation_label} solver with {worker_label}."
    )

    logger.debug(f"Classifying {tensor_count:,} Q tensor(s) for near isotropy.")
    stage_start = time.perf_counter()
    calculation_dtype = np.result_type(compact_tensor.dtype, np.float64)
    qxx = compact_tensor[..., 0]
    qyy = compact_tensor[..., 3]
    tensor_abs_max = np.maximum(
        np.max(np.abs(compact_tensor), axis=-1),
        np.abs(qxx + qyy),
    )
    isotropic_tolerance = (
        32 * np.finfo(calculation_dtype).eps * np.maximum(1.0, tensor_abs_max)
    )
    is_isotropic = tensor_abs_max <= isotropic_tolerance
    del isotropic_tolerance, tensor_abs_max
    isotropic_count = int(np.count_nonzero(is_isotropic))
    logger.debug(
        f"Classified {tensor_count:,} Q tensor(s) in "
        f"{time.perf_counter() - stage_start:.3f} seconds: "
        f"{isotropic_count:,} near-isotropic and "
        f"{tensor_count - isotropic_count:,} non-isotropic."
    )

    logger.debug(
        f"Computing {tensor_count:,} {calculation_label} result(s) with the "
        f"{backend_label} solver."
    )
    stage_start = time.perf_counter()
    tensor_components = (
        compact_tensor[..., 0],
        compact_tensor[..., 3],
        compact_tensor[..., 1],
        compact_tensor[..., 2],
        compact_tensor[..., 4],
    )
    if is_using_c:
        backend_values, backend_vectors, actual_worker_count = diagonalize_qfield5(
            compact_tensor,
            is_biaxial=is_biaxial,
            worker_count=worker_count,
        )
    else:
        previous_numexpr_threads = ne.get_num_threads()
        if worker_count is not None:
            ne.set_num_threads(worker_count)
        try:
            if is_biaxial:
                backend_values, backend_vectors = _eigh3_q_sd(*tensor_components)
            else:
                backend_values, backend_vectors = _dominant_eigenpair_q_sd(
                    *tensor_components
                )
        finally:
            if worker_count is not None:
                ne.set_num_threads(previous_numexpr_threads)
    logger.debug(
        f"Computed {tensor_count:,} {calculation_label} result(s) in "
        f"{time.perf_counter() - stage_start:.3f} seconds."
    )

    if is_biaxial:
        logger.debug("Ordering and normalizing complete eigensystem outputs.")
        stage_start = time.perf_counter()
        ascending_values, ascending_vectors = backend_values, backend_vectors
        ascending_values[is_isotropic] = 0.0
        ascending_vectors[is_isotropic] = np.eye(3)
        descending_values = ascending_values[..., ::-1]
        descending_vectors = ascending_vectors[..., :, ::-1]
        descending_vectors[is_isotropic] = np.eye(3)
        logger.debug(
            "Ordered and normalized complete eigensystem outputs in "
            f"{time.perf_counter() - stage_start:.3f} seconds."
        )

    if is_biaxial and is_right_handed:
        logger.debug(
            f"Converting {tensor_count:,} complete eigenvector frame(s) to "
            "right-handed orientation."
        )
        stage_start = time.perf_counter()
        is_left_handed = np.linalg.det(descending_vectors) < 0.0
        left_handed_count = int(np.count_nonzero(is_left_handed))
        descending_vectors[..., :, -1] = np.where(
            is_left_handed[..., None],
            -descending_vectors[..., :, -1],
            descending_vectors[..., :, -1],
        )
        logger.debug(
            f"Converted {tensor_count:,} eigenvector frame(s) in "
            f"{time.perf_counter() - stage_start:.3f} seconds: "
            f"{left_handed_count:,} frame(s) required an orientation flip."
        )

    logger.debug("Finalizing diagonalization results.")
    stage_start = time.perf_counter()
    if is_biaxial:
        largest_value = descending_values[..., 0]
        director = descending_vectors[..., :, 0]
    else:
        largest_value, director = backend_values, backend_vectors
        largest_value[is_isotropic] = 0.0
        director[is_isotropic] = np.array([1.0, 0.0, 0.0])
    if is_biaxial:
        scalar_order = 1.5 * largest_value
    else:
        largest_value *= 1.5
        scalar_order = largest_value
    isotropic_indices = [
        tuple(int(coordinate) for coordinate in index)
        for index in np.argwhere(is_isotropic)
    ]
    if is_biaxial:
        eigenvalues = descending_values
        eigenvectors = descending_vectors
    else:
        eigenvalues = None
        eigenvectors = None

    result = QDiagonalizationResult(
        S=scalar_order,
        n=director,
        isotropic_indices=isotropic_indices,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
    )
    logger.debug(
        f"Finalized diagonalization results in "
        f"{time.perf_counter() - stage_start:.3f} seconds."
    )

    if isotropic_count:
        logger.warning(
            f"{isotropic_count} near-isotropic grid point(s). Set S to 0 and "
            "assigned the default director [1, 0, 0] at those points. Inspect "
            "result.isotropic_indices before interpreting the director."
        )
    return result
