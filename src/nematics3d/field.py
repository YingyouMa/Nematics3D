"""
Field-level utilities for structured Q-tensor and director data.
"""

import time
from typing import Tuple, Union, List

import numpy as np

# from .general import *
from .datatypes import (
    QField5,
    QField9,
    as_qfield9,
    nField,
    SField,
    check_Sn,
    GeneralField,
    DimensionFlagInput,
    as_dimension_info,
)
from .logging_decorator import logging_and_warning_decorator


@logging_and_warning_decorator()
def Q_diagonalize(
    qtensor: Union[QField5, QField9], logger=None
) -> Tuple[SField, nField]:
    """
    Analytically diagonalize a Q-tensor field to obtain the scalar order parameter (S)
    and the director field (n).

    This implementation uses tensor invariants to compute the largest eigenvalue
    and corresponding eigenvector without calling `np.linalg.eigh` on each grid point,
    which is significantly faster for large 3D fields.

    Parameters
    ----------
    qtensor : QField5 or QField9
        The Q-tensor field to be diagonalized.
        - QField5: shape (..., 5), 5 independent components
        - QField9: shape (..., 3, 3), full symmetric traceless tensor

    logger : logging.Logger, optional
        Logger instance for diagnostic messages.

    Returns
    -------
    S : SField
        Scalar order parameter, shape (...,). Defined as 1.5 × λ_max.

    n : nField
        Director field (unit vector), shape (..., 3).

    Notes
    -----
    - The sign of `n` is not unique: (n, -n) are equivalent.
    - Future work may extend this to biaxial order and negative S cases.

    Raises
    ------
    TypeError
        If `qtensor` is not a float-type NumPy array.

    ValueError
        If `qtensor` shape is not a valid QField5 or QField9.
    """
    Q: QField9 = as_qfield9(qtensor, is_strict_3d_field=False)
    float_dtype = np.result_type(Q.dtype, np.float64)
    Q = np.asarray(Q, dtype=float_dtype)
    eps = np.finfo(Q.dtype).eps
    q_abs_max = np.max(np.abs(Q), axis=(-2, -1))
    q_scale = np.maximum(1.0, q_abs_max)

    # Compute tensor invariants
    logger.debug("Computing tensor invariants (p, q, r).")
    start = time.time()
    p = 0.5 * np.einsum("...ab, ...ba -> ...", Q, Q)
    q = np.linalg.det(Q)
    r = 2 * np.sqrt(p / 3)
    logger.debug(f"Tensor invariants computed in {time.time() - start:.2f} seconds.")

    # Largest eigenvalue λ (before scaling)
    logger.debug("Computing largest eigenvalue λ_max.")
    start = time.time()
    isotropic_tol = 32 * eps * q_scale
    is_near_isotropic = r <= isotropic_tol
    cos_arg = np.zeros_like(r)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(4 * q, r**3, out=cos_arg, where=~is_near_isotropic)
    cos_arg = np.clip(cos_arg, -1.0, 1.0)  # ensure valid domain
    lambda_max = np.zeros_like(r)
    lambda_max[~is_near_isotropic] = r[~is_near_isotropic] * np.cos(
        (1 / 3) * np.arccos(cos_arg[~is_near_isotropic])
    )
    logger.debug(f"λ_max computed in {time.time() - start:.2f} seconds.")

    # Director corresponding to λ_max
    logger.debug("Computing director field n.")
    start = time.time()
    n_raw = np.array(
        [
            Q[..., 0, 2] * (Q[..., 1, 1] - lambda_max) - Q[..., 0, 1] * Q[..., 1, 2],
            Q[..., 1, 2] * (Q[..., 0, 0] - lambda_max) - Q[..., 0, 1] * Q[..., 0, 2],
            Q[..., 0, 1] ** 2
            - (Q[..., 0, 0] - lambda_max) * (Q[..., 1, 1] - lambda_max),
        ]
    )
    n = np.zeros(Q.shape[:-1], dtype=Q.dtype)
    n[..., 0] = 1.0

    n_raw_norm = np.linalg.norm(n_raw, axis=0)
    n_raw_tol = 32 * eps * np.maximum(1.0, q_scale**2)
    is_fast_director_ok = (~is_near_isotropic) & (n_raw_norm > n_raw_tol)

    if np.any(is_fast_director_ok):
        n_fast = n_raw[:, is_fast_director_ok] / n_raw_norm[is_fast_director_ok]
        n[is_fast_director_ok] = np.moveaxis(n_fast, 0, -1)

    is_director_fallback = (~is_near_isotropic) & (~is_fast_director_ok)
    if np.any(is_director_fallback):
        q_fallback = Q[is_director_fallback]
        evals, evecs = np.linalg.eigh(q_fallback)
        lambda_max[is_director_fallback] = evals[..., -1]
        n[is_director_fallback] = evecs[..., :, -1]

    n: nField = n  # (..., 3)
    logger.debug(f"Director field computed in {time.time() - start:.2f} seconds.")

    # Scale eigenvalue to get S
    S: SField = 1.5 * lambda_max

    isotropic_count = int(np.count_nonzero(is_near_isotropic))
    if isotropic_count:
        logger.warning(
            "Q_diagonalize detected "
            f"{isotropic_count} near-isotropic grid point(s) where the tensor "
            "magnitude was too small for stable invariant-based diagonalization. "
            "Set S = 0 and assigned the default director [1, 0, 0] at those points."
        )

    fallback_count = int(np.count_nonzero(is_director_fallback))
    if fallback_count:
        logger.warning(
            "Q_diagonalize detected "
            f"{fallback_count} grid point(s) where the analytic director formula "
            "became degenerate or numerically unstable. Recomputed the dominant "
            "eigenpair with np.linalg.eigh at those points."
        )

    return S, n


def Q_diagonalize_linalg(
    qtensor: Union[QField5, QField9],
    *,
    is_right_handed: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize Q tensors with ``np.linalg.eigh``.

    This helper keeps the full eigensystem instead of only returning the
    dominant director.  Eigenvalues are sorted descending, and eigenvectors are
    returned as columns in the matching order.  The input may be any valid
    ``(..., 5)`` or ``(..., 3, 3)`` Q representation accepted by
    ``as_qfield9(..., is_strict_3d_field=False)``.

    Set ``is_right_handed=True`` when the returned eigenvector frames should
    satisfy ``axis0 x axis1 = axis2``.  The default is ``False`` because this
    extra convention requires a determinant calculation and is not needed for
    every caller.
    """

    Q = as_qfield9(
        qtensor,
        name="Q tensor to diagonalize",
        is_strict_3d_field=False,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    order = np.argsort(eigenvalues, axis=-1)[..., ::-1]

    eigenvalues = np.take_along_axis(eigenvalues, order, axis=-1)
    eigenvectors = np.take_along_axis(eigenvectors, order[..., None, :], axis=-1)

    if is_right_handed:
        is_left_handed = np.linalg.det(eigenvectors) < 0
        eigenvectors[..., :, -1] = np.where(
            is_left_handed[..., None],
            -eigenvectors[..., :, -1],
            eigenvectors[..., :, -1],
        )

    return eigenvalues, eigenvectors


@logging_and_warning_decorator()
def getQ(n: nField, S: SField = None, logger=None) -> QField9:
    #! biaxial
    """
    Compute the Q-tensor field from a given director field and optional scalar order parameter.

    This function constructs a symmetric, traceless, uniaxial Q-tensor of the form:
        Q_ij = S * (n_i n_j - δ_ij / 3)

    If `S` is not provided, the tensor is computed assuming S = 1.

    Parameters
    ----------
    n : nField
        Director field of shape (..., 3).

    S : SField, optional
        Scalar order parameter field of shape (...,). If provided, scales the Q-tensor accordingly.

    Returns
    -------
    Q : QField9
        The computed Q-tensor field of shape (..., 3, 3), symmetric and traceless.
    """

    n = check_Sn(n, "n", is_3d_strict=False)

    Q = np.einsum("...i, ...j -> ...ij", n, n) - np.eye(3) / 3
    if S is not None:
        S = check_Sn(S, "S", is_3d_strict=False)
        Q = np.einsum("..., ...ij -> ...ij", S, Q)
    else:
        logger.warning(">>> No S input. Set to be 1.")

    return Q


def add_periodic_boundary(
    data: GeneralField, is_boundary_periodic: DimensionFlagInput = 0
) -> GeneralField:
    #! loop
    """
    Extend a physical field with periodic boundary slices in specified dimensions.

    This function appends one extra grid slice along each of the periodic dimensions.
    The added slice is a copy of the first slice along that axis, ensuring periodic continuity.
    If a dimension is non-periodic, it is left unchanged.

    Parameters
    ----------
    data : GeneralField
        Input physical field of shape (Nx, Ny, Nz, ...), where (Nx, Ny, Nz) are spatial dimensions,
        and the remaining axes represent vector/tensor components or other per-voxel data.

    is_boundary_periodic : DimensionFlagInput, optional
        A 3-element flag indicating which spatial dimensions are periodic.
        - Can be a scalar (broadcasted), or
        - A list/tuple/array of booleans with shape (3,)
        - Default is 0 (all dimensions non-periodic)

    Returns
    -------
    output : GeneralField
        Extended field with one additional slice along each periodic dimension.
        Shape becomes:
            (Nx + is_periodic[0], Ny + is_periodic[1], Nz + is_periodic[2], ...)
    """
    is_boundary_periodic = as_dimension_info(is_boundary_periodic)

    if np.any(is_boundary_periodic):
        Nx, Ny, Nz, *rest_shape = data.shape  # Extract the first three dimensions
        output = np.empty(
            (
                Nx + is_boundary_periodic[0],
                Ny + is_boundary_periodic[1],
                Nz + is_boundary_periodic[2],
                *rest_shape,
            ),
            dtype=data.dtype,
        )
        output[:Nx, :Ny, :Nz] = data  # Copy original data into the new array

        # Copy first slices to last.
        if is_boundary_periodic[0]:
            output[Nx] = output[0]
        if is_boundary_periodic[1]:
            output[:, Ny] = output[:, 0]
        if is_boundary_periodic[2]:
            output[:, :, Nz] = output[:, :, 0]
    else:
        output = data

    return output


def align_directors(n_reference: nField, n_target: nField) -> nField:
    """
    Align target director to have similar orientation as reference.
    This is used to handle the nematic symmetry of directors.
    """
    n_reference = check_Sn(n_reference, "n", is_3d_strict=False)
    n_target = check_Sn(n_target, "n", is_3d_strict=False)
    dots = np.einsum("...i,...i->...", n_reference, n_target)
    signs = np.where(dots < 0, -1, 1)
    return np.einsum("...,...i->...i", signs, n_target)


def align_stack(stack):

    dots = np.einsum("...i,...i->...", stack[:-1], stack[1:])

    flips = np.ones(stack.shape[:-1], dtype=np.int8)
    flips[1:] = np.where(dots < 0, -1, 1).astype(np.int8, copy=False)

    acc_flips = np.cumprod(flips, axis=0)

    stack *= acc_flips[..., np.newaxis].astype(stack.dtype, copy=False)
    return stack


def n_color_immerse(n: nField) -> List[Tuple]:
    """
    Map a nematic director field to RGB colors for visualization.

    This function encodes the orientation of a unit director vector `n`
    into RGB color values using a nonlinear polynomial mapping followed by
    a fixed linear transformation and scaling. The colormap is an immersion
    from RP^2 to R^3. This ensures that similar orientatioin of n refer to
    similar colors but different orientations might refer to the same color.

    The colormap is modified from boy's surface.

    The color is specifically desined to be distince on white background,
    and to get x, y, z direction closed to red, blue and green colors.

    x: [0.90535893, 0.22874911, 0.22062688]
    y: [0.05416607, 0.27934554, 0.48937438]
    z: [0.30416607, 0.90434554, 0.22687438]

    Parameters
    ----------
    n : array_like, shape (..., 3)
        Nematic director field.
        Can be of arbitrary leading dimensions.

    Returns
    -------
    colors : list of tuples. Each tuple has 3 elements
        RGB color with values typically in [0, 1], suitable for plotting.

    Examples
    --------
    >>> n = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    >>> colors = n_color_immerse(n)
    >>> colors
    [(0.39345357, 0.1364875 , 0.60187625),
     (0.09285714, 0.47428571, 0.605     ),
     (0.51845357, 0.4489875 , 0.47062625)]
    """

    n = check_Sn(n, "n", is_3d_strict=False, is_norm=True)

    RGB = np.zeros((*(np.shape(n)[:-1]), 3))

    x = n[..., 0]
    y = n[..., 1]
    z = n[..., 2]

    x2 = x**2
    y2 = y**2
    z2 = z**2

    RGB[..., 0] = (
        (2 * x2 - y2 - z2)
        + 2 * y * z * (y2 - z2)
        + z * x * (x2 - z2)
        + x * y * (y2 - x2)
    )
    RGB[..., 1] = (y2 - z2) + z * x * (z2 - x2) + x * y * (y2 - x2)
    RGB[..., 2] = (x + y + z) * ((x + y + z) ** 3 + 4 * (y - x) * (z - y) * (x - z))

    RGB[..., 0] = RGB[..., 0] / 2
    RGB[..., 1] = RGB[..., 1] * 7 / 8
    RGB[..., 2] = RGB[..., 2] / 8

    M = np.array(
        [
            [1.01667, -0.3, -0.48333],
            [-1.01667, -1.5, -1.31667],
            [-0.18333, 0.3, 1.31667],
        ]
    )

    result = np.einsum("...i, ji -> ...j", RGB, M)

    scales = np.array([2.1, 4.2, 2.0])
    offsets = np.array([0.45, 0.51, 0.23])
    result = result / scales + offsets

    colors = []
    for color in result:
        colors.append(tuple(color))

    return colors
