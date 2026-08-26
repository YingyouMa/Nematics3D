"""Archived tuple-returning Q-tensor diagonalization implementation.

This file preserves the former public implementation from ``field.py`` at
commit ``cf45692``.  It is retained for historical comparison and is not part
of the supported package API.
"""

import time
from typing import Tuple, Union

import numpy as np

from nematics3d.datatypes import QField5, QField9, SField, as_qfield9, nField
from nematics3d.logging_decorator import logging_and_warning_decorator


@logging_and_warning_decorator()
def Q_diagonalize(
    qtensor: Union[QField5, QField9], logger=None
) -> Tuple[SField, nField]:
    """Return the scalar order parameter and director field as ``(S, n)``."""
    Q: QField9 = as_qfield9(qtensor, is_strict_3d_field=False)
    float_dtype = np.result_type(Q.dtype, np.float64)
    Q = np.asarray(Q, dtype=float_dtype)
    eps = np.finfo(Q.dtype).eps
    q_abs_max = np.max(np.abs(Q), axis=(-2, -1))
    q_scale = np.maximum(1.0, q_abs_max)

    logger.debug("Computing tensor invariants (p, q, r).")
    start = time.time()
    p = 0.5 * np.einsum("...ab, ...ba -> ...", Q, Q)
    q = np.linalg.det(Q)
    r = 2 * np.sqrt(p / 3)
    logger.debug(f"Tensor invariants computed in {time.time() - start:.2f} seconds.")

    logger.debug("Computing largest eigenvalue lambda_max.")
    start = time.time()
    isotropic_tol = 32 * eps * q_scale
    is_near_isotropic = r <= isotropic_tol
    cos_arg = np.zeros_like(r)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(4 * q, r**3, out=cos_arg, where=~is_near_isotropic)
    cos_arg = np.clip(cos_arg, -1.0, 1.0)
    lambda_max = np.zeros_like(r)
    lambda_max[~is_near_isotropic] = r[~is_near_isotropic] * np.cos(
        (1 / 3) * np.arccos(cos_arg[~is_near_isotropic])
    )
    logger.debug(f"lambda_max computed in {time.time() - start:.2f} seconds.")

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

    n: nField = n
    logger.debug(f"Director field computed in {time.time() - start:.2f} seconds.")
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
