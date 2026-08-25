"""Legacy director-field defect detection retained during its replacement."""

import time

import numpy as np

from nematics3d.datatypes import (
    DefectIndex,
    DimensionFlagInput,
    as_dimension_info,
    as_director_field,
    nField,
)
from nematics3d.field import add_periodic_boundary, align_stack
from nematics3d.logging_decorator import logging_and_warning_decorator


def defect_detects_xyplane(n: np.ndarray, threshold: float) -> np.ndarray:
    """Detect defects in xy-plaquettes of a reoriented director field."""
    n = as_director_field(n, name="n", is_spatial_3d_required=True)

    a_orig = n[:-1, :-1]
    b_orig = n[1:, :-1]
    c_orig = n[1:, 1:]
    d_orig = n[:-1, 1:]
    stack = np.stack([a_orig, b_orig, c_orig, d_orig], axis=0)
    aligned_stack = align_stack(stack)
    a, _, _, d = aligned_stack

    test = np.einsum("...i,...i->...", a, d)

    coords = np.array(np.where(test < threshold)).T.astype(float)
    coords[:, [0, 1]] += 0.5
    return coords


@logging_and_warning_decorator()
def defect_detect(
    n_origin: nField,
    threshold: float = 0,
    is_boundary_periodic: DimensionFlagInput = 0,
    planes: DimensionFlagInput = 1,
    logger=None,
) -> DefectIndex:
    """Legacy public implementation archived before its replacement."""
    n_origin = as_director_field(
        n_origin,
        name="n_origin",
        is_spatial_3d_required=True,
    )
    is_boundary_periodic = as_dimension_info(is_boundary_periodic)
    planes = as_dimension_info(planes)

    logger.debug(
        "Start to defect defects. \n"
        f"Periodic boundary flags: {is_boundary_periodic}. \n"
        "Threshold of the inner product between the first and last director "
        f"is {threshold}."
    )

    n = add_periodic_boundary(n_origin, is_boundary_periodic)
    defect_indices = np.empty((0, 3), dtype=float)
    axis_permutations = {
        0: (2, 1, 0),
        1: (0, 2, 1),
        2: (0, 1, 2),
    }
    now = time.time()

    for axis in range(3):
        if not planes[axis]:
            continue

        perm = axis_permutations[axis]
        n_rot = np.moveaxis(n, [0, 1, 2], perm)
        coords = defect_detects_xyplane(n_rot, threshold)
        coords = coords[:, np.argsort(perm)]
        defect_indices = np.vstack((defect_indices, coords))
        logger.debug(
            f"Finished axis {axis}-direction in {round(time.time() - now, 2)}s"
        )
        now = time.time()

    for axis, is_periodic in enumerate(is_boundary_periodic):
        if is_periodic:
            defect_indices[:, axis] %= n_origin.shape[axis]

    defect_indices, _ = np.unique(defect_indices, axis=0, return_index=True)
    return defect_indices
