from numbers import Integral
from time import perf_counter

import numexpr as ne
import numpy as np

from ...datatypes import (
    DefectIndex,
    DimensionInfo,
    as_bool,
    as_dimension_info,
    as_director_field,
    nField,
)
from ...field import add_periodic_boundary
from ...logging_decorator import logging_and_warning_decorator

_DEFECT_AXIS_PERMUTATIONS = (
    (2, 1, 0),
    (0, 2, 1),
    (0, 1, 2),
)


def _validate_worker_count(worker_count):
    if worker_count is None:
        return None
    if isinstance(worker_count, bool) or not isinstance(worker_count, Integral):
        raise TypeError("'worker_count' must be a positive integer or None.")

    worker_count = int(worker_count)
    if not 1 <= worker_count <= ne.MAX_THREADS:
        raise ValueError(
            "'worker_count' must be between 1 and the NumExpr limit "
            f"({ne.MAX_THREADS}), or None."
        )
    return worker_count


def _defect_detects_xyplane_unchecked(n, threshold):
    """Detect xy-plaquette defects in an already validated director field."""
    a = n[:-1, :-1]
    b = n[1:, :-1]
    c = n[1:, 1:]
    d = n[:-1, 1:]

    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    cx, cy, cz = c[..., 0], c[..., 1], c[..., 2]
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]

    mask = ne.evaluate(
        "where("
        "((((ax*bx + ay*by + az*bz) < 0) "
        "!= ((bx*cx + by*cy + bz*cz) < 0)) "
        "!= ((cx*dx + cy*dy + cz*dz) < 0)), "
        "-(ax*dx + ay*dy + az*dz), "
        "(ax*dx + ay*dy + az*dz)) < threshold",
        local_dict={
            "ax": ax,
            "ay": ay,
            "az": az,
            "bx": bx,
            "by": by,
            "bz": bz,
            "cx": cx,
            "cy": cy,
            "cz": cz,
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "threshold": float(threshold),
        },
        optimization="moderate",
    )

    coordinates = np.argwhere(mask).astype(float, copy=False)
    coordinates[:, :2] += 0.5
    return coordinates


@logging_and_warning_decorator()
def defect_detect(
    n_origin: nField,
    threshold: float = 0,
    is_boundary_periodic: DimensionInfo = 0,
    planes: DimensionInfo = 1,
    *,
    worker_count: int | None = None,
    is_input_validated: bool = False,
    logger=None,
) -> DefectIndex:
    """Detect plaquette defects in a three-dimensional director field.

    Set ``is_input_validated=True`` only when the caller guarantees that
    ``n_origin`` is a finite real array with shape ``(Nx, Ny, Nz, 3)``. This
    avoids repeating validation for trusted upstream results such as the
    director returned by ``q_diagonalize``.

    Parameters
    ----------
    n_origin : nField
        Director field with shape ``(Nx, Ny, Nz, 3)``.
    threshold : float, optional
        A plaquette is defective when its aligned closure dot product is less
        than this value.
    is_boundary_periodic : DimensionInfo, optional
        Periodicity along the three spatial axes.
    planes : DimensionInfo, optional
        Select plaquettes normal to the x, y, and z axes.
    worker_count : int or None, optional
        NumExpr thread count used during this call. The previous process-wide
        setting is restored before returning.
    is_input_validated : bool, optional
        Skip director-field validation when the input contract is already
        guaranteed by the caller. Default is ``False``.

    Returns
    -------
    DefectIndex
        Defect coordinates with one integer and two half-integer components.
        Coordinates are grouped by their plaquette-normal axis.

    Notes
    -----
    Changing NumExpr's thread count is process-wide. Concurrent calls should
    therefore leave ``worker_count=None`` and configure NumExpr externally.
    """
    is_input_validated = as_bool(
        is_input_validated,
        name="is_input_validated",
    )

    if is_input_validated:
        n_origin = np.asarray(n_origin)
    else:
        n_origin = as_director_field(
            n_origin,
            name="n_origin",
            is_spatial_3d_required=True,
            is_normalized=False,
        )

    is_boundary_periodic = as_dimension_info(
        is_boundary_periodic,
        name="is_boundary_periodic",
        is_bool=True,
    )
    planes = as_dimension_info(planes, name="planes", is_bool=True)
    worker_count = _validate_worker_count(worker_count)

    previous_worker_count = ne.get_num_threads()
    if worker_count is not None:
        ne.set_num_threads(worker_count)

    try:
        start = perf_counter()
        active_worker_count = ne.get_num_threads()
        original_shape = n_origin.shape[:3]
        periodic_axes = tuple(bool(value) for value in is_boundary_periodic)
        selected_planes = tuple(bool(value) for value in planes)
        logger.debug(
            f"Preparing to detect defects in director field "
            f"shape={n_origin.shape}; threshold={threshold}, "
            f"is_boundary_periodic={periodic_axes}, "
            f"planes={selected_planes}, worker_count={active_worker_count}."
        )

        n = add_periodic_boundary(n_origin, is_boundary_periodic)
        coordinate_chunks = []
        defect_counts = [0, 0, 0]

        for axis, is_plane_selected in enumerate(planes):
            if not is_plane_selected:
                continue

            permutation = _DEFECT_AXIS_PERMUTATIONS[axis]
            n_rotated = np.moveaxis(n, (0, 1, 2), permutation)
            n_rotated = n_rotated[:, :, : original_shape[axis], :]
            coordinates = _defect_detects_xyplane_unchecked(n_rotated, threshold)
            if coordinates.size:
                defect_counts[axis] = len(coordinates)
                coordinate_chunks.append(coordinates[:, permutation])

        if not coordinate_chunks:
            defects = np.empty((0, 3), dtype=float)
        else:
            defects = np.concatenate(coordinate_chunks, axis=0)

        logger.debug(
            f"Detected {len(defects):,} defects; counts by normal axis "
            f"(x, y, z)={tuple(defect_counts)}, "
            f"elapsed={perf_counter() - start:.3f} seconds."
        )
        return defects
    finally:
        if worker_count is not None:
            ne.set_num_threads(previous_worker_count)
