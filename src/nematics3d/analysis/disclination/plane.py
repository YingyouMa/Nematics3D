"""Defect detection on complete Cartesian and polar plane samplings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...field import align_directors
from ...geometry import points_membership_mask, wrap_angle_to_pi
from .detection import defect_detect
from .misc import defect_vicinity_grid


@dataclass(slots=True, frozen=True)
class PlaneDefectResult:
    """Defects detected on a complete plane before any bounds filtering."""

    positions_all: np.ndarray | None
    adjacent_mask_all: np.ndarray


def detect_defects_on_cartesian_plane(
    *,
    directors_all,
    grid_shape,
    grid_indices,
    origin_grid0,
    axis1,
    axis2,
    spacing,
    spacing_extra,
) -> PlaneDefectResult:
    """Detect director defects on one complete rectangular sampling plane."""
    grid_indices = np.asarray(grid_indices)
    shape_all = tuple(int(value) for value in grid_shape)
    if len(shape_all) != 2 or int(np.prod(shape_all)) != len(grid_indices):
        raise ValueError(
            "`grid_shape` must describe the complete flattened grid indices."
        )
    directors_all = np.asarray(directors_all, dtype=float).reshape((*shape_all, 1, 3))

    defect_plane_index = defect_detect(
        directors_all,
        planes=(False, False, True),
        is_input_validated=True,
    )
    defect_vicinity_index = defect_vicinity_grid(
        defect_plane_index,
        num_shell=1,
    ).astype(int)
    defect_vicinity_index = defect_vicinity_index.reshape((-1, 3))[:, :-1]
    defect_plane_index = defect_plane_index[:, :-1]

    adjacent_mask_all = points_membership_mask(
        grid_indices.reshape((-1, 2)).astype(int),
        defect_vicinity_index,
    )

    if len(defect_plane_index) == 0:
        positions_all = None
    else:
        step_both = np.array(
            [
                np.asarray(axis1, dtype=float) * float(spacing),
                np.asarray(axis2, dtype=float) * float(spacing_extra),
            ]
        )
        positions_all = np.einsum(
            "ai,ib->ab", defect_plane_index, step_both
        ) + np.asarray(origin_grid0, dtype=float)

    return PlaneDefectResult(
        positions_all=positions_all,
        adjacent_mask_all=np.asarray(adjacent_mask_all, dtype=bool),
    )


def detect_defects_on_polar_plane(
    *,
    points_all,
    polar_coords,
    ring_offsets,
    directors_all,
    threshold: float = 0.0,
) -> PlaneDefectResult:
    """Detect director defects on one complete polar sampling plane."""
    points = np.asarray(points_all, dtype=float)
    polar = np.asarray(polar_coords, dtype=float)
    ring_offsets = np.asarray(ring_offsets, dtype=np.int64)
    directors = np.asarray(directors_all, dtype=float)

    n_rings = ring_offsets.shape[0] - 1
    adjacent_mask = np.zeros((points.shape[0],), dtype=bool)
    defect_centers_chunks: list[np.ndarray] = []

    start_ring = 0
    if n_rings >= 1:
        s0, e0 = ring_offsets[0], ring_offsets[1]
        if (e0 - s0) == 1 and np.isclose(polar[s0, 0], 0.0):
            start_ring = 1

    def process_outer_to_inner(s_outer, e_outer, s_inner, e_inner):
        n_outer = e_outer - s_outer
        n_inner = e_inner - s_inner
        if n_outer < 2 or n_inner < 2:
            return

        theta_outer = polar[s_outer:e_outer, 1]
        theta_inner = polar[s_inner:e_inner, 1]
        j = np.arange(n_outer, dtype=np.int64)
        jn = (j + 1) % n_outer
        idx_a = s_outer + j
        idx_b = s_outer + jn
        theta_a = theta_outer[j]
        theta_b = theta_outer[jn]

        diff_b = wrap_angle_to_pi(theta_inner[None, :] - theta_b[:, None])
        c_local = np.argmin(np.abs(diff_b), axis=1).astype(np.int64)

        order = np.argsort(theta_inner)
        rank_of = np.empty_like(order)
        rank_of[order] = np.arange(n_inner, dtype=np.int64)
        c_rank = rank_of[c_local]
        prev_local = order[(c_rank - 1) % n_inner]
        next_local = order[(c_rank + 1) % n_inner]
        d_prev = np.abs(wrap_angle_to_pi(theta_inner[prev_local] - theta_a))
        d_next = np.abs(wrap_angle_to_pi(theta_inner[next_local] - theta_a))
        d_local = np.where(d_prev <= d_next, prev_local, next_local).astype(np.int64)

        idx_c = s_inner + c_local
        idx_d = s_inner + d_local
        pa, pb, pc, pd = points[idx_a], points[idx_b], points[idx_c], points[idx_d]
        a = directors[idx_a]
        b = align_directors(a, directors[idx_b])
        c = align_directors(b, directors[idx_c])
        d = align_directors(c, directors[idx_d])
        hit = np.einsum("...i,...i->...", a, d) < threshold
        if not np.any(hit):
            return

        defect_centers_chunks.append(((pa + pb + pc + pd) * 0.25)[hit])
        adjacent_mask[idx_a[hit]] = True
        adjacent_mask[idx_b[hit]] = True
        inner_idx = np.unique(np.concatenate([idx_c[hit], idx_d[hit]]))
        adjacent_mask[inner_idx] = True

    outermost = n_rings - 1
    last_good_ring = None
    for r in range(outermost, start_ring, -1):
        s_outer, e_outer = ring_offsets[r], ring_offsets[r + 1]
        s_inner, e_inner = ring_offsets[r - 1], ring_offsets[r]
        if (e_inner - s_inner) < 6:
            last_good_ring = r
            break
        process_outer_to_inner(s_outer, e_outer, s_inner, e_inner)

    if last_good_ring is None and start_ring < n_rings:
        last_good_ring = (
            start_ring
            if (ring_offsets[start_ring + 1] - ring_offsets[start_ring]) >= 6
            else None
        )

    if last_good_ring is not None:
        s, e = ring_offsets[last_good_ring], ring_offsets[last_good_ring + 1]
        v = directors[s:e]
        if len(v) >= 6:
            dots = np.einsum("ij,ij->i", v[:-1], v[1:])
            step_sign = np.where(dots < 0.0, -1.0, 1.0).astype(v.dtype)
            cum_sign = np.concatenate(
                [np.ones((1,), dtype=v.dtype), np.cumprod(step_sign)]
            )
            if float(np.dot(v[0], v[-1] * cum_sign[-1])) < threshold:
                adjacent_mask[s:e] = True
                defect_centers_chunks.append(points[s:e].mean(axis=0, keepdims=True))

    positions_all = (
        np.concatenate(defect_centers_chunks, axis=0).astype(float)
        if defect_centers_chunks
        else None
    )
    return PlaneDefectResult(positions_all, adjacent_mask)


__all__ = [
    "PlaneDefectResult",
    "detect_defects_on_cartesian_plane",
    "detect_defects_on_polar_plane",
]
