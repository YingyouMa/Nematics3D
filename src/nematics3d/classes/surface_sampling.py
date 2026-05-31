"""Host-style surface sampling schema built around PolyData surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from ..datatypes import UNSET, Unset, as_Number
from ..logging_decorator import logging_and_warning_decorator
from .host_base import HostBase, OptsBase
from .visual.plot_polydata import as_polydata_input, make_clean_polydata


def _as_surface_polydata_input(data, *, name: str):
    """
    Normalize one input to clean ``pyvista.PolyData`` and require real surface area.

    SurfaceSampling needs an actual surface, not merely any PolyData container.
    Point-only or line-only PolyData inputs are therefore rejected even though
    they remain valid inputs for the more general PlotPolyData visual wrapper.
    """

    poly = make_clean_polydata(as_polydata_input(data, name=name))
    surface = poly.extract_surface().triangulate().clean()

    if surface.n_cells == 0:
        raise ValueError(
            f"{name} must contain surface cells after triangulation; got an empty "
            "surface."
        )

    if float(surface.area) <= 0.0:
        raise ValueError(
            f"{name} must contain a nonzero-area surface; point-only or line-only "
            "PolyData is not valid for surface sampling."
        )

    return poly


def _helper_prepare_surface(poly: pv.PolyData) -> pv.PolyData:
    """Return one cleaned triangulated surface with point normals attached."""
    surface = poly.extract_surface().triangulate().clean()
    surface = surface.compute_normals(
        cell_normals=False,
        point_normals=True,
        split_vertices=False,
        consistent_normals=True,
        auto_orient_normals=False,
        inplace=False,
    )
    return surface


def _helper_resolve_sample_count_target(area: float, spacing: float) -> int:
    """Infer one target sample count from area and approximate mean spacing."""
    hex_area = np.sqrt(3.0) / 2.0
    return max(int(np.round(area / (hex_area * spacing**2))), 1)


def _helper_resolve_spacing_for_target_count(area: float, target_count: int) -> float:
    """Infer one effective spacing that yields roughly the requested target count."""
    hex_area = np.sqrt(3.0) / 2.0
    return float(np.sqrt(area / (hex_area * max(int(target_count), 1))))


def _helper_sample_triangle_candidates(
    surface: pv.PolyData,
    candidate_count: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample candidate points on a triangulated surface by area-weighted barycentrics."""
    faces = np.asarray(surface.faces, dtype=int).reshape(-1, 4)
    triangles = faces[:, 1:]
    vertices = np.asarray(surface.points, dtype=float)[triangles]

    cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = areas > 0.0
    if not np.any(valid):
        raise ValueError("Surface triangulation produced no positive-area triangles.")

    vertices = vertices[valid]
    areas = areas[valid]
    triangle_ids = rng.choice(
        len(vertices),
        size=int(candidate_count),
        p=areas / np.sum(areas),
    )
    sampled_triangles = vertices[triangle_ids]

    r1 = rng.random(candidate_count)
    r2 = rng.random(candidate_count)
    sqrt_r1 = np.sqrt(r1)
    bary0 = 1.0 - sqrt_r1
    bary1 = sqrt_r1 * (1.0 - r2)
    bary2 = sqrt_r1 * r2

    return (
        bary0[:, None] * sampled_triangles[:, 0]
        + bary1[:, None] * sampled_triangles[:, 1]
        + bary2[:, None] * sampled_triangles[:, 2]
    )


def _helper_farthest_point_sampling(
    candidates: np.ndarray,
    target_count: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Down-select candidates by seeded farthest-point sampling."""
    target_count = int(target_count)
    if target_count >= len(candidates):
        return np.asarray(candidates, dtype=float).copy()

    first_index = int(rng.integers(len(candidates)))
    selected_indices = [first_index]

    diff = candidates - candidates[first_index]
    min_dist2 = np.einsum("ij,ij->i", diff, diff)
    min_dist2[first_index] = -np.inf

    for _ in range(1, target_count):
        next_index = int(np.argmax(min_dist2))
        selected_indices.append(next_index)

        diff = candidates - candidates[next_index]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        min_dist2 = np.minimum(min_dist2, dist2)
        min_dist2[selected_indices] = -np.inf

    return np.asarray(candidates[selected_indices], dtype=float)


def _helper_project_points_to_surface(
    points: np.ndarray,
    surface: pv.PolyData,
) -> np.ndarray:
    """Project points to their closest locations on the current surface."""
    _, closest_points = surface.find_closest_cell(
        np.asarray(points, dtype=float),
        return_closest_point=True,
    )
    return np.asarray(closest_points, dtype=float)


def _helper_relax_points(
    points: np.ndarray,
    surface: pv.PolyData,
    *,
    relax_steps: int,
    k_neighbors: int,
) -> np.ndarray:
    """Run simple kNN repulsion with projection back to the surface."""
    points = np.asarray(points, dtype=float).copy()
    if relax_steps <= 0 or len(points) <= 1:
        return points

    move_ratio = 0.15
    eps = 1.0e-12
    for _ in range(int(relax_steps)):
        neighbor_count = min(int(k_neighbors) + 1, len(points))
        if neighbor_count <= 1:
            return points

        tree = cKDTree(points)
        neighbor_dist, neighbor_ids = tree.query(points, k=neighbor_count)
        if neighbor_ids.ndim == 1:
            neighbor_ids = neighbor_ids[:, None]
            neighbor_dist = neighbor_dist[:, None]

        neighbor_ids = neighbor_ids[:, 1:]
        neighbor_dist = neighbor_dist[:, 1:]
        if neighbor_ids.shape[1] == 0:
            return points

        # Use short-range repulsion rather than centroid attraction. The old
        # centroid-based update behaved like Laplacian smoothing and could
        # collapse the sample distribution after repeated projection.
        offset_vecs = points[:, None, :] - points[neighbor_ids]
        dist_safe = np.maximum(neighbor_dist, eps)
        repel_force = (offset_vecs / dist_safe[..., None] ** 3).sum(axis=1)
        repel_norm = np.linalg.norm(repel_force, axis=1, keepdims=True)
        repel_unit = np.divide(repel_force, np.maximum(repel_norm, eps))

        local_scale = np.median(dist_safe, axis=1, keepdims=True)
        moved_points = points + move_ratio * local_scale * repel_unit
        points = _helper_project_points_to_surface(moved_points, surface)

    return points


@dataclass(slots=True, repr=False)
class OptsSurfaceSampling(OptsBase):
    """
    Options controlling reproducible near-uniform sampling on one surface.

    The paired host takes one valid PolyData-like surface object and uses these
    options to determine how many points to sample and how strongly to regularize
    them afterwards. The input surface is not restricted to an isosurface; any
    legal PyVista/VTK PolyData object carrying surface geometry is acceptable.
    """

    spacing: float | None | Unset = UNSET
    seed: int | Unset = UNSET
    oversample: int | Unset = UNSET
    relax_steps: int | Unset = UNSET
    k_neighbors: int | Unset = UNSET
    default_sample_count_target: int | Unset = UNSET

    # fmt: off
    __attrs__: ClassVar[Mapping[str, str]] = {
        **dict(OptsBase.__attrs__),
        "spacing":     "Approximate average spacing between sampled points on the surface. If None, the host estimates a spacing that yields roughly default_sample_count_target points.",
        "seed":        "Random seed controlling candidate generation and the first FPS choice so sampling remains reproducible.",
        "oversample":  "Candidate-pool multiplier used before farthest-point down-selection.",
        "relax_steps": "Number of optional local relaxation passes applied after farthest-point sampling.",
        "k_neighbors": "Nearest-neighbor count used by each relaxation pass when relax_steps is positive.",
        "default_sample_count_target": "Fallback target point count used when spacing is None and the host enters automatic spacing mode.",
    }

    impl_validators: ClassVar[Mapping[str, Any]] = {
        **dict(OptsBase.impl_validators),
        "spacing":     lambda v, d: None if v is None else as_Number(v, name=d, value_range=(1e-12, np.inf)),
        "seed":        lambda v, d: as_Number(v, name=d, is_int=True),
        "oversample":  lambda v, d: as_Number(v, name=d, is_int=True, value_range=(1, np.inf)),
        "relax_steps": lambda v, d: as_Number(v, name=d, is_int=True, value_range=(0, np.inf)),
        "k_neighbors": lambda v, d: as_Number(v, name=d, is_int=True, value_range=(1, np.inf)),
        "default_sample_count_target": lambda v, d: as_Number(v, name=d, is_int=True, value_range=(1, np.inf)),
    }

    impl_defaults_frozen: ClassVar[Mapping[str, Any]] = MappingProxyType({
        **dict(OptsBase.impl_defaults_frozen),
        "tag":                         "surface sampling options",
        "spacing":                     None,
        "seed":                        0,
        "oversample":                  6,
        "relax_steps":                 0,
        "k_neighbors":                 8,
        "default_sample_count_target": 100,
    })
    # fmt: on


class SurfaceSampling(HostBase):
    """
    Host object pairing one surface with derived sampled point-cloud state.

    This declaration stage intentionally focuses on the managed schema only:
    one canonical PolyData-like surface input and the derived caches directly
    needed by the current surface-sampling workflow.
    """

    # fmt: off
    __attr_defs__: ClassVar[Mapping[str, dict[str, Any]]] = {
        **dict(HostBase.__attr_defs__),
        "raw_surface": {
            "doc": (
                "The canonical input surface stored as cleaned pyvista.PolyData. "
                "Any legal PyVista/VTK PolyData-like object with valid surface "
                "geometry may be provided."
            ),
            "validator": lambda v, d: _as_surface_polydata_input(v, name=d),
            "is_reapply_opts_after_raw": True,
        },
        "calc_surface_clean": {
            "doc": (
                "The cleaned surface currently used for sampling after later "
                "host-side preprocessing such as surface extraction or triangulation."
            ),
        },
        "calc_surface_area": {
            "doc": (
                "Resolved total area of the cleaned surface used to infer the "
                "target sample count from the requested spacing."
            ),
        },
        "calc_sample_count_target": {
            "doc": (
                "Resolved target number of sample points inferred from the current "
                "surface area and opts spacing before candidate oversampling."
            ),
        },
        "calc_surface_points": {
            "doc": "Point coordinates of the cleaned input surface as an (N, 3) array.",
        },
        "calc_surface_normals": {
            "doc": (
                "Resolved surface normals associated with calc_surface_points as "
                "an (N, 3) array."
            ),
        },
        "calc_sample_points": {
            "doc": (
                "The final sampled point coordinates exposed by this host as an "
                "(N, 3) array."
            ),
        },
        "field": {
            "doc": "The interpolated field object attached to this surface sampling.",
            "kind": "relation",
            "is_weak_by_default": True,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "result": {
            "doc": "Read-only: Alias of `calc_sample_points`.",
            "kind": "property",
        },
    }
    # fmt: on

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.get("kind") not in ("relation", "property")
    )

    # ==================== OVERRIDE ====================
    # SurfaceSampling overrides HostBase.__init__ because it must initialize
    # its derived surface/sampling caches and trigger the first sampling pass.
    # ==================================================
    def __init__(
        self,
        surface,
        name: str | None = None,
        name_replace: str = "surface_sampling",
        opts: OptsSurfaceSampling | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            OptsSurfaceSampling,
            opts,
            opts_defaults_override,
            name=name,
            name_replace=name_replace,
            surface=surface,
            **kwargs,
        )

        object.__setattr__(self, "calc_surface_clean", None)
        object.__setattr__(self, "calc_surface_area", 0.0)
        object.__setattr__(self, "calc_sample_count_target", 0)
        object.__setattr__(self, "calc_surface_points", np.empty((0, 3), dtype=float))
        object.__setattr__(self, "calc_surface_normals", np.empty((0, 3), dtype=float))
        object.__setattr__(self, "calc_sample_points", np.empty((0, 3), dtype=float))

        self.opts.act_finalize(defaults=self.opts_defaults)
        self._helper_commit_apply_opts(is_reapply_opts=True)

    @property
    def result(self):
        """Return the sampled point coordinates."""
        return self.calc_sample_points

    # ==================== OVERRIDE ====================
    # SurfaceSampling overrides HostBase._helper_commit_apply_opts_main because
    # finalized opts must be translated into a fresh surface-sampling result.
    # ==================================================
    @logging_and_warning_decorator()
    def _helper_commit_apply_opts_main(
        self,
        is_reapply_opts: bool = False,
        logger=None,
        **kwargs,
    ):
        if not is_reapply_opts and not kwargs:
            return

        with self.opts.act_internal_update():
            for key, value in kwargs.items():
                setattr(self.opts, key, value)

        surface_clean = _helper_prepare_surface(self.raw_surface)
        surface_area = float(surface_clean.area)
        if self.opts.spacing is None:
            spacing_effective = _helper_resolve_spacing_for_target_count(
                surface_area,
                int(self.opts.default_sample_count_target),
            )
        else:
            spacing_effective = float(self.opts.spacing)

        sample_count_target = _helper_resolve_sample_count_target(
            surface_area,
            spacing_effective,
        )
        candidate_count = max(
            int(self.opts.oversample) * sample_count_target,
            sample_count_target + 1,
        )

        rng = np.random.default_rng(int(self.opts.seed))
        candidate_points = _helper_sample_triangle_candidates(
            surface_clean,
            candidate_count,
            rng=rng,
        )
        sample_points = _helper_farthest_point_sampling(
            candidate_points,
            sample_count_target,
            rng=rng,
        )
        sample_points = _helper_relax_points(
            sample_points,
            surface_clean,
            relax_steps=int(self.opts.relax_steps),
            k_neighbors=int(self.opts.k_neighbors),
        )

        surface_points = np.asarray(surface_clean.points, dtype=float)
        surface_normals = np.asarray(surface_clean.point_data["Normals"], dtype=float)
        object.__setattr__(self, "calc_surface_clean", surface_clean)
        object.__setattr__(self, "calc_surface_area", surface_area)
        object.__setattr__(self, "calc_sample_count_target", sample_count_target)
        object.__setattr__(self, "calc_surface_points", surface_points)
        object.__setattr__(self, "calc_surface_normals", surface_normals)
        object.__setattr__(self, "calc_sample_points", sample_points)

        logger.info(
            f"Resampled {self.name!r}: area={surface_area:.6g}, "
            f"spacing_effective={spacing_effective:.6g}, "
            f"target_count={sample_count_target}, candidate_count={candidate_count}."
        )

        if self.field:
            self.field.act_refresh()
