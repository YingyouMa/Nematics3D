"""Host-style surface sampling schema built around PolyData surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from ..datatypes import UNSET, Unset, as_bool, as_number, as_readonly_array
from ..logging_decorator import logging_and_warning_decorator
from ..core.class_base import AttrDef
from ..core.host_base import HostBase, OptsBase
from ..geometry.polydata import as_polydata_input, copy_polydata_geometry


def _as_surface_polydata_input(data, *, name: str):
    """
    Normalize one input to an isolated geometry-only ``pyvista.PolyData``.

    Surface extraction, triangulation, topology cleaning, and area validation
    are intentionally deferred to :func:`_helper_prepare_surface`, so each raw
    input is processed through that comparatively expensive pipeline only once.
    """

    return copy_polydata_geometry(as_polydata_input(data, name=name))


def _helper_prepare_surface(poly: pv.PolyData) -> pv.PolyData:
    """Return one validated, triangulated surface with point normals attached."""
    surface = poly.extract_surface().triangulate().clean()

    if surface.n_cells == 0:
        raise ValueError(
            "surface must contain surface cells after triangulation; got an empty "
            "surface."
        )

    if float(surface.area) <= 0.0:
        raise ValueError(
            "surface must contain a nonzero-area surface; point-only or line-only "
            "PolyData is not valid for surface sampling."
        )

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
    """
    Infer a target count from surface area and an approximate mean spacing.

    The spacing is a density-control estimate based on ideal hexagonal packing;
    it is not a guaranteed nearest-neighbor distance or minimum separation.
    """
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
    """
    Run simple 3D-Euclidean kNN repulsion and project back to the surface.

    Neighborhoods are determined by ambient-space Euclidean distance rather
    than mesh connectivity or geodesic distance. Consequently, separate parts
    of a folded or closely self-approaching surface may influence each other.
    """
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


def _helper_calc_sample_surface_geometry(
    points: np.ndarray,
    surface: pv.PolyData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve closest triangles, barycentrics, and interpolated point normals."""
    cell_ids = np.asarray(surface.find_closest_cell(points), dtype=int).reshape(-1)
    point_ids = []
    for cell_id in cell_ids:
        cell_point_ids = np.asarray(surface.get_cell(int(cell_id)).point_ids, dtype=int)
        if cell_point_ids.shape != (3,):
            raise ValueError(
                "Sample-normal interpolation requires a fully triangulated surface; "
                f"closest cell {int(cell_id)} has {len(cell_point_ids)} points."
            )
        point_ids.append(cell_point_ids)

    triangle_point_ids = np.asarray(point_ids, dtype=int)
    triangles = np.asarray(surface.points, dtype=float)[triangle_point_ids]
    point_offsets = np.asarray(points, dtype=float) - triangles[:, 0]
    edge_0 = triangles[:, 1] - triangles[:, 0]
    edge_1 = triangles[:, 2] - triangles[:, 0]

    dot_00 = np.einsum("ij,ij->i", edge_0, edge_0)
    dot_01 = np.einsum("ij,ij->i", edge_0, edge_1)
    dot_11 = np.einsum("ij,ij->i", edge_1, edge_1)
    dot_20 = np.einsum("ij,ij->i", point_offsets, edge_0)
    dot_21 = np.einsum("ij,ij->i", point_offsets, edge_1)
    denominator = dot_00 * dot_11 - dot_01 * dot_01
    if np.any(denominator <= np.finfo(float).eps):
        raise ValueError(
            "Sample-normal interpolation encountered a degenerate triangle."
        )

    bary_1 = (dot_11 * dot_20 - dot_01 * dot_21) / denominator
    bary_2 = (dot_00 * dot_21 - dot_01 * dot_20) / denominator
    barycentric = np.column_stack((1.0 - bary_1 - bary_2, bary_1, bary_2))
    barycentric = np.clip(barycentric, 0.0, 1.0)
    barycentric /= np.sum(barycentric, axis=1, keepdims=True)

    point_normals = np.asarray(surface.point_data["Normals"], dtype=float)
    triangle_normals = point_normals[triangle_point_ids]
    sample_normals = np.einsum("ni,nij->nj", barycentric, triangle_normals)
    normal_magnitudes = np.linalg.norm(sample_normals, axis=1, keepdims=True)
    if np.any(normal_magnitudes <= np.finfo(float).eps):
        raise ValueError("Interpolated sample normal has zero magnitude.")
    sample_normals /= normal_magnitudes
    return cell_ids, barycentric, sample_normals


def _helper_calc_spacing_statistics(points: np.ndarray) -> tuple[float, float, float]:
    """Return mean, minimum, and standard deviation of nearest-neighbor distance."""
    if len(points) <= 1:
        return float("nan"), float("nan"), float("nan")

    nearest_distances = cKDTree(points).query(points, k=2)[0][:, 1]
    return (
        float(np.mean(nearest_distances)),
        float(np.min(nearest_distances)),
        float(np.std(nearest_distances)),
    )


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
    max_sample_count: int | None | Unset = UNSET
    is_calc_sample_normals: bool | Unset = UNSET
    is_calc_spacing_statistics: bool | Unset = UNSET

    # fmt: off
    __attrs__: ClassVar[Mapping[str, str]] = {
        **dict(OptsBase.__attrs__),
        "spacing":     "Approximate mean spacing used to estimate sampling density from surface area. It does not guarantee a nearest-neighbor distance or minimum separation. If None, the host estimates a spacing that yields roughly default_sample_count_target points.",
        "seed":        "Random seed controlling candidate generation and the first FPS choice so sampling remains reproducible.",
        "oversample":  "Candidate-pool multiplier used before farthest-point down-selection.",
        "relax_steps": "Number of optional local relaxation passes applied after farthest-point sampling.",
        "k_neighbors": "Nearest-neighbor count used by each relaxation pass when relax_steps is positive.",
        "default_sample_count_target": "Fallback target point count used when spacing is None and the host enters automatic spacing mode.",
        "max_sample_count": "Hard safety limit checked before candidate allocation. None disables the limit.",
        "is_calc_sample_normals": "Whether each update computes closest triangle IDs, barycentric coordinates, and interpolated sample-point normals.",
        "is_calc_spacing_statistics": "Whether each update computes nearest-neighbor mean, minimum, and standard-deviation spacing statistics.",
    }

    impl_validators: ClassVar[Mapping[str, Any]] = {
        **dict(OptsBase.impl_validators),
        "spacing":     lambda v, d: None if v is None else as_number(v, name=d, value_range=(1e-12, np.inf)),
        "seed":        lambda v, d: as_number(v, name=d, is_integer=True),
        "oversample":  lambda v, d: as_number(v, name=d, is_integer=True, value_range=(1, np.inf)),
        "relax_steps": lambda v, d: as_number(v, name=d, is_integer=True, value_range=(0, np.inf)),
        "k_neighbors": lambda v, d: as_number(v, name=d, is_integer=True, value_range=(1, np.inf)),
        "default_sample_count_target": lambda v, d: as_number(v, name=d, is_integer=True, value_range=(1, np.inf)),
        "max_sample_count": lambda v, d: None if v is None else as_number(v, name=d, is_integer=True, value_range=(1, np.inf)),
        "is_calc_sample_normals": lambda v, d: as_bool(v, name=d),
        "is_calc_spacing_statistics": lambda v, d: as_bool(v, name=d),
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
        "max_sample_count":            100_000,
        "is_calc_sample_normals":      False,
        "is_calc_spacing_statistics":  False,
    })
    # fmt: on


class SurfaceSampling(HostBase):
    """
    Host object pairing one surface with derived sampled point-cloud state.

    This declaration stage intentionally focuses on the managed schema only:
    one canonical PolyData-like surface input and the derived caches directly
    needed by the current surface-sampling workflow.

    ``opts.spacing`` controls the approximate mean sampling density; it does
    not enforce an exact nearest-neighbor spacing or minimum separation. The
    optional relaxation stage finds neighbors using 3D Euclidean distance, not
    mesh-geodesic distance, so nearby sheets of a folded surface can interact.

    Updates use local cache signatures. Replacing ``raw_surface`` invalidates
    every layer; sampling options invalidate sampled points and enabled derived
    outputs; optional-output switches update only their own derived layer.
    In-place mutation of ``raw_surface`` is not detected automatically.
    """

    # fmt: off
    __attr_defs__ = {
        "raw_surface": AttrDef(
            doc=(
                "The canonical input stored as an isolated, geometry-only "
                "pyvista.PolyData. Surface extraction, triangulation, topology "
                "cleaning, and normal calculation belong to calc_surface_clean."
            ),
            kind="raw",
            validator=lambda v, d: _as_surface_polydata_input(v, name=d),
            is_reapply_opts_after_raw=True,
        ),
        "calc_surface_clean": AttrDef(
            doc=(
                "The cleaned surface currently used for sampling after later "
                "host-side preprocessing such as surface extraction or triangulation."
            ),
            kind="calc",
        ),
        "calc_surface_area": AttrDef(
            doc=(
                "Resolved total area of the cleaned surface used to infer the "
                "target sample count from the requested spacing."
            ),
            kind="calc",
        ),
        "calc_spacing_effective": AttrDef(
            doc=(
                "The approximate mean spacing actually used to infer the target "
                "sample count, including automatically resolved spacing."
            ),
            kind="calc",
        ),
        "calc_sample_count_target": AttrDef(
            doc=(
                "Resolved target number of sample points inferred from the current "
                "surface area and opts spacing before candidate oversampling."
            ),
            kind="calc",
        ),
        "calc_surface_points": AttrDef(
            doc="Point coordinates of the cleaned input surface as an (N, 3) array.",
            kind="calc",
        ),
        "calc_surface_normals": AttrDef(
            doc=(
                "Resolved surface normals associated with calc_surface_points as "
                "an (N, 3) array."
            ),
            kind="calc",
        ),
        "calc_sample_points": AttrDef(
            doc=(
                "The final sampled point coordinates exposed by this host as an "
                "(N, 3) array."
            ),
            kind="calc",
        ),
        "calc_sample_cell_ids": AttrDef(
            doc=(
                "Closest triangulated-surface cell ID for each sample point, or "
                "UNSET when is_calc_sample_normals is false."
            ),
            kind="calc",
        ),
        "calc_sample_barycentric": AttrDef(
            doc=(
                "Triangle barycentric coordinates for each sample point, or UNSET "
                "when is_calc_sample_normals is false."
            ),
            kind="calc",
        ),
        "calc_sample_normals": AttrDef(
            doc=(
                "Unit normals interpolated from triangle-vertex normals at sample "
                "locations, or UNSET when is_calc_sample_normals is false."
            ),
            kind="calc",
        ),
        "calc_nearest_distance_mean": AttrDef(
            doc=(
                "Mean nearest-neighbor sample distance, or UNSET when spacing "
                "statistics are disabled. NaN when only one point is sampled."
            ),
            kind="calc",
        ),
        "calc_nearest_distance_min": AttrDef(
            doc=(
                "Minimum nearest-neighbor sample distance, or UNSET when spacing "
                "statistics are disabled. NaN when only one point is sampled."
            ),
            kind="calc",
        ),
        "calc_nearest_distance_std": AttrDef(
            doc=(
                "Standard deviation of nearest-neighbor sample distances, or UNSET "
                "when spacing statistics are disabled. NaN for one sample point."
            ),
            kind="calc",
        ),
        "impl_surface_signature": AttrDef(
            doc="Internal identity signature for the prepared surface cache.",
            kind="impl",
        ),
        "impl_sampling_signature": AttrDef(
            doc="Internal option signature for the sampled-point cache.",
            kind="impl",
        ),
        "impl_is_sample_normals_enabled": AttrDef(
            doc="Internal cached state of the optional sample-normal layer.",
            kind="impl",
        ),
        "impl_is_spacing_statistics_enabled": AttrDef(
            doc="Internal cached state of the optional spacing-statistics layer.",
            kind="impl",
        ),
        "field": AttrDef(
            doc="The interpolated field object attached to this surface sampling.",
            kind="relation",
            is_weak_by_default=True,
        ),
        "result": AttrDef(
            doc="Read-only: Alias of `calc_sample_points`.",
            kind="property",
        ),
    }
    # fmt: on

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
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
        object.__setattr__(self, "calc_spacing_effective", 0.0)
        object.__setattr__(self, "calc_sample_count_target", 0)
        object.__setattr__(self, "calc_surface_points", np.empty((0, 3), dtype=float))
        object.__setattr__(self, "calc_surface_normals", np.empty((0, 3), dtype=float))
        object.__setattr__(self, "calc_sample_points", np.empty((0, 3), dtype=float))
        object.__setattr__(self, "calc_sample_cell_ids", UNSET)
        object.__setattr__(self, "calc_sample_barycentric", UNSET)
        object.__setattr__(self, "calc_sample_normals", UNSET)
        object.__setattr__(self, "calc_nearest_distance_mean", UNSET)
        object.__setattr__(self, "calc_nearest_distance_min", UNSET)
        object.__setattr__(self, "calc_nearest_distance_std", UNSET)
        object.__setattr__(self, "impl_surface_signature", UNSET)
        object.__setattr__(self, "impl_sampling_signature", UNSET)
        object.__setattr__(self, "impl_is_sample_normals_enabled", UNSET)
        object.__setattr__(self, "impl_is_spacing_statistics_enabled", UNSET)

        self.opts.act_finalize(defaults=self.opts_defaults)
        self._helper_commit_apply_opts(is_reapply_opts=True)

    @property
    def result(self):
        """Return the sampled point coordinates."""
        return self.calc_sample_points

    # -------------------- cache signatures --------------------

    def _helper_make_surface_signature(self) -> int:
        """Return the identity signature used for normalized raw surfaces."""
        return id(self.raw_surface)

    def _helper_make_sampling_signature(self) -> tuple[Any, ...]:
        """Return the option signature that determines sampled point positions."""
        return (
            self.opts.spacing,
            self.opts.seed,
            self.opts.oversample,
            self.opts.relax_steps,
            self.opts.k_neighbors,
            self.opts.default_sample_count_target,
            self.opts.max_sample_count,
        )

    # -------------------- cache update layers --------------------

    def _helper_update_surface_cache(self) -> None:
        """Rebuild cleaned surface geometry and its basic derived arrays."""
        surface_clean = _helper_prepare_surface(self.raw_surface)
        surface_points = as_readonly_array(surface_clean.points, dtype=None, copy=False)
        surface_normals = as_readonly_array(
            surface_clean.point_data["Normals"], dtype=None, copy=False
        )

        object.__setattr__(self, "calc_surface_clean", surface_clean)
        object.__setattr__(self, "calc_surface_area", float(surface_clean.area))
        object.__setattr__(self, "calc_surface_points", surface_points)
        object.__setattr__(self, "calc_surface_normals", surface_normals)

    def _helper_update_sampling_cache(self, *, logger) -> int:
        """Regenerate sampled points from the current surface and sampling opts."""
        surface_area = self.calc_surface_area
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
        if self.opts.max_sample_count is not None and sample_count_target > int(
            self.opts.max_sample_count
        ):
            candidate_count_requested = int(self.opts.oversample) * sample_count_target
            raise ValueError(
                f"Requested spacing {spacing_effective:.6g} on a surface with area "
                f"{surface_area:.6g} implies {sample_count_target} sample points and "
                f"at least {candidate_count_requested} candidates, exceeding "
                f"max_sample_count={int(self.opts.max_sample_count)}. Increase the "
                "spacing, raise max_sample_count, or set max_sample_count=None to "
                "disable this safety limit."
            )
        if sample_count_target > 10_000:
            logger.warning(
                f"Sampling {sample_count_target} points may be slow because the "
                "current farthest-point selection scales approximately quadratically."
            )

        candidate_count = max(
            int(self.opts.oversample) * sample_count_target,
            sample_count_target + 1,
        )
        rng = np.random.default_rng(int(self.opts.seed))
        candidate_points = _helper_sample_triangle_candidates(
            self.calc_surface_clean,
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
            self.calc_surface_clean,
            relax_steps=int(self.opts.relax_steps),
            k_neighbors=int(self.opts.k_neighbors),
        )

        object.__setattr__(self, "calc_spacing_effective", spacing_effective)
        object.__setattr__(self, "calc_sample_count_target", sample_count_target)
        object.__setattr__(
            self,
            "calc_sample_points",
            as_readonly_array(sample_points, dtype=None, copy=False),
        )
        return candidate_count

    def _helper_update_sample_normals_cache(self) -> None:
        """Compute or clear optional triangle-location and sample-normal outputs."""
        if not self.opts.is_calc_sample_normals:
            object.__setattr__(self, "calc_sample_cell_ids", UNSET)
            object.__setattr__(self, "calc_sample_barycentric", UNSET)
            object.__setattr__(self, "calc_sample_normals", UNSET)
            return

        sample_cell_ids, sample_barycentric, sample_normals = (
            _helper_calc_sample_surface_geometry(
                self.calc_sample_points,
                self.calc_surface_clean,
            )
        )
        object.__setattr__(
            self,
            "calc_sample_cell_ids",
            as_readonly_array(sample_cell_ids, dtype=None, copy=False),
        )
        object.__setattr__(
            self,
            "calc_sample_barycentric",
            as_readonly_array(sample_barycentric, dtype=None, copy=False),
        )
        object.__setattr__(
            self,
            "calc_sample_normals",
            as_readonly_array(sample_normals, dtype=None, copy=False),
        )

    def _helper_update_spacing_statistics_cache(self) -> None:
        """Compute or clear optional nearest-neighbor spacing statistics."""
        if not self.opts.is_calc_spacing_statistics:
            object.__setattr__(self, "calc_nearest_distance_mean", UNSET)
            object.__setattr__(self, "calc_nearest_distance_min", UNSET)
            object.__setattr__(self, "calc_nearest_distance_std", UNSET)
            return

        nearest_mean, nearest_min, nearest_std = _helper_calc_spacing_statistics(
            self.calc_sample_points
        )
        object.__setattr__(self, "calc_nearest_distance_mean", nearest_mean)
        object.__setattr__(self, "calc_nearest_distance_min", nearest_min)
        object.__setattr__(self, "calc_nearest_distance_std", nearest_std)

    # ==================== OVERRIDE ====================
    # SurfaceSampling overrides HostBase._helper_commit_apply_opts_main to map
    # finalized opts onto independently invalidated surface, sampling, and
    # optional-derived caches.
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

        surface_signature = self._helper_make_surface_signature()
        sampling_signature = self._helper_make_sampling_signature()
        is_surface_dirty = surface_signature != self.impl_surface_signature
        is_sampling_dirty = (
            is_surface_dirty or sampling_signature != self.impl_sampling_signature
        )
        is_sample_normals_dirty = (
            is_sampling_dirty
            or self.opts.is_calc_sample_normals != self.impl_is_sample_normals_enabled
        )
        is_spacing_statistics_dirty = (
            is_sampling_dirty
            or self.opts.is_calc_spacing_statistics
            != self.impl_is_spacing_statistics_enabled
        )

        if is_surface_dirty:
            self._helper_update_surface_cache()
            object.__setattr__(self, "impl_surface_signature", surface_signature)

        candidate_count = None
        if is_sampling_dirty:
            candidate_count = self._helper_update_sampling_cache(logger=logger)
            object.__setattr__(self, "impl_sampling_signature", sampling_signature)

        if is_sample_normals_dirty:
            self._helper_update_sample_normals_cache()
            object.__setattr__(
                self,
                "impl_is_sample_normals_enabled",
                self.opts.is_calc_sample_normals,
            )

        if is_spacing_statistics_dirty:
            self._helper_update_spacing_statistics_cache()
            object.__setattr__(
                self,
                "impl_is_spacing_statistics_enabled",
                self.opts.is_calc_spacing_statistics,
            )

        if is_sampling_dirty:
            logger.info(
                f"Resampled {self.name!r}: area={self.calc_surface_area:.6g}, "
                f"spacing_effective={self.calc_spacing_effective:.6g}, "
                f"target_count={self.calc_sample_count_target}, "
                f"candidate_count={candidate_count}."
            )
            if self.field:
                self.field.act_refresh()
