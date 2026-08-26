"""Vector glyph visuals built on the shared PlotGlyph pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping, Sequence

import numpy as np
import pyvista as pv

from nematics3d.datatypes import UNSET, Unset, as_number, as_points, as_str
from nematics3d.general import fmt_value
from nematics3d.logging_decorator import logging_and_warning_decorator

from ..bounds import BoundsData
from ..class_base import AttrDef
from ..host_base import HostBase
from .glyph import OptsGlyph, PlotGlyph, _as_resolver_source_or_none
from .plot_figure import FigureData

LengthMode = float | Callable | Sequence


def _as_positive_resolver_mode(value, name):
    """Validate a positive scalar/array resolver input, or pass callables through."""
    if callable(value):
        return value
    if np.isscalar(value):
        return as_number(value, name=name, value_range=(1e-12, np.inf))

    arr = np.asarray(value, dtype=float)
    if np.any(arr <= 0):
        raise ValueError(f"{name} must contain only positive values.")
    return value


@dataclass(slots=True, repr=False)
class OptsVector(OptsGlyph):
    """
    Visual configuration object for `PlotVector`.

    `OptsVector` stores vector-arrow appearance and topology controls. The
    total vector length uses the same flexible resolver model as other glyphs:

    - `length` may be one shared value, one value per vector, or a callable
      resolver.
    - `radius` is inherited from `OptsGlyph` and represents shaft radius. It
      may also be one shared value, one value per vector, or a callable
      resolver.
    - `anchor` decides whether the input `coords` represent vector tails or
      vector centers.

    Arrow tips intentionally stay simpler. Their dimensions are derived from
    ratios:

    - `tip_length_fraction` is `tip_length / length`
    - `tip_radius_ratio` multiplies the resolved `radius`

    This keeps per-vector vector-field scaling expressive while avoiding four
    separate fully-resolved geometry fields for routine vector plots.
    """

    # --- Geometry & Topology (Vector-specific) ---
    length: LengthMode | Unset = UNSET
    tip_length_fraction: float | Unset = UNSET
    tip_radius_ratio: float | Unset = UNSET
    anchor: str | Unset = UNSET

    # fmt: off
    __attrs__: ClassVar[Mapping[str, str]] = {
        **dict(OptsGlyph.__attrs__),
        "length":           "The total display length of vectors.",
        "tip_length_fraction": "Tip length as a fraction of total vector length.",
        "tip_radius_ratio": "Tip radius as a multiplier of resolved shaft radius.",
        "anchor":           "How input coords are interpreted: 'tail' or 'center'.",
    }

    impl_validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        **dict(OptsGlyph.impl_validators),
        "length": lambda v, d: _as_positive_resolver_mode(v, d),
        "radius": lambda v, d: _as_positive_resolver_mode(v, d),
        "tip_length_fraction": lambda v, d: as_number(
            v, name=d, value_range=(1e-12, 1), is_clipped=True
        ),
        "tip_radius_ratio": lambda v, d: as_number(
            v, name=d, value_range=(1e-12, np.inf)
        ),
        "resolver_source": lambda v, d: as_str(
            v,
            name=d,
            pool=("coords", "orient", "orient_length"),
        ),
        "resolver_source_color": lambda v, d: _as_resolver_source_or_none(
            v, d, pool=("coords", "orient", "orient_length")
        ),
        "resolver_source_opacity": lambda v, d: _as_resolver_source_or_none(
            v, d, pool=("coords", "orient", "orient_length")
        ),
        "resolver_source_radius": lambda v, d: _as_resolver_source_or_none(
            v, d, pool=("coords", "orient", "orient_length")
        ),
        "resolver_source_scalars": lambda v, d: _as_resolver_source_or_none(
            v, d, pool=("coords", "orient", "orient_length")
        ),
        "anchor": lambda v, d: as_str(
            v,
            name=d,
            pool=("tail", "center"),
        ),
    }

    impl_defaults_frozen: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            **dict(OptsGlyph.impl_defaults_frozen),
            "length":           3,
            "radius":           0.3,
            "tip_length_fraction": 0.2,
            "tip_radius_ratio": 2.5,
            "resolver_source":  "orient_length",
            "anchor":           "center",
        }
    )
    # fmt: on


# PlotVector follows the PlotRod host shape, but separates shaft geometry from
# arrow-tip proportions.
class PlotVector(PlotGlyph):
    """
    Render one arrow-style vector at each input point.

    This first implementation establishes the managed host/opts surface and
    initialization path. Mesh construction is intentionally left as an empty
    placeholder so the class contract can settle before the arrow geometry is
    filled in.
    """

    # fmt: off
    __attr_defs__ = {
        "raw_orient": AttrDef(
            doc="The orientation vectors of plotted vectors.",
            kind="raw",
            validator=lambda v, d: as_points(v, name=d),
            is_reapply_opts_after_raw=True,
        ),
        "calc_length": AttrDef(
            doc="The resolved per-vector total display length array.",
            kind="calc",
        ),
        "calc_shaft_length": AttrDef(
            doc="The resolved per-vector shaft length array.",
            kind="calc",
        ),
        "calc_tip_length": AttrDef(
            doc="The resolved per-vector tip length array derived from length.",
            kind="calc",
        ),
        "calc_tip_radius": AttrDef(
            doc="The resolved per-vector tip radius array derived from radius.",
            kind="calc",
        ),
        "calc_orient_unit": AttrDef(
            doc="The unit direction vectors used for plotting vector geometry.",
            kind="calc",
        ),
        "calc_tail": AttrDef(
            doc="The resolved per-vector tail coordinates.",
            kind="calc",
        ),
        "calc_shaft_end": AttrDef(
            doc="The resolved per-vector shaft-end coordinates.",
            kind="calc",
        ),
        "calc_tip_end": AttrDef(
            doc="The resolved per-vector tip-end coordinates.",
            kind="calc",
        ),
        "calc_keep_index": AttrDef(
            doc="Indices of raw vector anchors kept after center-based point filtering.",
            kind="calc",
        ),
    }
    # fmt: on

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
        and name not in HostBase.__slots__
    )

    _pending_resolution_attrs: Sequence[str] = PlotGlyph._pending_resolution_attrs + [
        "length"
    ]

    # -------------------------------
    # Initialization
    # -------------------------------

    # ==================== OVERRIDE ====================
    # PlotVector overrides PlotGlyph.__init__ because it must accept
    # vector-specific orientation data before the generic glyph
    # initialization and opts resolution are performed.
    # ==================================================
    def __init__(
        self,
        coords: np.ndarray,
        orient: np.ndarray,
        name: str = "vector",
        name_replace: str = "vector",
        category: str = "vectors",
        figure: FigureData | None = None,
        opts: OptsVector | None = None,
        bounds: BoundsData | None = None,
        clip_mode: str = "center",
        is_clip_inside: bool = True,
        opts_defaults_override: Mapping[str, Any] | None = None,
        **kwargs,
    ):

        orient = (
            type(self)
            .__attr_defs__["raw_orient"]
            .validator(
                orient,
                type(self).__attr_defs__["raw_orient"].doc,
            )
        )
        object.__setattr__(self, "raw_orient", orient)

        super().__init__(
            coords=coords,
            opts_type=OptsVector,
            category=category,
            name=name,
            name_replace=name_replace,
            opts=opts,
            figure=figure,
            bounds=bounds,
            clip_mode=clip_mode,
            is_clip_inside=is_clip_inside,
            opts_defaults_override=opts_defaults_override,
            **kwargs,
        )

        if len(self.raw_orient) != len(self.raw_coords):
            raise ValueError(
                f"There are {len(self.raw_orient)} points for orientation, "
                f"while {len(self.raw_coords)} points for positions."
            )

        object.__setattr__(self, "calc_keep_index", None)

        from .qt.interact_vector import InteractVector

        self.act_set_interact_func(lambda: InteractVector.show_once(self, self.fig))

        self._helper_init_end()

    # -------------------------------
    # Resolver helpers
    # -------------------------------

    # ==================== OVERRIDE ====================
    # PlotVector overrides PlotGlyph._helper_get_resolver_source to add vector
    # orientation and vector-length resolver sources.
    # ==================================================
    def _helper_get_resolver_source_name(self, attr_name=None):
        source_name = None
        if attr_name is not None:
            override_attr = self._resolver_source_override_attr_names.get(attr_name)
            if override_attr is not None:
                source_name = getattr(self.opts, override_attr, None)
        if source_name is None:
            source_name = self.opts.resolver_source
        return as_str(
            source_name,
            name="glyph resolver source",
            pool=("coords", "orient", "orient_length"),
        )

    def _helper_get_resolver_source(self, attr_name=None):
        source_name = self._helper_get_resolver_source_name(attr_name)
        if source_name == "orient":
            return self.raw_orient
        if source_name == "orient_length":
            return np.linalg.norm(self.raw_orient, axis=1)
        return super()._helper_get_resolver_source(attr_name)

    def _helper_sync_derived_geometry(self):
        """Update shaft/tip derived arrays from resolved length and radius."""
        if not hasattr(self, "calc_length") or not hasattr(self, "calc_radius"):
            return

        tip_length_fraction = float(self.opts.tip_length_fraction)
        shaft_length_fraction = 1.0 - tip_length_fraction

        length = np.asarray(self.calc_length, dtype=np.float32)
        radius = np.asarray(self.calc_radius, dtype=np.float32)

        object.__setattr__(self, "calc_shaft_length", length * shaft_length_fraction)
        object.__setattr__(self, "calc_tip_length", length * tip_length_fraction)
        object.__setattr__(
            self,
            "calc_tip_radius",
            radius * float(self.opts.tip_radius_ratio),
        )

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_get_orient_unit(self, logger=None):
        orient = np.asarray(self.raw_orient, dtype=float)
        orient_norm = np.linalg.norm(orient, axis=1, keepdims=True)

        mask = orient_norm.squeeze() > 1e-12
        orient_unit = np.zeros_like(orient, dtype=float)
        orient_unit[mask] = orient[mask] / orient_norm[mask]

        if not np.all(mask):
            n_bad = np.count_nonzero(~mask)
            logger.warning(
                f"{n_bad} vector(s) have near-zero orientation norm (<= 1e-12). "
                "Their generated geometry will be degenerate."
            )

        return orient_unit

    def _helper_sync_vector_key_points(self):
        """Update vector tail, shaft-end, and tip-end coordinates."""
        if (
            not hasattr(self, "calc_length")
            or not hasattr(self, "calc_shaft_length")
            or not hasattr(self, "calc_tip_length")
        ):
            return

        orient_unit = self._helper_get_orient_unit()
        length = np.asarray(self.calc_length, dtype=float).reshape(-1, 1)
        shaft_length = np.asarray(self.calc_shaft_length, dtype=float).reshape(-1, 1)

        if self.opts.anchor == "tail":
            tail = np.asarray(self.raw_coords, dtype=float).copy()
        else:
            tail = np.asarray(self.raw_coords, dtype=float) - 0.5 * length * orient_unit

        shaft_end = tail + shaft_length * orient_unit
        tip_end = tail + length * orient_unit

        object.__setattr__(self, "calc_orient_unit", orient_unit)
        object.__setattr__(self, "calc_tail", tail)
        object.__setattr__(self, "calc_shaft_end", shaft_end)
        object.__setattr__(self, "calc_tip_end", tip_end)

    # ==================== OVERRIDE ====================
    # PlotVector overrides PlotGlyph._helper_resolver_spec so resolved total
    # length/radius immediately refresh the derived shaft and tip dimensions.
    # ==================================================
    def _helper_resolver_spec(self, attr_name, attr_value=None):
        result = super()._helper_resolver_spec(attr_name, attr_value=attr_value)
        if attr_name == "length":
            self._helper_sync_derived_geometry()
            self._helper_sync_vector_key_points()
        elif attr_name == "radius":
            self._helper_sync_derived_geometry()
        return result

    # -------------------------------
    # Commit pipeline
    # -------------------------------

    # ==================== OVERRIDE ====================
    # PlotVector overrides PlotGlyph._helper_commit_apply_opts_main because
    # tip ratios and anchor are scalar vector-shape controls that still require
    # the glyph geometry to be rebuilt.
    # ==================================================
    def _helper_commit_apply_opts_main(self, is_reapply_opts=False, **kwargs):
        vector_shape_keys = ("tip_length_fraction", "tip_radius_ratio", "anchor")
        is_vector_shape_update = False
        for key in vector_shape_keys:
            if key not in kwargs:
                continue
            object.__setattr__(self.opts, key, kwargs.pop(key))
            is_vector_shape_update = True

        if is_vector_shape_update:
            self._helper_sync_derived_geometry()
            self._helper_sync_vector_key_points()
            is_reapply_opts = True

        return super()._helper_commit_apply_opts_main(
            is_reapply_opts=is_reapply_opts,
            **kwargs,
        )

    # -------------------------------
    # Center clipping and polydata preparation
    # -------------------------------

    def _helper_expand_endpoint_values(self, values, keep_index=None):
        values = np.asarray(values)
        if keep_index is not None:
            keep_index = np.asarray(keep_index, dtype=int)
            values = values[keep_index]
        return np.repeat(values, 2, axis=0)

    # ==================== OVERRIDE ====================
    # PlotVector overrides PlotGlyph._helper_bound_coords because vectors can
    # center-clip by filtering their raw anchor points directly.
    # ==================================================
    def _helper_bound_coords(self):
        bounds = self._helper_get_bounds_effective()
        if bounds is None:
            keep_index = np.arange(len(self.raw_coords), dtype=int)
            object.__setattr__(self, "calc_keep_index", keep_index)
            return self.raw_coords.copy()

        axis1 = np.asarray(bounds.opts.axis1, dtype=float)
        axis2 = np.asarray(bounds.calc_axis2, dtype=float)
        axis3 = np.asarray(bounds.calc_axis3, dtype=float)
        length1 = float(bounds.opts.length1)
        length2 = length1 if bounds.opts.length2 is None else float(bounds.opts.length2)
        length3 = length1 if bounds.opts.length3 is None else float(bounds.opts.length3)
        origin = np.asarray(bounds.opts.origin, dtype=float)

        if bounds.opts.alignment == "min_corner":
            origin_min_corner = origin
        else:
            origin_min_corner = origin - 0.5 * (
                length1 * axis1 + length2 * axis2 + length3 * axis3
            )

        basis = np.column_stack([axis1, axis2, axis3])
        coords_local = (self.raw_coords - origin_min_corner) @ basis
        tol = 1e-10
        upper = np.array([length1, length2, length3], dtype=float)
        mask_inside = np.all(
            (coords_local >= -tol) & (coords_local <= upper + tol), axis=1
        )
        mask_keep = mask_inside if self.state_is_clip_inside else ~mask_inside
        keep_index = np.nonzero(mask_keep)[0].astype(int, copy=False)
        object.__setattr__(self, "calc_keep_index", keep_index)
        return self.raw_coords[keep_index]

    # ==================== OVERRIDE ====================
    # PlotVector overrides PlotGlyph._helper_build_poly because vector shafts
    # are represented by oriented line segments before tube meshing.
    # ==================================================
    def _helper_build_poly(self):
        keep_index = getattr(self, "calc_keep_index", None)
        if keep_index is None:
            keep_index = np.arange(len(self.raw_coords), dtype=int)

        if not hasattr(self, "calc_tail") or not hasattr(self, "calc_shaft_end"):
            self._helper_sync_vector_key_points()

        keep_index = np.asarray(keep_index, dtype=int)
        n_vectors = len(keep_index)
        if n_vectors == 0:
            poly = pv.PolyData(np.empty((0, 3), dtype=float))
            object.__setattr__(self, "calc_poly", poly)
            self._helper_set_poly(poly)
            return

        tail = np.asarray(self.calc_tail, dtype=float)[keep_index]
        shaft_end = np.asarray(self.calc_shaft_end, dtype=float)[keep_index]

        endpoints = np.empty((2 * n_vectors, 3), dtype=float)
        endpoints[0::2] = tail
        endpoints[1::2] = shaft_end

        lines = np.empty((n_vectors, 3), dtype=np.int64)
        lines[:, 0] = 2
        lines[:, 1] = 2 * np.arange(n_vectors)
        lines[:, 2] = 2 * np.arange(n_vectors) + 1

        poly = pv.PolyData(endpoints, lines=lines.ravel())
        object.__setattr__(self, "calc_poly", poly)
        self._helper_set_poly(poly)

    # ==================== OVERRIDE ====================
    # PlotVector overrides PlotGlyph._helper_set_poly so shaft endpoints share
    # the same resolved per-vector visual data.
    # ==================================================
    def _helper_set_poly(self, poly):
        if poly.n_points == 0:
            return

        keep_index = getattr(self, "calc_keep_index", None)
        if keep_index is None:
            keep_index = np.arange(len(self.raw_coords), dtype=int)

        color = self._helper_expand_endpoint_values(self.calc_color, keep_index)
        opacity = self._helper_expand_endpoint_values(self.calc_opacity, keep_index)
        radius = self._helper_expand_endpoint_values(self.calc_radius, keep_index)
        scalars = self._helper_expand_endpoint_values(self.calc_scalars, keep_index)

        poly.point_data["radius"] = radius
        poly.point_data["opacity"] = opacity
        poly.point_data["scalars"] = scalars
        poly.point_data["rgba"] = np.hstack([color, opacity.reshape(-1, 1)])

    # ==================== OVERRIDE ====================
    # PlotVector overrides PlotGlyph._helper_build_mesh because vectors combine
    # a tube shaft with a manually generated cone tip.
    # ==================================================
    def _helper_build_shaft_mesh(self):
        poly = self.calc_poly
        if poly.n_points < 2 or "radius" not in poly.point_data:
            return pv.PolyData()

        mesh = poly.tube(
            scalars="radius",
            n_sides=self.opts.sides,
            absolute=True,
        )
        return mesh

    def _helper_make_perp_basis(self, direction):
        direction = np.asarray(direction, dtype=float)
        if np.linalg.norm(direction) <= 1e-12:
            return None, None

        helper = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(direction, helper))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=float)

        axis1 = np.cross(direction, helper)
        axis1_norm = np.linalg.norm(axis1)
        if axis1_norm <= 1e-12:
            return None, None
        axis1 /= axis1_norm

        axis2 = np.cross(direction, axis1)
        axis2_norm = np.linalg.norm(axis2)
        if axis2_norm <= 1e-12:
            return None, None
        axis2 /= axis2_norm

        return axis1, axis2

    def _helper_build_tip_mesh(self):
        keep_index = getattr(self, "calc_keep_index", None)
        if keep_index is None:
            keep_index = np.arange(len(self.raw_coords), dtype=int)
        keep_index = np.asarray(keep_index, dtype=int)

        if len(keep_index) == 0:
            return pv.PolyData()

        if not hasattr(self, "calc_tail") or not hasattr(self, "calc_tip_end"):
            self._helper_sync_vector_key_points()

        sides = int(self.opts.sides)
        angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
        cos_vals = np.cos(angles)
        sin_vals = np.sin(angles)

        direction = np.asarray(self.calc_orient_unit, dtype=float)[keep_index]
        base = np.asarray(self.calc_shaft_end, dtype=float)[keep_index]
        tip = np.asarray(self.calc_tip_end, dtype=float)[keep_index]
        radius = np.asarray(self.calc_tip_radius, dtype=float)[keep_index]

        helper = np.tile(np.array([1.0, 0.0, 0.0], dtype=float), (len(keep_index), 1))
        mask_parallel = np.abs(np.sum(direction * helper, axis=1)) > 0.9
        helper[mask_parallel] = np.array([0.0, 1.0, 0.0], dtype=float)

        axis1 = np.cross(direction, helper)
        axis1_norm = np.linalg.norm(axis1, axis=1, keepdims=True)
        mask_valid = axis1_norm[:, 0] > 1e-12
        if not np.any(mask_valid):
            return pv.PolyData()

        axis1 = axis1[mask_valid] / axis1_norm[mask_valid]
        direction = direction[mask_valid]
        axis2 = np.cross(direction, axis1)
        axis2 /= np.linalg.norm(axis2, axis=1, keepdims=True)

        base = base[mask_valid]
        tip = tip[mask_valid]
        radius = radius[mask_valid]

        raw_idx = keep_index[mask_valid]
        color = np.asarray(self.calc_color, dtype=np.float32)[raw_idx]
        opacity = np.asarray(self.calc_opacity, dtype=np.float32)[raw_idx]
        scalars = np.asarray(self.calc_scalars, dtype=np.float32)[raw_idx]

        n_vectors = len(raw_idx)
        n_points_per_vector = sides + 2
        n_faces_per_vector = 2 * sides

        ring_offsets = radius[:, None, None] * (
            cos_vals[None, :, None] * axis1[:, None, :]
            + sin_vals[None, :, None] * axis2[:, None, :]
        )
        ring_points = base[:, None, :] + ring_offsets

        points = np.empty((n_vectors, n_points_per_vector, 3), dtype=float)
        points[:, 0, :] = base
        points[:, 1, :] = tip
        points[:, 2:, :] = ring_points
        points = points.reshape(-1, 3)

        face_template = np.empty((n_faces_per_vector, 4), dtype=np.int64)
        for side_idx in range(sides):
            row_tip = 2 * side_idx
            row_base = row_tip + 1
            j0 = 2 + side_idx
            j1 = 2 + ((side_idx + 1) % sides)
            face_template[row_tip] = (3, 1, j0, j1)
            face_template[row_base] = (3, 0, j1, j0)

        vector_offsets = (
            np.arange(n_vectors, dtype=np.int64)[:, None, None] * n_points_per_vector
        )
        faces = np.broadcast_to(
            face_template[None, :, :], (n_vectors, n_faces_per_vector, 4)
        ).copy()
        faces[:, :, 1:] += vector_offsets
        faces = faces.reshape(-1)

        rgba = np.concatenate([color, opacity[:, None]], axis=1)
        point_rgba = np.repeat(rgba, n_points_per_vector, axis=0)
        point_opacity = np.repeat(opacity, n_points_per_vector)
        point_scalars = np.repeat(scalars, n_points_per_vector)

        mesh = pv.PolyData(points, faces=faces)
        mesh.point_data["opacity"] = point_opacity
        mesh.point_data["scalars"] = point_scalars
        mesh.point_data["rgba"] = point_rgba
        return mesh

    def _helper_set_merged_mesh_arrays(self, mesh, shaft_mesh, tip_mesh):
        """Restore shared point-data arrays after merging shaft and tip meshes."""
        if mesh.n_points == 0:
            return mesh

        points_merged = np.asarray(mesh.points)
        points_shaft = np.asarray(shaft_mesh.points)
        points_tip = np.asarray(tip_mesh.points)

        n_shaft = len(points_shaft)
        n_tip = len(points_tip)

        if (
            len(points_merged) == n_tip + n_shaft
            and np.allclose(points_merged[:n_tip], points_tip)
            and np.allclose(points_merged[n_tip:], points_shaft)
        ):
            sources = (tip_mesh, shaft_mesh)
        elif (
            len(points_merged) == n_shaft + n_tip
            and np.allclose(points_merged[:n_shaft], points_shaft)
            and np.allclose(points_merged[n_shaft:], points_tip)
        ):
            sources = (shaft_mesh, tip_mesh)
        else:
            raise RuntimeError(
                "Unexpected point ordering after merging vector shaft and tip meshes."
            )

        for name in ("opacity", "scalars", "rgba"):
            if any(name not in source.point_data for source in sources):
                continue
            mesh.point_data[name] = np.concatenate(
                [np.asarray(source.point_data[name]) for source in sources],
                axis=0,
            )
        return mesh

    def _helper_build_mesh(self):
        shaft_mesh = self._helper_build_shaft_mesh()
        tip_mesh = self._helper_build_tip_mesh()

        if shaft_mesh.n_points == 0:
            return tip_mesh
        if tip_mesh.n_points == 0:
            return shaft_mesh

        mesh = shaft_mesh.merge(tip_mesh, merge_points=False)
        self._helper_set_merged_mesh_arrays(mesh, shaft_mesh, tip_mesh)
        return mesh

    # -------------------------------
    # Picking
    # -------------------------------

    # ==================== OVERRIDE ====================
    # PlotVector overrides PlotGlyph._helper_resolve_pick to expose vector
    # direction and derived arrow geometry in pick reports.
    # ==================================================
    def _helper_resolve_pick(self, picked_point):
        pos, msg, idx = super()._helper_resolve_pick(picked_point)
        orient = np.asarray(self.raw_orient[idx], dtype=float)
        orient_length = float(np.linalg.norm(orient))

        msg_head = (
            f"Local orientation: {fmt_value(orient)} \n"
            f"Local orientation length: {orient_length:.6g} \n"
            f"Local display length: {fmt_value(self.calc_length[idx])} \n"
            f"Local shaft length: {fmt_value(self.calc_shaft_length[idx])} \n"
            f"Local tip length: {fmt_value(self.calc_tip_length[idx])} \n"
        )
        return pos, msg_head + msg, idx
