from dataclasses import dataclass
from typing import Callable, Sequence, Any, Mapping, ClassVar
import numpy as np
import pyvista as pv
from types import MappingProxyType

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import UNSET, Unset, as_bool, as_Number
from .plot_figure import PlotFigure
from .glyph import OptsGlyph, PlotGlyph
from Nematics3D.general import closest_point_on_polyline, fmt_value
from .qt.interact_tube import InteractTube
from Nematics3D.classes.host_base import HostBase

#! light dark pbr

#! info log extra attr
# 1 del
#! orphan figure

#! test
#! color invalid


@dataclass(slots=True, repr=False)
class OptsTube(OptsGlyph):

    # --- Geometry & Topology (Tube-specific) ---
    is_capping: bool | Unset = UNSET
    smooth_iter: int | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        **dict(OptsGlyph.__attrs__),
        "is_capping": "Whether to close the ends of the tube.",
        "smooth_iter": "Path smoothing iterations to remove jagged edges.",
    }

    _validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        **dict(OptsGlyph._validators),
        "is_capping": lambda v, d: as_bool(v, name=d),
        "smooth_iter": lambda v, d: as_Number(
            v, name=d, is_int=True, value_range=(0, 1000), bounded=True
        ),
    }

    _DEFAULTS_FROZEN: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            **dict(OptsGlyph._DEFAULTS_FROZEN),
            "is_capping": True,
            "smooth_iter": 0,
        }
    )


# PlotTube inherits the generic glyph pipeline but adds polyline-specific raw
# topology and build logic.
#
# Subclasses should keep the line-index contract aligned with `raw_coords`,
# route extra raw inputs through the pre-opts commit stage, and preserve the
# polyline-specific pick semantics when overriding geometry helpers.
class PlotTube(PlotGlyph):
    """
    PlotTube visualizes one or more connected line paths as tube geometry.

    For normal use, provide the centerline coordinates and optionally a
    `line_index` array to split the points into multiple disconnected paths.
    Visual settings can be read from `tube.opts`, changed through
    `tube.opts.<name> = value` or `tube.act_commit(...)`, and inspected with
    `tube.show_modifiable_attrs()`.
    """

    __attrs__ = {
        **dict(PlotGlyph.__attrs__),
        "raw_name": "The name identifier of the PlotTube instance",
        "raw_line_index": "Optional polyline membership indices.",
        "_calc_line_index": "The effective polyline membership indices used for the current glyph build after clip-mode preprocessing.",
        "_calc_keep_index": "Indices of raw centerline points kept after center-based point filtering.",
    }

    __slots__ = tuple(
        k
        for k, v in __attrs__.items()
        if not v.startswith("Property:") and k not in PlotGlyph.__slots__
    )
    _impl_attrs_reapply_opts_after_raw = (
        PlotGlyph._impl_attrs_reapply_opts_after_raw | {"line_index"}
    )

    # ==================== OVERRIDE ====================
    # PlotTube overrides PlotGlyph.__init__ only to accept
    # and validate the tube-specific raw field `line_index`.
    # ==================================================
    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        coords: np.ndarray,
        name: str | None = None,
        name_replace: str = "line",
        category: str = "tube",
        figure: PlotFigure | None = None,
        opts: OptsTube | None = None,
        line_index: Sequence | None = None,
        clip_mode: str = "center",
        opts_defaults_override: Mapping[str, Any] | None = None,
        logger=None,
        **kwargs,
    ):

        super().__init__(
            coords=coords,
            opts_type=OptsTube,
            category=category,
            name=name,
            name_replace=name_replace,
            opts=opts,
            figure=figure,
            clip_mode=clip_mode,
            opts_defaults_override=opts_defaults_override,
            **kwargs,
        )

        object.__setattr__(self, "raw_line_index", None)
        object.__setattr__(self, "_calc_line_index", None)
        object.__setattr__(self, "_calc_keep_index", None)
        self._helper_commit_line_index({"line_index": line_index})

        self._helper_init_end()
        self.act_set_interact_func(lambda: InteractTube(self, self.fig).show())

    def _helper_check_index(self, line_index, name):
        if line_index is None:
            return None
        try:
            line_index = np.asarray(line_index, dtype=int)
            if line_index.ndim != 1 or len(line_index) != self.raw_coords.shape[0]:
                raise ValueError(
                    f"`line_index` is {name}. "
                    f"It must be a ({self.raw_coords.shape[0]},) array. "
                    f"Got shape {line_index.shape} instead."
                )
            return line_index
        except (ValueError, TypeError):
            raise

    def _helper_commit_line_index(self, kwargs):
        return HostBase._helper_commit_pop_raw(
            self,
            kwargs,
            "line_index",
            validator=self._helper_check_index,
            exception_msg="Invalid `line_index` input",
            recovery_msg="Set line_index=None in the following (no stop points within the tube)",
        )

    def _helper_sync_calc_line_index(self):
        idx = self.raw_line_index
        if idx is None:
            object.__setattr__(self, "_calc_line_index", None)
        else:
            object.__setattr__(
                self,
                "_calc_line_index",
                np.asarray(idx, dtype=int).copy(),
            )

    def _helper_iter_line_ranges(self):
        if self.raw_line_index is None or len(np.unique(self.raw_line_index)) == 1:
            return [(0, len(self.raw_coords))]

        breaks = np.nonzero(self.raw_line_index[1:] != self.raw_line_index[:-1])[0] + 1
        starts = np.r_[0, breaks]
        ends = np.r_[breaks, len(self.raw_line_index)]
        return list(zip(starts, ends))

    # ==================== OVERRIDE ====================
    # PlotTube overrides PlotGlyph._helper_bound_coords because
    # tubes can first filter centerline points against bounds and then
    # build the tube mesh from the surviving centerline segments.
    # ==================================================
    def _helper_bound_coords(self):
        bounds = self.bounds
        if bounds is None:
            self._helper_sync_calc_line_index()
            keep_index = np.arange(len(self.raw_coords), dtype=int)
            object.__setattr__(self, "_calc_keep_index", keep_index)
            return self.raw_coords.copy()

        axis1 = np.asarray(bounds.opts.axis1, dtype=float)
        axis2 = np.asarray(bounds._calc_axis2, dtype=float)
        axis3 = np.asarray(bounds._calc_axis3, dtype=float)
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
        mask_keep = mask_inside if self.opts.is_clip_inside else ~mask_inside

        coords_segments = []
        keep_segments = []
        for start, end in self._helper_iter_line_ranges():
            keep_range = np.nonzero(mask_keep[start:end])[0] + start
            if len(keep_range) == 0:
                continue

            breaks = np.nonzero(np.diff(keep_range) != 1)[0] + 1
            seg_starts = np.r_[0, breaks]
            seg_ends = np.r_[breaks, len(keep_range)]
            for s, e in zip(seg_starts, seg_ends):
                raw_idx = keep_range[s:e]
                if len(raw_idx) < 2:
                    continue
                keep_segments.append(raw_idx.astype(int, copy=False))
                coords_segments.append(self.raw_coords[raw_idx])

        if len(coords_segments) == 0:
            object.__setattr__(self, "_calc_line_index", None)
            object.__setattr__(self, "_calc_keep_index", np.empty((0,), dtype=int))
            return np.empty((0, 3), dtype=float)

        if len(coords_segments) == 1:
            object.__setattr__(self, "_calc_line_index", None)
            object.__setattr__(self, "_calc_keep_index", keep_segments[0])
            return np.asarray(coords_segments[0], dtype=float)

        coords_all = np.vstack(coords_segments).astype(float, copy=False)
        keep_index = np.concatenate(keep_segments).astype(int, copy=False)
        line_index = np.concatenate(
            [np.full(len(seg), i, dtype=int) for i, seg in enumerate(coords_segments)]
        )
        object.__setattr__(self, "_calc_line_index", line_index)
        object.__setattr__(self, "_calc_keep_index", keep_index)
        return coords_all

    # ==================== OVERRIDE ====================
    # PlotTube overrides PlotGlyph._helper_build_poly because
    # a tube may represent one or multiple disconnected polylines,
    # so the PolyData topology cannot reuse the glyph default.
    # ==================================================
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_build_poly(self, logger=None):

        points = self._calc_coords
        if self.state_clip_mode == "center":
            idx = self._calc_line_index
        else:
            self._helper_sync_calc_line_index()
            object.__setattr__(
                self,
                "_calc_keep_index",
                np.arange(len(self.raw_coords), dtype=int),
            )
            idx = self._calc_line_index

        if len(points) < 2:
            poly = pv.PolyData(np.asarray(points, dtype=float))
            object.__setattr__(self, "_calc_poly", poly)
            self._helper_set_poly(poly)
            return

        # Decide whether to treat the input as a single continuous polyline
        is_use_multi = (idx is None) or (len(np.unique(idx)) == 1)
        if is_use_multi:
            poly = pv.MultipleLines(points)
        else:
            breaks = np.nonzero(idx[1:] != idx[:-1])[0] + 1
            starts = np.r_[0, breaks]
            ends = np.r_[breaks, len(idx)]

            chunks = []
            for s, e in zip(starts, ends):
                k = e - s
                if k < 2:
                    logger.warning(
                        f"Detect one invalid line segment with only one point at index={s}."
                        "This will not be plotted."
                    )
                    continue
                chunks.append(np.r_[k, np.arange(s, e, dtype=np.int64)])

            if len(chunks) == 0:
                poly = pv.PolyData(np.asarray(points, dtype=float))
            else:
                lines = np.concatenate(chunks).astype(np.int64)
                poly = pv.PolyData(points, lines=lines)

        if self.opts.smooth_iter > 0 and poly.n_points > 1:
            poly = poly.smooth(n_iter=self.opts.smooth_iter)

        object.__setattr__(self, "_calc_poly", poly)
        self._helper_set_poly(poly)

    # ==================== OVERRIDE ====================
    # PlotTube overrides PlotGlyph._helper_set_poly so center-based clipping
    # can directly filter pointwise visual data with the kept center indices.
    # ==================================================
    def _helper_set_poly(self, poly):
        if self.state_clip_mode != "center":
            return super()._helper_set_poly(poly)

        if poly.n_points == 0:
            return

        keep_index = getattr(self, "_calc_keep_index", None)
        if keep_index is None:
            keep_index = np.arange(len(self.raw_coords), dtype=int)

        radius = self._calc_radius[keep_index]
        opacity = self._calc_opacity[keep_index]
        scalars = self._calc_scalars[keep_index]
        color = self._calc_color[keep_index]

        if len(radius) > 0:
            poly.point_data["radius"] = radius
        poly.point_data["opacity"] = opacity
        poly.point_data["scalars"] = scalars
        rgba_values = np.hstack([color, opacity.reshape(-1, 1)])
        poly.point_data["rgba"] = rgba_values

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_build_mesh(self, logger=None):
        """
        Internal: generate tube geometry from the prepared polyline dataset.
        """

        poly = self._calc_poly
        if poly.n_points < 2 or "radius" not in poly.point_data:
            return pv.PolyData()

        mesh = poly.tube(
            scalars="radius",
            n_sides=self.opts.sides,
            capping=self.opts.is_capping,
            absolute=True,
        )

        return mesh

    # ==================== OVERRIDE ====================
    # PlotTube overrides HostBase/PlotGlyph pre-opts handling only
    # to route the extra raw field `line_index` through the new
    # commit-pop-raw validator path and trigger opts re-apply.
    # ==================================================
    def _helper_commit_pre_opts(self, kwargs):
        kwargs_applied, is_reapply_opts = super()._helper_commit_pre_opts(kwargs)
        kwargs_applied_line, is_reapply_opts_line = self._helper_commit_line_index(
            kwargs
        )
        return (
            kwargs_applied | kwargs_applied_line,
            is_reapply_opts or is_reapply_opts_line,
        )

    # ==================== OVERRIDE ====================
    # PlotTube overrides PlotGlyph._helper_resolve_pick to report
    # tube-specific information such as normalized arc position
    # and, when available, the local tangent direction.
    # ==================================================
    def _helper_resolve_pick(self, picked_point):

        pos_close, msg, idx = super()._helper_resolve_pick(picked_point)
        x_param = idx / len(self.raw_coords) * 100
        msg_head = (
            f"The closest point on the tube is {fmt_value(pos_close)}, where: \n"
            f"The normalized position along the tube is {x_param:2f} \n"
        )
        try:
            smooth = self.owner.owner
            tgt = smooth.act_calc_tgt(x_param)
            msg_head += f"Local tangent: {fmt_value(tgt)} \n"
        except:
            pass
        msg = msg_head + msg

        pos = closest_point_on_polyline(picked_point, self.raw_coords)

        return pos, msg, idx
