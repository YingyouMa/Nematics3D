from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence, Any, Mapping, ClassVar
import numpy as np
import pyvista as pv
from types import MappingProxyType

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import UNSET, Unset, as_str
from .plot_figure import FigureData, PlotFigure
from .glyph import OptsGlyph, PlotGlyph
from ..bounds import BoundsData
from .qt.interact_rod import InteractRod
from Nematics3D.datatypes import as_points
from Nematics3D.general import fmt_value

LengthMode = float | Callable | Sequence


@dataclass(slots=True, repr=False)
class OptsRod(OptsGlyph):

    # --- Geometry & Topology (Tube-specific) ---
    length: LengthMode | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        **dict(OptsGlyph.__attrs__),
        "length": "The length of rods",
    }

    _validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        **dict(OptsGlyph._validators),
        "resolver_source": lambda v, d: as_str(
            v,
            name=d,
            pool=("coords", "u_percent", "orient"),
        ),
    }

    _DEFAULTS_FROZEN: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            **dict(OptsGlyph._DEFAULTS_FROZEN),
            "length": 3,
            "radius": 0.3,
            "resolver_source": "orient",
        }
    )


# PlotRod inherits the generic glyph host but replaces the geometry path with
# rod-specific orientation and length handling.
#
# Subclasses must keep `raw_orient` aligned with `raw_coords`, update any
# endpoint-expanded arrays together, and be careful when overriding attribute
# access because several resolved arrays are intentionally repeated per rod.
class PlotRod(PlotGlyph):
    """
    PlotRod visualizes oriented rods centered at the provided coordinates.

    Normal users create rods from positions plus orientation vectors, then tune
    appearance and geometry through `rod.opts` or `rod.act_commit(...)`.
    Use `rod.show_modifiable_attrs()` to inspect configurable settings and
    `repr(rod)` for a compact summary of the plotted object.
    """

    __attrs__ = {
        **dict(PlotGlyph.__attrs__),
        "raw_name": "The name identifier of the PlotRod instance",
        "raw_orient": "The orientation of rods",
        "_calc_length": "The resolved per-point length array used for rods length.",
        "_calc_keep_index": "Indices of raw rod centers kept after center-based point filtering.",
    }
    __slots__ = tuple(
        k
        for k, v in __attrs__.items()
        if not v.startswith("Property:") and k not in PlotGlyph.__slots__
    )

    _pending_resolution_attrs: Sequence[str] = PlotGlyph._pending_resolution_attrs + [
        "length"
    ]
    _impl_attrs_reapply_opts_after_raw = (
        PlotGlyph._impl_attrs_reapply_opts_after_raw | {"orient"}
    )
    _impl_validators = {
        **PlotGlyph._impl_validators,
        "orient": lambda v, d: as_points(v, name=d),
    }

    # ==================== OVERRIDE ====================
    # PlotRod overrides PlotGlyph.__init__ because it must accept
    # rod-specific raw orientation data before the generic glyph
    # initialization and mesh setup are performed.
    # ==================================================
    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        coords: np.ndarray,
        orient: np.ndarray,
        name: str = "rod",
        name_replace: str = "rod",
        category: str = "rods",
        figure: FigureData | None = None,
        opts: OptsRod | None = None,
        bounds: BoundsData | None = None,
        clip_mode: str = "center",
        opts_defaults_override: Mapping[str, Any] | None = None,
        logger=None,
        **kwargs,
    ):

        orient = self.__class__._impl_validators["orient"](
            orient,
            self.show_attr_desc("raw_orient"),
        )
        object.__setattr__(self, "raw_orient", orient)

        super().__init__(
            coords=coords,
            opts_type=OptsRod,
            category=category,
            name=name,
            name_replace=name_replace,
            opts=opts,
            figure=figure,
            bounds=bounds,
            clip_mode=clip_mode,
            opts_defaults_override=opts_defaults_override,
            **kwargs,
        )

        if len(self.raw_orient) != len(self.raw_coords):
            raise ValueError(
                f"There are {len(self.raw_orient)} points for orientation, while {len(self.raw_coords)} points for positions."
            )

        object.__setattr__(self, "_calc_keep_index", None)

        self.act_set_interact_func(lambda: InteractRod(self, self.fig).show())

        self._helper_init_end()

    # ==================== OVERRIDE ====================
    # PlotRod overrides PlotGlyph._helper_get_resolver_source to add rod
    # orientation as a valid callable-resolver input source.
    # ==================================================
    def _helper_get_resolver_source(self):
        source_name = as_str(
            self.opts.resolver_source,
            name="glyph resolver source",
            pool=("coords", "u_percent", "orient"),
        )
        if source_name == "orient":
            return self.raw_orient
        return super()._helper_get_resolver_source()

    # ==================== OVERRIDE ====================
    # PlotRod overrides PlotGlyph.__getattribute__ because rod geometry expands
    # each logical sample into two endpoints, so several resolved per-rod arrays
    # must be repeated to stay aligned with the endpoint-based polydata.
    # ==================================================
    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if name in ["_calc_color", "_calc_opacity", "_calc_radius", "_calc_scalars"]:
            value = np.repeat(value, 2, axis=0)
        return value

    # ==================== OVERRIDE ====================
    # PlotRod overrides PlotGlyph._helper_bound_coords because rods can
    # center-clip by filtering their raw center points directly.
    # ==================================================
    def _helper_bound_coords(self):
        bounds = self.bounds
        if bounds is None:
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
        mask_keep = mask_inside if self.state_is_clip_inside else ~mask_inside
        keep_index = np.nonzero(mask_keep)[0].astype(int, copy=False)
        object.__setattr__(self, "_calc_keep_index", keep_index)
        return self.raw_coords[keep_index]

    # ==================== OVERRIDE ====================
    # PlotRod overrides PlotGlyph._helper_build_poly because rod glyphs are
    # represented by oriented line segments built from center points plus
    # per-sample length and orientation data.
    # ==================================================
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_build_poly(self, logger=None):

        keep_index = getattr(self, "_calc_keep_index", None)
        if keep_index is None:
            keep_index = np.arange(len(self.raw_coords), dtype=int)

        points = self._calc_coords
        if len(points) == 0:
            poly = pv.PolyData(np.empty((0, 3), dtype=float))
            object.__setattr__(self, "_calc_poly", poly)
            self._helper_set_poly(poly)
            return

        length = self._calc_length[keep_index].reshape(-1, 1)
        orient = self.raw_orient[keep_index].copy()

        orient_norm = np.linalg.norm(orient, axis=1, keepdims=True)
        mask = orient_norm.squeeze() > 1e-5
        if not np.all(mask):
            n_bad = np.count_nonzero(~mask)
            logger.warning(
                f"{n_bad} rod(s) have near-zero orientation norm (<= 1e-5). "
                "Their directions are left unnormalized, which may lead to degenerate or invisible rods."
            )
        orient[mask] /= orient_norm[mask]

        n_rods = points.shape[0]
        half = 0.5 * length
        p_minus = points - half * orient
        p_plus = points + half * orient
        endpoints = np.empty((2 * n_rods, 3), dtype=p_minus.dtype)
        endpoints[0::2] = p_minus
        endpoints[1::2] = p_plus

        lines = np.empty((n_rods, 3), dtype=np.int64)
        lines[:, 0] = 2
        lines[:, 1] = 2 * np.arange(n_rods)
        lines[:, 2] = 2 * np.arange(n_rods) + 1

        poly = pv.PolyData(endpoints, lines=lines.ravel())

        object.__setattr__(self, "_calc_poly", poly)
        self._helper_set_poly(poly)

    # ==================== OVERRIDE ====================
    # PlotRod overrides PlotGlyph._helper_set_poly so center-based clipping
    # can directly filter per-rod pointwise visual data with the kept indices.
    # ==================================================
    def _helper_set_poly(self, poly):
        if self.state_clip_mode != "center":
            return super()._helper_set_poly(poly)

        if poly.n_points == 0:
            return

        keep_index = getattr(self, "_calc_keep_index", None)
        if keep_index is None:
            keep_index = np.arange(len(self.raw_coords), dtype=int)

        color_raw = object.__getattribute__(self, "_calc_color")
        opacity_raw = object.__getattribute__(self, "_calc_opacity")
        radius_raw = object.__getattribute__(self, "_calc_radius")
        scalars_raw = object.__getattribute__(self, "_calc_scalars")

        color = np.repeat(color_raw[keep_index], 2, axis=0)
        opacity = np.repeat(opacity_raw[keep_index], 2, axis=0)
        radius = np.repeat(radius_raw[keep_index], 2, axis=0)
        scalars = np.repeat(scalars_raw[keep_index], 2, axis=0)

        poly.point_data["radius"] = radius
        poly.point_data["opacity"] = opacity
        poly.point_data["scalars"] = scalars
        rgba_values = np.hstack([color, opacity.reshape(-1, 1)])
        poly.point_data["rgba"] = rgba_values

    # ==================== OVERRIDE ====================
    # PlotRod overrides PlotGlyph._helper_build_mesh because rods use the
    # rod-specific endpoint polydata and rely on tube filtering without capping
    # or extra spline processing.
    # ==================================================
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_build_mesh(self, logger=None):

        poly = self._calc_poly
        if poly.n_points < 2 or "radius" not in poly.point_data:
            return pv.PolyData()

        mesh = poly.tube(
            scalars="radius",
            n_sides=self.opts.sides,
            absolute=True,
        )

        object.__setattr__(self, "_calc_poly", poly)
        return mesh

    # ==================== OVERRIDE ====================
    # PlotRod overrides PlotGlyph._helper_resolve_pick to expose
    # the local rod orientation in addition to the generic glyph info.
    # ==================================================
    def _helper_resolve_pick(self, picked_point):
        pos, msg, idx = super()._helper_resolve_pick(picked_point)
        value = fmt_value(self.raw_orient[idx])
        msg = f"Local orientation: {value} \n" + msg
        return pos, msg, idx
