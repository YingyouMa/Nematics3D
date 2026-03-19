from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence, Any, Mapping, ClassVar
import numpy as np
import pyvista as pv
from types import MappingProxyType

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import UNSET, Unset, as_bool, as_Number
from .plot_figure import FigureData, PlotFigure
from .glyph import OptsGlyph, PlotGlyph
from ..bounds import BoundsData
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
    """
    Option container controlling how `PlotTube` objects are rendered.

    `OptsTube` stores the visual settings for tube glyphs. It does not
    define the centerline coordinates themselves; those come from the
    `coords` passed to `PlotTube`. Instead, this class controls how the
    resulting tube geometry is drawn.

    You will usually use `OptsTube` in one of three ways:

    - create an `OptsTube(...)` instance and pass it into `PlotTube`
    - modify fields on an existing `tube.opts`
    - apply one set of settings on an object via `tube.act_commit(opts=opts_given)`

    Many of the most important fields support several input styles. In
    particular, quantities such as `radius`, `color`, `opacity`, and `scalars`
    are commonly given as:

    - one shared value applied to every tube point
    - an array providing one value per point along the centerline
    - a function that receives the point coordinates and computes values from
      them

    The most commonly adjusted fields are:

    - `radius`: tube thickness along the path
    - `color`: direct RGB coloring
    - `opacity`: transparency
    - `paint_by`: choose between direct color and scalar-colormap rendering
    - `scalars`: numeric values used when `paint_by="scalars"`
    - `sides`: roundness of the tube cross-section
    - `is_capping`: whether the tube ends are closed
    - `smooth_iter`: smoothing applied to the centerline before meshing

    A few useful relationships to keep in mind:

    - `color` and `scalars` are different pipelines; `paint_by` decides which
      one is used for rendering
    - `scalars` are numeric data, not RGB colors
    - `smooth_iter` changes the plotted path geometry, while lighting fields
      such as `ambient`, `diffuse`, `specular`, `metallic`, and `roughness`
      only affect appearance

    If you want the full field list and their short descriptions, see
    `OptsTube.__attrs__`.
    For the shared glyph option model, validation behavior, and lower-level
    commit/update rules, see the docstrings of `OptsGlyph` and `OptsBase`.

    Examples
    --------
    Create reusable tube options:

    >>> opts = OptsTube(radius=0.2, color=(0.9, 0.2, 0.2), opacity=0.8)
    >>> tube = PlotTube(coords, opts=opts)

    Set one radius for the whole tube:

    >>> opts = OptsTube(radius=0.15)

    Set a different radius along the centerline:

    >>> opts = OptsTube(radius=np.array([0.1, 0.2, 0.3, 0.2])) # four points

    Compute values from the point coordinates:

    >>> opts = OptsTube(
    ...     radius=lambda pts: 0.05 + 0.1 * np.abs(pts[:, 2]),
    ...     opacity=lambda pts: np.clip(1.0 - pts[:, 0], 0.2, 1.0),
    ... )

    Use scalar coloring:

    >>> opts = OptsTube(
    ...     paint_by="scalars",
    ...     scalars=lambda pts: pts[:, 2],
    ...     scalars_cmap="viridis",
    ... )
    """

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
    Render one or more connected centerlines as tube geometry.

    `PlotTube` is a concrete glyph class for visualizing polyline data as
    tubes. The input `coords` defines the centerline points, and the visual
    appearance of the resulting tube geometry is controlled through `opts`,
    keyword arguments, or later updates via `act_commit(...)`.

    Each visual channel is resolved point-by-point along the centerline. In
    particular, values such as `radius`, `color`, `opacity`, and `scalars` may
    be given as a single shared setting, as explicit per-point data, or as a
    function that computes those values from the input point coordinates
    through the normal glyph option pipeline.

    Typical workflow:

    - provide an `N x 3` coordinate array describing the centerline points
    - optionally provide `line_index` to split the points into multiple
      disconnected tube paths
    - optionally attach the object to an existing figure or plotter so
      multiple objects share the same scene, or let `PlotTube` create a new
      figure automatically when `figure=None`
    - optionally bind `bounds` to clip which paths or which parts of paths are
      shown
    - choose visual settings such as `radius`, `color`, `opacity`, or
      scalar-based coloring, either as constants, arrays, or coordinate-based
      functions
    - update the object later with `act_commit(...)`, direct edits on
      `tube.opts`, or by reusing `opts`

    Parameters
    ----------
    coords
        Centerline coordinates with shape `(N, 3)`. Each row gives one point
        on the tube path. A single point given as shape `(3,)` is also
        accepted, but tube geometry needs at least two points to be drawn.
    name
        Optional readable object name.
    category
        Category label used when the object is registered in a figure.
        The default is `"tube"`.
    figure
        Optional figure/container for this glyph. You may pass an existing
        `PlotFigure`, a `pyvistaqt.BackgroundPlotter`, or a `pyvista.Plotter`.
        Non-`PlotFigure` inputs are wrapped into a `PlotFigure` internally so
        this glyph can join an existing scene without extra setup. If `None`,
        a new figure is created automatically.
    opts
        Optional `OptsTube` instance holding the visual configuration.
        You can also reuse an existing options object later with
        `tube.act_commit(opts=other.opts)` to apply another object's current
        option settings directly.
        If both `opts` and explicit option keyword arguments are provided,
        the explicit keyword arguments are merged in and take precedence.
    line_index
        Optional 1D integer array with length `N` that groups centerline points
        into separate paths. Consecutive runs with the same index are treated
        as one polyline. Use this when one `PlotTube` should contain multiple
        disconnected tube segments.
    bounds
        Optional clipping object forwarded through the underlying `PlotGlyph`
        interface.
    clip_mode
        Controls how bounds clipping is applied.
        - `"center"`: decide whether to keep tube points from their centerline
          positions, then rebuild tube segments from the surviving points.
          This is the default setting.
        - `"mesh"`: build the tube geometry first, then clip the resulting
          mesh against the bounds.
    name_replace, opts_defaults_override, and other advanced keyword arguments
        These mostly affect default resolution and higher-level host/glyph
        behavior. New users can usually ignore them at first; see the
        docstring of `PlotGlyph` if you want the full forwarding model.
    **kwargs
        Additional option values forwarded into the glyph configuration
        pipeline. For the full list of supported visual options, see the
        docstring of `OptsTube` and its base option classes.

    Interactive Behavior
    --------------------
    In an interactive figure window:

    - left double-click adds a numbered marker at the resolved picked
      location on the tube, or removes the nearest existing marker if you
      double-click near one
    - right click toggles the silhouette highlight of the picked tube object
      and prints the object summary in the figure console
    - right double-click opens the tube-specific interaction panel

    Examples
    --------
    Create one tube from centerline coordinates:

    >>> import numpy as np
    >>> pts = np.array([
    ...     [0.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.0],
    ...     [1.0, 1.0, 0.0],
    ... ])
    >>> tube = PlotTube(
    ...     pts,
    ...     radius=0.1,
    ...     color=(0.9, 0.2, 0.2),
    ...     opacity=0.8,
    ... )

    Split one coordinate array into multiple disconnected tube paths:

    >>> line_index = np.array([0, 0, 0, 1, 1, 1])
    >>> tube = PlotTube(coords, line_index=line_index, radius=0.08)

    Update the appearance after creation:

    >>> tube.act_commit(radius=0.12, color=(0.2, 0.4, 0.9))
    >>> tube.opts.opacity = 1

    Reuse another opts:

    >>> tube.act_commit(opts=other_tube.opts)

    Compute per-point values from the coordinates:

    >>> tube.act_commit(
    ...     radius=lambda pts: 0.05 + 0.05 * np.abs(pts[:, 2]),
    ...     color=lambda pts: np.column_stack([
    ...         np.clip(pts[:, 0], 0.0, 1.0),
    ...         np.clip(pts[:, 1], 0.0, 1.0),
    ...         np.full(len(pts), 0.4),
    ...     ]),
    ... )

    Use scalar coloring instead of a fixed RGB color:

    >>> tube.act_commit(
    ...     paint_by="scalars",
    ...     scalars=given_scalars_array,
    ...     scalars_cmap="viridis",
    ... )

    See Also
    --------
    OptsTube
        Tube-specific options.
    PlotGlyph
        Base glyph pipeline shared by drawable plot objects.
    PlotFigure
        Figure container that manages plotted objects.
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
        figure: FigureData | None = None,
        opts: OptsTube | None = None,
        line_index: Sequence | None = None,
        bounds: BoundsData | None = None,
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
            bounds=bounds,
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
        mask_keep = mask_inside if self.state_is_clip_inside else ~mask_inside

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
