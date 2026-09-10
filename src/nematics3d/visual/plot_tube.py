"""Tube glyph visuals built on the shared PlotGlyph pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping, Sequence

import numpy as np
import pyvista as pv

from nematics3d.datatypes import UNSET, Unset, as_bool
from nematics3d.format import fmt_value
from nematics3d.geometry import closest_point_on_polyline
from nematics3d.logging_decorator import logging_and_warning_decorator

from nematics3d.analysis.bounds import BoundsData
from nematics3d.core.class_base import AttrDef
from nematics3d.core.host_base import HostBase
from nematics3d.visual.glyph import OptsGlyph, PlotGlyph
from nematics3d.visual.plot_figure import FigureData


#! test
#! color invalid


@dataclass(slots=True, repr=False)
class OptsTube(OptsGlyph):
    """
    Visual configuration object for `PlotTube`.

    `OptsTube` stores the settings that control how tube glyphs look after
    they are created. It does not define the centerline coordinates; those
    come from the `coords` passed to `PlotTube`. Instead, this class
    controls appearance, coloring, shading, scalar mapping, and tube
    meshing details.

    Important readable attributes:

    - `host`: the PlotTube currently using this opts object, if any.
    - `radius`, `color`, `opacity`, `scalars`: the main pointwise visual
      controls for tube appearance.
    - `paint_by`: chooses direct RGBA painting or scalar-colormap rendering.
    - `resolver_source`: selects the input used by callable visual resolvers.
    - `sides`, `is_capping`: the main tube-meshing controls.

    Common user actions:

    - `act_finalize()`: validate defaults and lock the opts into functioning use.
    - `act_asdict()`: export the current opts values as a plain dictionary.
    - `act_save_json()`: save the current opts to JSON, using sidecar `.npy`
      files when large arrays are present.
    - `act_load_json()`: load a JSON snapshot into this existing opts object.

    Common ways to use this object:

    - create `OptsTube(...)` first and pass it into `PlotTube`
    - modify fields on `tube.opts` after a tube glyph already exists
    - apply a prepared settings object with `tube.act_commit(opts=opts)`

    Most visual fields support the same three input styles:

    - one shared value applied to the whole tube
    - one value per point along the centerline, provided as an array
    - a callable resolver that computes values from the source selected by
      `resolver_source`

    The most useful fields for day-to-day work are usually:

    - `radius`: tube thickness along the path
    - `color`: direct RGB coloring
    - `opacity`: transparency
    - `paint_by`: choose direct coloring or scalar-colormap rendering
    - `scalars`: numeric values used when `paint_by="scalars"`
    - `resolver_source`: choose what callable resolvers receive
    - `sides`: roundness of the tube cross-section
    - `is_capping`: whether the tube ends are closed

    `resolver_source` controls the input passed to callable visual resolvers:

    - `"coords"`: the callable receives the raw centerline point coordinates
    - `"u_percent"`: the callable receives point-index percentages from 0
      to 100 along the glyph ordering. For tubes this follows the raw point
      order, not true arc length, and does not restart for each `line_index`
      segment.

    A few useful relationships to keep in mind:

    - `color` and `scalars` belong to different rendering pipelines;
      `paint_by` decides which one is active
    - `scalars` are numeric data, not RGB colors
    - `resolver_source` matters only when a visual field is provided as a
      callable
    - lighting fields such as `ambient`, `diffuse`, `specular`, `metallic`,
      and `roughness` change appearance but not topology

    If you want the full field list and their short descriptions, see
    `OptsTube.__attrs__`.
    For the shared glyph option model and lower-level commit/update rules,
    see the docstrings of `OptsGlyph` and `OptsBase`.

    Examples
    --------
    Create reusable tube options:

    >>> opts = OptsTube(radius=0.2, color=(0.9, 0.2, 0.2), opacity=0.8)
    >>> tube = PlotTube(coords, opts=opts)

    Use one radius for the whole tube:

    >>> opts = OptsTube(radius=0.15)

    Use one radius per centerline point:

    >>> opts = OptsTube(radius=np.array([0.1, 0.2, 0.3, 0.2])) # four points

    Resolve values from coordinates:

    >>> opts = OptsTube(
    ...     resolver_source="coords",
    ...     radius=lambda pts: 0.05 + 0.1 * np.abs(pts[:, 2]),
    ...     opacity=lambda pts: np.clip(1.0 - pts[:, 0], 0.2, 1.0),
    ... )

    Resolve values from position along the glyph order:

    >>> opts = OptsTube(
    ...     resolver_source="u_percent",
    ...     radius=lambda u: 0.05 + 0.03 * np.sin(u / 100 * np.pi),
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

    __attrs__: ClassVar[Mapping[str, str]] = {
        **dict(OptsGlyph.__attrs__),
        "is_capping": "Whether to close the ends of the tube.",
    }

    impl_validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        **dict(OptsGlyph.impl_validators),
        "is_capping": lambda v, d: as_bool(v, name=d),
    }

    impl_defaults_frozen: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            **dict(OptsGlyph.impl_defaults_frozen),
            "is_capping": True,
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
    Render one or more centerlines as tube geometry.

    `PlotTube` is the tube-based concrete glyph class. It takes point
    coordinates as centerline samples and builds tube geometry around those
    paths. This makes it useful whenever your geometry is naturally curve-
    like and should be shown with finite thickness rather than as a bare
    polyline.

    Visual appearance is controlled through `opts`, explicit keyword
    arguments, or later updates with `act_commit(...)`. Most pointwise
    visual fields such as `radius`, `color`, `opacity`, and `scalars` can be
    provided as shared constants, per-point arrays, or callable resolvers.
    Callable resolvers use the source selected by `resolver_source`.

    Important readable attributes:

    - `opts`: the paired OptsTube controlling tube appearance and meshing.
    - `fig`: the PlotFigure currently hosting this glyph, if any.
    - `bounds`: the currently bound clipping object, if any.
    - `raw_coords`: the raw centerline coordinates.
    - `raw_line_index`: the optional raw segmentation array for multiple paths.
    - `calc_line_index`: the effective segmentation after center-based clipping.
    - `calc_keep_index`: the raw point indices kept after center-based clipping.

    Common inspection helpers:

    - `show_readable_attrs()`: show the main readable tube attributes.
    - `show_modifiable_attrs()`: show which tube or opts attributes can be changed.
    - `show_attr_desc(name)`: describe a specific readable attribute.
    - `show_relations()`: show object relations inherited from PlotGlyph.

    Common user actions:

    - `act_commit(...)`: update tube raw fields or visual options.
    - `act_set_name(name)`: rename the tube object.
    - `act_remove()`: remove the tube actor from its figure.

    Typical workflow:

    - provide centerline coordinates for one tube path
    - optionally provide `line_index` if one array should define multiple
      disconnected tube paths
    - optionally attach the glyph to an existing figure or plotter so
      multiple objects share the same scene
    - optionally bind `bounds` and choose how clipping should work
    - set visual properties such as thickness, color, opacity, scalar
      coloring, or lighting
    - update the object later with `act_commit(...)`, edits on `tube.opts`,
      or by reusing another prepared `opts` object

    Parameters
    ----------
    coords
        Centerline coordinates with shape `(N, 3)`. Each row gives one
        sampled point on the path. A single point given as shape `(3,)` is
        also accepted, but visible tube geometry needs at least two points.
    name
        Optional readable object name.
    name_replace
        Fallback name used when `name` is not provided.
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
        option settings directly. If both `opts` and explicit option keyword
        arguments are provided, the explicit keyword arguments are merged in
        and take precedence.
    line_index
        Optional 1D integer array with length `N` that groups centerline
        points into separate paths. Consecutive runs with the same index are
        treated as one polyline. Use this when one `PlotTube` should contain
        multiple disconnected tube segments.
    clip_mode
        Controls how bounds clipping is applied.
        - `"center"`: decide whether to keep a tube from its centerline
          points. Surviving points are then regrouped into tube segments.
          This is the default setting.
        - `"mesh"`: build the tube geometry first, then clip the resulting
          mesh against the bounds. Use this when you want the clipped tube
          surface itself, for example to show a tube cut by a plane or box.
    is_clip_inside
        Controls whether clipping keeps the region inside the active bounds
        (`True`) or outside it (`False`). This is a glyph/host setting, not
        an `OptsTube` field.
    bounds
        Optional clipping object forwarded through the underlying `PlotGlyph`
        interface.
    opts_defaults_override and other advanced keyword arguments
        These mostly affect default resolution and higher-level host/glyph
        behavior. New users can usually ignore them at first; see the
        docstring of `PlotGlyph` if you want the full forwarding model.
    **kwargs
        Additional option values forwarded into the glyph configuration
        pipeline. For the full list of supported visual options, see the
        docstring of `OptsTube` and its base option classes.

    Resolver Behavior
    -----------------
    When a visual field is provided as a callable, the callable input is
    chosen by `resolver_source`:

    - `"coords"`: the callable receives the raw centerline coordinates
    - `"u_percent"`: the callable receives point-index percentages from 0 to
      100 along the glyph ordering. For `PlotTube`, this is based on the raw
      centerline point order rather than true arc length, and it does not
      restart separately for each disconnected `line_index` segment

    Interactive Behavior
    --------------------
    In an interactive figure window:

    - left double-click adds a numbered marker at the resolved picked
      location on the tube, or removes the nearest existing marker if you
      double-click near one
    - right click toggles the silhouette highlight of the picked tube glyph
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

    Reuse another options object:

    >>> tube.act_commit(opts=other_tube.opts)

    Resolve values from coordinates:

    >>> tube.act_commit(
    ...     resolver_source="coords",
    ...     radius=lambda pts: 0.05 + 0.05 * np.abs(pts[:, 2]),
    ...     color=lambda pts: np.column_stack([
    ...         np.clip(pts[:, 0], 0.0, 1.0),
    ...         np.clip(pts[:, 1], 0.0, 1.0),
    ...         np.full(len(pts), 0.4),
    ...     ]),
    ... )

    Resolve values from point-order percentage:

    >>> tube.act_commit(
    ...     resolver_source="u_percent",
    ...     radius=lambda u: 0.05 + 0.03 * np.sin(u / 100 * np.pi),
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

    # fmt: off
    __attr_defs__ = {
        "raw_line_index": AttrDef(
            doc="Optional polyline membership indices.",
            kind="raw",
            is_reapply_opts_after_raw=True,
        ),
        "calc_line_index": AttrDef(
            doc=(
                "The effective polyline membership indices used for the "
                "current glyph build after clip-mode preprocessing."
            ),
            kind="calc",
        ),
        "calc_keep_index": AttrDef(
            doc="Indices of raw centerline points kept after center-based point filtering.",
            kind="calc",
        ),
    }

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
        and name not in HostBase.__slots__
    )
    # -------------------------------
    # Initialization
    # -------------------------------

    # ==================== OVERRIDE ====================
    # PlotTube overrides PlotGlyph.__init__ only to accept
    # and validate the tube-specific raw field `line_index`.
    # ==================================================
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
        is_clip_inside: bool = True,
        opts_defaults_override: Mapping[str, Any] | None = None,
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
            is_clip_inside=is_clip_inside,
            opts_defaults_override=opts_defaults_override,
            **kwargs,
        )

        object.__setattr__(self, "raw_line_index", None)
        object.__setattr__(self, "calc_line_index", None)
        object.__setattr__(self, "calc_keep_index", None)
        self._helper_commit_line_index({"line_index": line_index})

        self._helper_init_end()
        def _show_interact():
            from nematics3d.visual.qt.interact_tube import InteractTube

            return InteractTube.show_once(self, self.fig)

        self.act_set_interact_func(_show_interact)

    # -------------------------------
    # Tube raw topology helpers
    # -------------------------------

    def _helper_check_index(self, line_index, name):
        if line_index is None:
            return None
        line_index = np.asarray(line_index, dtype=int)
        if line_index.ndim != 1 or len(line_index) != self.raw_coords.shape[0]:
            raise ValueError(
                f"`line_index` is {name}. "
                f"It must be a ({self.raw_coords.shape[0]},) array. "
                f"Got shape {line_index.shape} instead."
            )
        return line_index

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
            object.__setattr__(self, "calc_line_index", None)
        else:
            object.__setattr__(
                self,
                "calc_line_index",
                np.asarray(idx, dtype=int).copy(),
            )

    def _helper_iter_line_ranges(self):
        return self._helper_iter_index_ranges(self.raw_line_index, len(self.raw_coords))

    @staticmethod
    def _helper_iter_index_ranges(line_index, n_points):
        """Return consecutive half-open ranges for one polyline-membership array."""
        if line_index is None or n_points == 0:
            return [] if n_points == 0 else [(0, n_points)]
        line_index = np.asarray(line_index)
        if len(np.unique(line_index)) == 1:
            return [(0, n_points)]
        breaks = np.nonzero(line_index[1:] != line_index[:-1])[0] + 1
        starts = np.r_[0, breaks]
        ends = np.r_[breaks, n_points]
        return list(zip(starts, ends))

    @staticmethod
    def _helper_split_contiguous_indices(indices):
        """Split sorted raw indices wherever adjacency is broken."""
        indices = np.asarray(indices, dtype=int)
        if len(indices) == 0:
            return []
        breaks = np.nonzero(np.diff(indices) != 1)[0] + 1
        return [
            chunk
            for chunk in np.split(indices, breaks)
            if len(chunk) > 0
        ]

    # -------------------------------
    # Center clipping and polyline preparation
    # -------------------------------

    # ==================== OVERRIDE ====================
    # PlotTube overrides PlotGlyph._helper_bound_coords because
    # tubes can first filter centerline points against bounds and then
    # build the tube mesh from the surviving centerline segments.
    # ==================================================
    def _helper_bound_coords(self):
        bounds = self._helper_get_bounds_effective()
        if bounds is None:
            self._helper_sync_calc_line_index()
            keep_index = np.arange(len(self.raw_coords), dtype=int)
            object.__setattr__(self, "calc_keep_index", keep_index)
            return self.raw_coords.copy()

        mask_inside = bounds.act_contains_points(self.raw_coords, atol=1e-10)
        mask_keep = mask_inside if self.state_is_clip_inside else ~mask_inside

        coords_segments = []
        keep_segments = []
        for start, end in self._helper_iter_line_ranges():
            keep_range = np.nonzero(mask_keep[start:end])[0] + start
            if len(keep_range) == 0:
                continue

            for raw_idx in self._helper_split_contiguous_indices(keep_range):
                if len(raw_idx) < 2:
                    continue
                keep_segments.append(raw_idx.astype(int, copy=False))
                coords_segments.append(self.raw_coords[raw_idx])

        if len(coords_segments) == 0:
            object.__setattr__(self, "calc_line_index", None)
            object.__setattr__(self, "calc_keep_index", np.empty((0,), dtype=int))
            return np.empty((0, 3), dtype=float)

        if len(coords_segments) == 1:
            object.__setattr__(self, "calc_line_index", None)
            object.__setattr__(self, "calc_keep_index", keep_segments[0])
            return np.asarray(coords_segments[0], dtype=float)

        coords_all = np.vstack(coords_segments).astype(float, copy=False)
        keep_index = np.concatenate(keep_segments).astype(int, copy=False)
        line_index = np.concatenate(
            [np.full(len(seg), i, dtype=int) for i, seg in enumerate(coords_segments)]
        )
        object.__setattr__(self, "calc_line_index", line_index)
        object.__setattr__(self, "calc_keep_index", keep_index)
        return coords_all

    # ==================== OVERRIDE ====================
    # PlotTube overrides PlotGlyph._helper_build_poly because
    # a tube may represent one or multiple disconnected polylines,
    # so the PolyData topology cannot reuse the glyph default.
    # ==================================================
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_build_poly(self, logger=None):

        points = self.calc_coords
        if self.state_clip_mode == "center":
            idx = self.calc_line_index
        else:
            self._helper_sync_calc_line_index()
            object.__setattr__(
                self,
                "calc_keep_index",
                np.arange(len(self.raw_coords), dtype=int),
            )
            idx = self.calc_line_index

        if len(points) < 2:
            poly = pv.PolyData(np.asarray(points, dtype=float))
            object.__setattr__(self, "calc_poly", poly)
            self._helper_set_poly(poly)
            return

        # Decide whether to treat the input as a single continuous polyline
        is_use_multi = (idx is None) or (len(np.unique(idx)) == 1)
        if is_use_multi:
            poly = pv.MultipleLines(points)
        else:
            chunks = []
            for s, e in self._helper_iter_index_ranges(idx, len(points)):
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

        object.__setattr__(self, "calc_poly", poly)
        self._helper_set_poly(poly)

    # ==================== OVERRIDE ====================
    # PlotTube overrides PlotGlyph._helper_set_poly so center-based clipping
    # can directly filter pointwise visual data with the kept center indices.
    # ==================================================
    def _helper_set_poly(self, poly):
        if self.state_clip_mode != "center":
            super()._helper_set_poly(poly)
            return

        if poly.n_points == 0:
            return

        keep_index = getattr(self, "calc_keep_index", None)
        if keep_index is None:
            keep_index = np.arange(len(self.raw_coords), dtype=int)

        radius = self.calc_radius[keep_index]
        opacity = self.calc_opacity[keep_index]
        scalars = self.calc_scalars[keep_index]
        color = self.calc_color[keep_index]

        if len(radius) > 0:
            poly.point_data["radius"] = radius
        poly.point_data["opacity"] = opacity
        poly.point_data["scalars"] = scalars
        rgba_values = np.hstack([color, opacity.reshape(-1, 1)])
        poly.point_data["rgba"] = rgba_values

    # -------------------------------
    # Mesh generation and commit hooks
    # -------------------------------

    def _helper_materialize_mesh(self):
        """
        Internal: generate tube geometry from the prepared polyline dataset.
        """

        poly = self.calc_poly
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

    # -------------------------------
    # Picking
    # -------------------------------

    # ==================== OVERRIDE ====================
    # PlotTube overrides PlotGlyph._helper_resolve_pick to report
    # tube-specific information such as normalized arc position
    # and, when available, the local tangent direction.
    # ==================================================
    def _helper_resolve_pick(self, picked_point):

        pos_close, msg, idx = super()._helper_resolve_pick(picked_point)
        u_percent = idx / len(self.raw_coords) * 100
        msg_head = (
            f"The closest point on the tube is {fmt_value(pos_close)}, where: \n"
            f"The normalized position along the tube is {u_percent:.3f} \n"
        )
        try:
            smooth = self.wrapper.owner
            tgt = smooth.act_calc_tangent(u_percent)
            msg_head += f"Local tangent: {fmt_value(tgt)} \n"
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass
        msg = msg_head + msg

        pos = self._helper_closest_point_on_raw_paths(picked_point)

        return pos, msg, idx

    def _helper_closest_point_on_raw_paths(self, picked_point):
        """Resolve a pick against real polyline segments without bridging line breaks."""
        if len(self.raw_coords) == 0:
            raise RuntimeError("Cannot resolve pick on an empty tube.")

        query = np.asarray(picked_point, dtype=float)
        closest = None
        closest_d2 = None
        for start, end in self._helper_iter_line_ranges():
            path = self.raw_coords[start:end]
            if len(path) < 2:
                continue
            candidate = closest_point_on_polyline(query, path)
            delta = candidate - query
            distance_squared = float(np.dot(delta, delta))
            if closest_d2 is None or distance_squared < closest_d2:
                closest = candidate
                closest_d2 = distance_squared

        if closest is not None:
            return closest
        return self.raw_coords[0].copy()
