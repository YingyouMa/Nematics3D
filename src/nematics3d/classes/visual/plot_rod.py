"""Rod glyph visuals built on the shared PlotGlyph pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping, Sequence

import numpy as np
import pyvista as pv

from nematics3d.datatypes import UNSET, Unset, as_Number, as_points, as_str
from nematics3d.general import fmt_value
from nematics3d.logging_decorator import logging_and_warning_decorator

from ..bounds import BoundsData
from .glyph import OptsGlyph, PlotGlyph
from .plot_figure import FigureData
from .qt.interact_rod import InteractRod

LengthMode = float | Callable | Sequence


@dataclass(slots=True, repr=False)
class OptsRod(OptsGlyph):
    """
    Visual configuration object for `PlotRod`.

    `OptsRod` stores the settings that control how rod glyphs look after
    they are created. It does not define rod centers or orientations;
    those come from the `coords` and `orient` passed to `PlotRod`.
    Instead, this class controls appearance, coloring, shading, scalar
    mapping, thickness, rod length, and meshing details.

    Important readable attributes:

    - `host`: the PlotRod currently using this opts object, if any.
    - `length`, `radius`, `color`, `opacity`, `scalars`: the main per-rod
      visual controls.
    - `paint_by`: chooses direct RGBA painting or scalar-colormap rendering.
    - `resolver_source`: selects the input used by callable visual resolvers.
    - `sides`: the main rod-meshing control.

    Common user actions:

    - `act_finalize()`: validate defaults and lock the opts into functioning use.
    - `act_asdict()`: export the current opts values as a plain dictionary.
    - `act_save_json()`: save the current opts to JSON, using sidecar `.npy`
      files when large arrays are present.
    - `act_load_json()`: load a JSON snapshot into this existing opts object.

    Common ways to use this object:

    - create `OptsRod(...)` first and pass it into `PlotRod`
    - modify fields on `rod.opts` after a rod glyph already exists
    - apply a prepared settings object with `rod.act_commit(opts=opts)`

    Most visual fields support the same three input styles:

    - one shared value applied to every rod
    - one value per rod, provided as an array
    - a callable resolver that computes values from the source selected by
      `resolver_source`

    The most useful fields for day-to-day work are usually:

    - `length`: rod length
    - `radius`: rod thickness
    - `color`: direct RGB coloring
    - `opacity`: transparency
    - `paint_by`: choose direct coloring or scalar-colormap rendering
    - `scalars`: numeric values used when `paint_by="scalars"`
    - `resolver_source`: choose what callable resolvers receive
    - `sides`: roundness of the rod cross-section

    `resolver_source` controls the input passed to callable visual resolvers:

    - `"coords"`: the callable receives the raw rod-center coordinates
    - `"u_percent"`: the callable receives point-index percentages from 0
      to 100 along the glyph ordering
    - `"orient"`: the callable receives the raw orientation vectors. This is
      the default setting for rods.

    A few useful relationships to keep in mind:

    - `color` and `scalars` belong to different rendering pipelines;
      `paint_by` decides which one is active
    - `scalars` are numeric data, not RGB colors
    - `resolver_source` matters only when a visual field is provided as a
      callable
    - `length` and `orient` together determine the rod endpoints before the
      rod is meshed
    - lighting fields such as `ambient`, `diffuse`, `specular`, `metallic`,
      and `roughness` change appearance but not geometry

    If you want the full field list and their short descriptions, see
    `OptsRod.__attrs__`.
    For the shared glyph option model and lower-level commit/update rules,
    see the docstrings of `OptsGlyph` and `OptsBase`.

    Examples
    --------
    Create reusable rod options:

    >>> opts = OptsRod(length=3.0, radius=0.3, color=(0.9, 0.2, 0.2))
    >>> rods = PlotRod(coords, orient, opts=opts)

    Use one length for every rod:

    >>> opts = OptsRod(length=2.0)

    Use one length per rod:

    >>> opts = OptsRod(length=np.array([1.0, 2.0, 3.0])) # three rods

    Resolve values from coordinates:

    >>> opts = OptsRod(
    ...     resolver_source="coords",
    ...     length=lambda pts: 1.0 + np.abs(pts[:, 2]),
    ... )

    Resolve values from orientation vectors:

    >>> opts = OptsRod(
    ...     resolver_source="orient",
    ...     color=lambda n: np.abs(n),
    ...     length=lambda n: 1.0 + 2.0 * np.abs(n[:, 2]),
    ... )

    Use scalar coloring:

    >>> opts = OptsRod(
    ...     paint_by="scalars",
    ...     scalars=lambda n: n[:, 2],
    ...     resolver_source="orient",
    ...     scalars_cmap="viridis",
    ... )
    """

    # --- Geometry & Topology (Rod-specific) ---
    length: LengthMode | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        **dict(OptsGlyph.__attrs__),
        "length": "The length of rods",
    }

    impl_validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        **dict(OptsGlyph.impl_validators),
        "length": lambda v, d: as_Number(v, name=d, value_range=(1e-12, np.inf)),
        "resolver_source": lambda v, d: as_str(
            v,
            name=d,
            pool=("coords", "u_percent", "orient"),
        ),
    }

    impl_defaults_frozen: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            **dict(OptsGlyph.impl_defaults_frozen),
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
    Render one oriented rod at each input point.

    `PlotRod` is the rod-based concrete glyph class. It takes point
    coordinates as rod centers and orientation vectors that define the rod
    directions. Each rod is then built as an oriented line segment with
    finite thickness.

    This makes it useful whenever your geometry is naturally point-like but
    also has a meaningful local direction, such as directors, normals, or
    vector samples.

    Visual appearance is controlled through `opts`, explicit keyword
    arguments, or later updates with `act_commit(...)`. Most pointwise
    visual fields such as `length`, `radius`, `color`, `opacity`, and
    `scalars` can be provided as shared constants, per-rod arrays, or
    callable resolvers. Callable resolvers use the source selected by
    `resolver_source`.

    Important readable attributes:

    - `opts`: the paired OptsRod controlling rod appearance and length.
    - `fig`: the PlotFigure currently hosting this glyph, if any.
    - `bounds`: the currently bound clipping object, if any.
    - `raw_coords`: the raw rod-center coordinates.
    - `raw_orient`: the raw rod orientation vectors.
    - `calc_length`: the resolved per-rod lengths used for geometry building.
    - `calc_keep_index`: the raw rod indices kept after center-based clipping.

    Common inspection helpers:

    - `show_readable_attrs()`: show the main readable rod attributes.
    - `show_modifiable_attrs()`: show which rod or opts attributes can be changed.
    - `show_attr_desc(name)`: describe a specific readable attribute.
    - `show_relations()`: show object relations inherited from PlotGlyph.

    Common user actions:

    - `act_commit(...)`: update rod raw fields or visual options.
    - `act_set_name(name)`: rename the rod object.
    - `act_remove()`: remove the rod actor from its figure.

    Typical workflow:

    - provide rod-center coordinates
    - provide one orientation vector per rod
    - optionally attach the glyph to an existing figure or plotter so
      multiple objects share the same scene
    - optionally bind `bounds` and choose how clipping should work
    - set visual properties such as length, thickness, color, opacity,
      scalar coloring, or lighting
    - update the object later with `act_commit(...)`, edits on `rod.opts`,
      or by reusing another prepared `opts` object

    Parameters
    ----------
    coords
        Rod-center coordinates with shape `(N, 3)`. Each row gives one rod
        center. A single point given as shape `(3,)` is also accepted and
        treated as one rod center.
    orient
        Rod orientation vectors with shape `(N, 3)`. There must be exactly
        one orientation vector for each center point.
    name
        Optional readable object name.
    name_replace
        Fallback name used when `name` is not provided.
    category
        Category label used when the object is registered in a figure.
        The default is `"rods"`.
    figure
        Optional figure/container for this glyph. You may pass an existing
        `PlotFigure`, a `pyvistaqt.BackgroundPlotter`, or a `pyvista.Plotter`.
        Non-`PlotFigure` inputs are wrapped into a `PlotFigure` internally so
        this glyph can join an existing scene without extra setup. If `None`,
        a new figure is created automatically.
    opts
        Optional `OptsRod` instance holding the visual configuration. You can
        also reuse an existing options object later with
        `rod.act_commit(opts=other.opts)` to apply another object's current
        option settings directly. If both `opts` and explicit option keyword
        arguments are provided, the explicit keyword arguments are merged in
        and take precedence.
    clip_mode
        Controls how bounds clipping is applied.
        - `"center"`: decide whether to keep a rod from its center point.
          This is the default setting.
        - `"mesh"`: build the rod geometry first, then clip the resulting
          mesh against the bounds. Use this when you want the clipped rod
          surface itself, for example to show rods cut by a plane or box.
    is_clip_inside
        Controls whether clipping keeps the region inside the active bounds
        (`True`) or outside it (`False`). This is a glyph/host setting, not
        an `OptsRod` field.
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
        docstring of `OptsRod` and its base option classes.

    Resolver Behavior
    -----------------
    When a visual field is provided as a callable, the callable input is
    chosen by `resolver_source`:

    - `"coords"`: the callable receives the raw rod-center coordinates
    - `"u_percent"`: the callable receives point-index percentages from 0 to
      100 along the glyph ordering
    - `"orient"`: the callable receives the raw orientation vectors. This is
      the default resolver source for rods.

    Interactive Behavior
    --------------------
    In an interactive figure window:

    - left double-click adds a numbered marker at the resolved picked
      location, or removes the nearest existing marker if you double-click
      near one
    - right click toggles the silhouette highlight of the picked rod glyph
      and prints the object summary in the figure console
    - right double-click opens the rod-specific interaction panel

    Examples
    --------
    Create rods from centers and orientations:

    >>> import numpy as np
    >>> pts = np.array([
    ...     [0.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.0],
    ...     [0.0, 1.0, 0.0],
    ... ])
    >>> orient = np.array([
    ...     [1.0, 0.0, 0.0],
    ...     [0.0, 1.0, 0.0],
    ...     [0.0, 0.0, 1.0],
    ... ])
    >>> rods = PlotRod(
    ...     pts,
    ...     orient,
    ...     length=2.0,
    ...     radius=0.2,
    ...     color=(0.9, 0.2, 0.2),
    ... )

    Update the appearance after creation:

    >>> rods.act_commit(length=3.0, color=(0.2, 0.4, 0.9))
    >>> rods.opts.opacity = 1

    Reuse another options object:

    >>> rods.act_commit(opts=other_rods.opts)

    Resolve values from coordinates:

    >>> rods.act_commit(
    ...     resolver_source="coords",
    ...     length=lambda pts: 1.0 + np.abs(pts[:, 2]),
    ... )

    Resolve values from orientations:

    >>> rods.act_commit(
    ...     resolver_source="orient",
    ...     color=lambda n: np.abs(n),
    ...     length=lambda n: 1.0 + 2.0 * np.abs(n[:, 2]),
    ... )

    Use scalar coloring instead of a fixed RGB color:

    >>> rods.act_commit(
    ...     paint_by="scalars",
    ...     resolver_source="orient",
    ...     scalars=lambda n: n[:, 2],
    ...     scalars_cmap="viridis",
    ... )

    See Also
    --------
    OptsRod
        Rod-specific options.
    PlotGlyph
        Base glyph pipeline shared by drawable plot objects.
    PlotFigure
        Figure container that manages plotted objects.
    """

    __attr_defs__ = {
        **dict(PlotGlyph.__attr_defs__),
        "raw_orient": {
            "doc": "The orientation vectors of rods.",
            "validator": lambda v, d: as_points(v, name=d),
            "is_reapply_opts_after_raw": True,
        },
        "calc_length": {
            "doc": "The resolved per-rod length array used for rod geometry building.",
        },
        "calc_keep_index": {
            "doc": "Indices of raw rod centers kept after center-based point filtering.",
        },
    }
    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.get("kind") not in ("relation", "property")
    )

    _pending_resolution_attrs: Sequence[str] = PlotGlyph._pending_resolution_attrs + [
        "length"
    ]

    # -------------------------------
    # Initialization
    # -------------------------------

    # ==================== OVERRIDE ====================
    # PlotRod overrides PlotGlyph.__init__ because it must accept
    # rod-specific raw orientation data before the generic glyph
    # initialization and mesh setup are performed.
    # ==================================================
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
        is_clip_inside: bool = True,
        opts_defaults_override: Mapping[str, Any] | None = None,
        **kwargs,
    ):

        orient = type(self).__attr_defs__["raw_orient"]["validator"](
            orient,
            type(self).__attr_defs__["raw_orient"]["doc"],
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

        self.act_set_interact_func(lambda: InteractRod(self, self.fig).show())

        self._helper_init_end()

    # -------------------------------
    # Resolver helpers
    # -------------------------------

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

    # -------------------------------
    # Center clipping and polyline preparation
    # -------------------------------

    # ==================== OVERRIDE ====================
    # PlotRod overrides PlotGlyph._helper_bound_coords because rods can
    # center-clip by filtering their raw center points directly.
    # ==================================================
    def _helper_expand_endpoint_values(self, values, keep_index=None):
        values = np.asarray(values)
        if keep_index is not None:
            keep_index = np.asarray(keep_index, dtype=int)
            values = values[keep_index]
        return np.repeat(values, 2, axis=0)

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
    # PlotRod overrides PlotGlyph._helper_build_poly because rod glyphs are
    # represented by oriented line segments built from center points plus
    # per-sample length and orientation data.
    # ==================================================
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_build_poly(self, logger=None):

        keep_index = getattr(self, "calc_keep_index", None)
        if keep_index is None:
            keep_index = np.arange(len(self.raw_coords), dtype=int)

        points = self.calc_coords
        if len(points) == 0:
            poly = pv.PolyData(np.empty((0, 3), dtype=float))
            object.__setattr__(self, "calc_poly", poly)
            self._helper_set_poly(poly)
            return

        length = self.calc_length[keep_index].reshape(-1, 1)
        orient = self.raw_orient[keep_index].copy()

        orient_norm = np.linalg.norm(orient, axis=1, keepdims=True)
        mask = orient_norm.squeeze() > 1e-5
        if not np.all(mask):
            n_bad = np.count_nonzero(~mask)
            logger.warning(
                f"{n_bad} rod(s) have near-zero orientation norm (<= 1e-5). "
                "Their directions are left unnormalized, which may lead to "
                "degenerate or invisible rods."
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

        object.__setattr__(self, "calc_poly", poly)
        self._helper_set_poly(poly)

    # ==================== OVERRIDE ====================
    # PlotRod overrides PlotGlyph._helper_set_poly so center-based clipping
    # can directly filter per-rod pointwise visual data with the kept indices.
    # ==================================================
    def _helper_set_poly(self, poly):
        if poly.n_points == 0:
            return

        keep_index = getattr(self, "calc_keep_index", None)
        if keep_index is None:
            keep_index = np.arange(len(self.raw_coords), dtype=int)

        color_raw = self.calc_color
        opacity_raw = self.calc_opacity
        radius_raw = self.calc_radius
        scalars_raw = self.calc_scalars

        color = self._helper_expand_endpoint_values(color_raw, keep_index)
        opacity = self._helper_expand_endpoint_values(opacity_raw, keep_index)
        radius = self._helper_expand_endpoint_values(radius_raw, keep_index)
        scalars = self._helper_expand_endpoint_values(scalars_raw, keep_index)

        poly.point_data["radius"] = radius
        poly.point_data["opacity"] = opacity
        poly.point_data["scalars"] = scalars
        rgba_values = np.hstack([color, opacity.reshape(-1, 1)])
        poly.point_data["rgba"] = rgba_values

    # -------------------------------
    # Mesh generation
    # -------------------------------

    # ==================== OVERRIDE ====================
    # PlotRod overrides PlotGlyph._helper_build_mesh because rods use the
    # rod-specific endpoint polydata and rely on tube filtering without capping
    # or extra spline processing.
    # ==================================================
    def _helper_build_mesh(self):

        poly = self.calc_poly
        if poly.n_points < 2 or "radius" not in poly.point_data:
            return pv.PolyData()

        mesh = poly.tube(
            scalars="radius",
            n_sides=self.opts.sides,
            absolute=True,
        )

        object.__setattr__(self, "calc_poly", poly)
        return mesh

    # -------------------------------
    # Picking
    # -------------------------------

    # ==================== OVERRIDE ====================
    # PlotRod overrides PlotGlyph._helper_resolve_pick to expose
    # the local rod orientation in addition to the generic glyph info.
    # ==================================================
    def _helper_resolve_pick(self, picked_point):
        pos, msg, idx = super()._helper_resolve_pick(picked_point)
        value = fmt_value(self.raw_orient[idx])
        msg = f"Local orientation: {value} \n" + msg
        return pos, msg, idx
