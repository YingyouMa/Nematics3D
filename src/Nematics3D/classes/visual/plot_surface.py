from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, Mapping, ClassVar
import numpy as np
import pyvista as pv
from types import MappingProxyType

from Nematics3D.logging_decorator import logging_and_warning_decorator
from .plot_figure import FigureData, PlotFigure
from .glyph import OptsGlyph, PlotGlyph
from ..bounds import BoundsData
from .qt.interact_surface import InteractSurface


@dataclass(slots=True, repr=False)
class OptsSurface(OptsGlyph):
    """
    Visual configuration object for `PlotSurface`.

    `OptsSurface` stores the settings that control how reconstructed surface
    glyphs look after they are created. It does not define the input point
    cloud itself; those points come from the `coords` passed to
    `PlotSurface`. Instead, this class controls coloring, opacity, shading,
    scalar mapping, and other display properties of the generated surface
    mesh.

    Important readable attributes:

    - `host`: the PlotSurface currently using this opts object, if any.
    - `color`, `opacity`, `scalars`: the main pointwise visual controls.
    - `paint_by`: chooses direct RGBA painting or scalar-colormap rendering.
    - `resolver_source`: selects the input used by callable visual resolvers.
    - `shading_type`, `ambient`, `diffuse`, `specular`, `metallic`,
      `roughness`: the main surface-lighting controls.

    Common user actions:

    - `act_finalize()`: validate defaults and lock the opts into functioning use.
    - `act_asdict()`: export the current opts values as a plain dictionary.
    - `act_save_json()`: save the current opts to JSON, using sidecar `.npy`
      files when large arrays are present.
    - `act_load_json()`: load a JSON snapshot into this existing opts object.

    Common ways to use this object:

    - create `OptsSurface(...)` first and pass it into `PlotSurface`
    - modify fields on `surface.opts` after a surface glyph already exists
    - apply a prepared settings object with `surface.act_commit(opts=opts)`

    The most useful fields for day-to-day work are usually:

    - `color`: direct RGB coloring
    - `opacity`: transparency
    - `paint_by`: choose direct coloring or scalar-colormap rendering
    - `scalars`: numeric values used when `paint_by="scalars"`
    - `resolver_source`: choose what callable resolvers receive
    - lighting fields such as `shading_type`, `ambient`, `diffuse`,
      `specular`, `metallic`, and `roughness`

    Most pointwise visual fields that still matter for surfaces, especially
    `color`, `opacity`, and `scalars`, support the same three input styles:

    - one shared value applied to the whole surface
    - one value per input point, provided as an array
    - a callable resolver that computes values from the source selected by
      `resolver_source`

    `resolver_source` controls the input passed to callable visual resolvers:

    - `"coords"`: the callable receives the raw surface sample coordinates
    - `"u_percent"`: the callable receives point-index percentages from 0
      to 100 along the raw point ordering

    A few useful relationships to keep in mind:

    - `color` and `scalars` belong to different rendering pipelines;
      `paint_by` decides which one is active
    - `scalars` are numeric data, not RGB colors
    - `resolver_source` matters only when a visual field is provided as a
      callable
    - `radius` and `sides` are deprecated placeholders here and currently do
      not affect surface rendering
    - lighting fields change appearance but not the reconstructed geometry

    If you want the full field list and their short descriptions, see
    `OptsSurface.__attrs__`.
    For the shared glyph option model and lower-level commit/update rules,
    see the docstrings of `OptsGlyph` and `OptsBase`.

    Examples
    --------
    Create reusable surface options:

    >>> opts = OptsSurface(color=(0.9, 0.2, 0.2), opacity=0.8)
    >>> surface = PlotSurface(coords, opts=opts)

    Resolve values from coordinates:

    >>> opts = OptsSurface(
    ...     resolver_source="coords",
    ...     opacity=lambda pts: np.clip(1.0 - np.abs(pts[:, 2]), 0.2, 1.0),
    ... )

    Use scalar coloring:

    >>> opts = OptsSurface(
    ...     paint_by="scalars",
    ...     scalars=lambda pts: pts[:, 2],
    ...     scalars_cmap="viridis",
    ... )
    """

    __attrs__: ClassVar[Mapping[str, str]] = {
        **(OptsGlyph.__attrs__),
        "radius": (
            "Deprecated placeholder. "
            "Currently has no effect in surface plots. "
            "Kept temporarily to avoid refactoring overhead."
        ),
        "sides": (
            "Deprecated placeholder. "
            "Currently has no effect in surface plots. "
            "Kept temporarily to avoid refactoring overhead."
        ),
    }

    _validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        k: v for k, v in OptsGlyph._validators.items() if k not in ("radius", "sides")
    }

    _DEFAULTS_FROZEN: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {**(OptsGlyph._DEFAULTS_FROZEN), "ambient": 0.5}
    )


# PlotSurface keeps the generic glyph host behavior but replaces the geometry
# generation and silhouette handling with surface-specific logic.
#
# Subclasses should preserve the assumption that surfaces are resolved from
# point clouds through a mesh-building stage, and keep silhouette updates in
# sync with any actor or mesh replacement.
class PlotSurface(PlotGlyph):
    """
    Reconstruct and render a surface mesh from input points.

    `PlotSurface` is the surface-based concrete glyph class. It takes a set
    of 3D sample points, builds a surface mesh from them, and then renders
    that mesh with the normal glyph display pipeline. This makes it useful
    when your geometry is naturally a sampled surface rather than isolated
    points, tubes, or oriented rods.

    Visual appearance is controlled through `opts`, explicit keyword
    arguments, or later updates with `act_commit(...)`. The most relevant
    pointwise visual fields for surfaces are `color`, `opacity`, and
    `scalars`; these can be provided as shared constants, per-point arrays,
    or callable resolvers. Callable resolvers use the source selected by
    `resolver_source`.

    Important readable attributes:

    - `opts`: the paired OptsSurface controlling surface appearance.
    - `fig`: the PlotFigure currently hosting this glyph, if any.
    - `bounds`: the currently bound clipping object, if any.
    - `raw_coords`: the raw surface sample coordinates.
    - `_calc_keep_index`: the raw point indices kept after center-based clipping.

    Common inspection helpers:

    - `show_getattrs()`: show the main readable surface attributes.
    - `show_modifiable_attrs()`: show which surface or opts attributes can be changed.
    - `show_attr_desc(name)`: describe a specific readable attribute.
    - `show_relations()`: show object relations inherited from PlotGlyph.

    Common user actions:

    - `act_commit(...)`: update surface raw fields or visual options.
    - `act_set_name(name)`: rename the surface object.
    - `act_remove()`: remove the surface actor from its figure.

    Typical workflow:

    - provide surface sample coordinates
    - optionally attach the glyph to an existing figure or plotter so
      multiple objects share the same scene
    - optionally bind `bounds` and choose how clipping should work
    - set visual properties such as color, opacity, scalar coloring, or
      lighting
    - update the object later with `act_commit(...)`, edits on
      `surface.opts`, or by reusing another prepared `opts` object

    Parameters
    ----------
    coords
        Surface sample coordinates with shape `(N, 3)`. Each row gives one
        point used for surface reconstruction. At least three points are
        needed to build a visible surface mesh.
    name
        Optional readable object name.
    name_replace
        Fallback name used when `name` is not provided.
    category
        Category label used when the object is registered in a figure.
        The default is `"surface"`.
    figure
        Optional figure/container for this glyph. You may pass an existing
        `PlotFigure`, a `pyvistaqt.BackgroundPlotter`, or a `pyvista.Plotter`.
        Non-`PlotFigure` inputs are wrapped into a `PlotFigure` internally so
        this glyph can join an existing scene without extra setup. If `None`,
        a new figure is created automatically.
    opts
        Optional `OptsSurface` instance holding the visual configuration.
        You can also reuse an existing options object later with
        `surface.act_commit(opts=other.opts)` to apply another object's
        current option settings directly. If both `opts` and explicit option
        keyword arguments are provided, the explicit keyword arguments are
        merged in and take precedence.
    clip_mode
        Controls how bounds clipping is applied.
        - `"center"`: decide whether to keep surface sample points from
          their coordinates before the surface mesh is reconstructed. This is
          the default setting.
        - `"mesh"`: reconstruct the surface first, then clip the resulting
          mesh against the bounds. Use this when you want the clipped surface
          itself, for example to show a cut surface patch.
    is_clip_inside
        Controls whether clipping keeps the region inside the active bounds
        (`True`) or outside it (`False`). This is a glyph/host setting, not
        an `OptsSurface` field.
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
        docstring of `OptsSurface` and its base option classes.

    Resolver Behavior
    -----------------
    When a visual field is provided as a callable, the callable input is
    chosen by `resolver_source`:

    - `"coords"`: the callable receives the raw surface sample coordinates
    - `"u_percent"`: the callable receives point-index percentages from 0 to
      100 along the raw point ordering

    Notes
    -----
    `PlotSurface` currently reconstructs the visible mesh from the prepared
    point cloud with a 2D Delaunay step. This means the result depends on the
    input point distribution and is best suited to point sets that already
    sample a surface reasonably well.

    Interactive Behavior
    --------------------
    In an interactive figure window:

    - left double-click adds a numbered marker at the resolved picked
      location on the surface, or removes the nearest existing marker if you
      double-click near one
    - right click toggles the silhouette highlight of the picked surface
      glyph and prints the object summary in the figure console
    - right double-click opens the surface-specific interaction panel

    Examples
    --------
    Create a surface from sample points:

    >>> import numpy as np
    >>> pts = np.array([
    ...     [0.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.1],
    ...     [0.0, 1.0, 0.0],
    ...     [1.0, 1.0, 0.2],
    ... ])
    >>> surface = PlotSurface(
    ...     pts,
    ...     color=(0.8, 0.5, 0.2),
    ...     opacity=0.9,
    ... )

    Update the appearance after creation:

    >>> surface.act_commit(color=(0.2, 0.4, 0.9), opacity=0.7)
    >>> surface.opts.opacity = 1

    Reuse another options object:

    >>> surface.act_commit(opts=other_surface.opts)

    Resolve values from coordinates:

    >>> surface.act_commit(
    ...     resolver_source="coords",
    ...     color=lambda pts: np.column_stack([
    ...         np.clip(pts[:, 0], 0.0, 1.0),
    ...         np.clip(pts[:, 1], 0.0, 1.0),
    ...         np.full(len(pts), 0.5),
    ...     ]),
    ... )

    Use scalar coloring instead of a fixed RGB color:

    >>> surface.act_commit(
    ...     paint_by="scalars",
    ...     scalars=lambda pts: pts[:, 2],
    ...     scalars_cmap="viridis",
    ... )

    See Also
    --------
    OptsSurface
        Surface-specific options.
    PlotGlyph
        Base glyph pipeline shared by drawable plot objects.
    PlotFigure
        Figure container that manages plotted objects.
    """

    __attrs__: ClassVar[Mapping[str, str]] = {
        **{k: v for k, v in PlotGlyph.__attrs__.items() if k != "_calc_radius"},
        "_calc_keep_index": "Indices of raw surface points kept after center-based point filtering.",
    }

    __slots__ = tuple(
        k
        for k, v in __attrs__.items()
        if not v.startswith("Property:") and k not in PlotGlyph.__slots__
    )

    _pending_resolution_attrs = ["color", "scalars", "opacity"]

    # -------------------------------
    # Initialization
    # -------------------------------

    # ==================== OVERRIDE ====================
    # PlotSurface overrides PlotGlyph.__init__ only to select the surface opts
    # type and install the surface-specific interaction entry point.
    # ==================================================
    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        coords: np.ndarray,
        name: str | None = None,
        name_replace: str = "surface",
        category: str = "surface",
        figure: FigureData | None = None,
        opts: OptsSurface | None = None,
        bounds: BoundsData | None = None,
        clip_mode: str = "center",
        is_clip_inside: bool = True,
        opts_defaults_override: Mapping[str, Any] | None = None,
        logger=None,
        **kwargs,
    ):

        super().__init__(
            coords=coords,
            opts_type=OptsSurface,
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

        object.__setattr__(self, "_calc_keep_index", None)
        self.act_set_interact_func(lambda: InteractSurface(self, self.fig).show())

        self._helper_init_end()

    # -------------------------------
    # Center clipping and point preparation
    # -------------------------------

    # ==================== OVERRIDE ====================
    # PlotSurface overrides PlotGlyph._helper_bound_coords because
    # surface glyphs can center-clip by filtering raw surface points directly.
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
    # PlotSurface overrides PlotGlyph._helper_set_poly so center-based clipping
    # can directly filter pointwise visual data with the kept indices.
    # ==================================================
    def _helper_set_poly(self, poly):
        if self.state_clip_mode != "center":
            return super()._helper_set_poly(poly)

        if poly.n_points == 0:
            return

        keep_index = getattr(self, "_calc_keep_index", None)
        if keep_index is None:
            keep_index = np.arange(len(self.raw_coords), dtype=int)

        opacity = self._calc_opacity[keep_index]
        scalars = self._calc_scalars[keep_index]
        color = self._calc_color[keep_index]

        poly.point_data["opacity"] = opacity
        poly.point_data["scalars"] = scalars
        rgba_values = np.hstack([color, opacity.reshape(-1, 1)])
        poly.point_data["rgba"] = rgba_values

    # -------------------------------
    # Mesh generation and silhouette
    # -------------------------------

    # ==================== OVERRIDE ====================
    # PlotSurface overrides PlotGlyph._helper_build_mesh because surfaces are
    # reconstructed from the prepared point cloud with a 2D Delaunay stage
    # instead of glyph or tube extrusion logic.
    # ==================================================
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_build_mesh(self, logger=None):

        poly = self._calc_poly
        if poly.n_points < 3:
            return pv.PolyData()
        mesh = poly.delaunay_2d(alpha=0.0)

        return mesh

    # ==================== OVERRIDE ====================
    # PlotSurface overrides PlotGlyph._helper_add_silhouette because surface
    # objects need a feature-edge outline generated from the triangulated mesh
    # rather than the generic glyph silhouette behavior.
    # ==================================================
    def _helper_add_silhouette(self):

        plotter = self.fig.pl

        silhouette_id = f"{self._impl_name_pv}__silhouette"
        if silhouette_id in plotter.actors:
            plotter.remove_actor(silhouette_id)

        mesh = self._entity.mapper.dataset
        surf = mesh.extract_surface().triangulate().clean()
        outline = surf.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )

        actor_silhouette = plotter.add_mesh(
            outline,
            color=(0, 0, 0),
            line_width=6,
            opacity=0.8,
        )
        actor_silhouette.visibility = False
        actor_silhouette.pickable = False

        object.__setattr__(self, "_entity_silhouette", actor_silhouette)
