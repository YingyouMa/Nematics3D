from dataclasses import dataclass
from typing import Any, Mapping, ClassVar
import numpy as np
import pyvista as pv
from types import MappingProxyType

from Nematics3D.logging_decorator import logging_and_warning_decorator
from .plot_figure import FigureData, PlotFigure
from .glyph import OptsGlyph, PlotGlyph
from .qt.interact_sphere import InteractSphere


@dataclass(slots=True, repr=False)
class OptsSphere(OptsGlyph):
    """
    Visual configuration object for `PlotSphere`.

    `OptsSphere` stores the settings that control how sphere glyphs look
    after they are created. It does not define the sphere centers; those
    come from the `coords` passed to `PlotSphere`. Instead, this class
    controls appearance, coloring, shading, scalar mapping, and sphere
    meshing details.

    Common ways to use this object:

    - create `OptsSphere(...)` first and pass it into `PlotSphere`
    - modify fields on `sphere.opts` after a sphere glyph already exists
    - apply a prepared settings object with `sphere.act_commit(opts=opts)`

    Important readable attributes on `OptsSphere` include:

    - `host` to access the owning `PlotSphere` when the opts is attached

    Most visual fields support the same three input styles:

    - one shared value applied to every sphere
    - one value per sphere, provided as an array
    - a callable resolver that computes values from the source selected by
      `resolver_source`

    The most useful fields for day-to-day work are usually:

    - `radius`: sphere size
    - `color`: direct RGB coloring
    - `opacity`: transparency
    - `paint_by`: choose direct coloring or scalar-colormap rendering
    - `scalars`: numeric values used when `paint_by="scalars"`
    - `resolver_source`: choose what callable resolvers receive
    - `sides`: sphere smoothness

    `resolver_source` controls the input passed to callable visual resolvers:

    - `"coords"`: the callable receives the raw point coordinates
    - `"u_percent"`: the callable receives point-index percentages from 0
      to 100 along the glyph ordering

    A few useful relationships to keep in mind:

    - `color` and `scalars` belong to different rendering pipelines;
      `paint_by` decides which one is active
    - `scalars` are numeric data, not RGB colors
    - `resolver_source` matters only when a visual field is provided as a
      callable
    - lighting fields such as `ambient`, `diffuse`, `specular`, `metallic`,
      and `roughness` change appearance but not geometry

    If you want the full field list and their short descriptions, see
    `OptsSphere.__attrs__`.
    For the shared glyph option model and lower-level commit/update rules,
    see the docstrings of `OptsGlyph` and `OptsBase`.

    Examples
    --------
    Create reusable sphere options:

    >>> opts = OptsSphere(radius=0.2, color=(0.9, 0.2, 0.2), opacity=0.8)
    >>> spheres = PlotSphere(coords, opts=opts)

    Use one radius for every sphere:

    >>> opts = OptsSphere(radius=0.15)

    Use one radius per sphere:

    >>> opts = OptsSphere(radius=np.array([0.1, 0.2, 0.3])) # three spheres

    Resolve values from coordinates:

    >>> opts = OptsSphere(
    ...     resolver_source="coords",
    ...     radius=lambda pts: 0.1 + 0.2 * np.linalg.norm(pts, axis=1),
    ...     opacity=lambda pts: np.clip(pts[:, 2], 0.2, 1.0),
    ... )

    Resolve values from position along the glyph order:

    >>> opts = OptsSphere(
    ...     resolver_source="u_percent",
    ...     radius=lambda u: 0.08 + 0.04 * np.sin(u / 100 * np.pi),
    ... )

    Use scalar coloring:

    >>> opts = OptsSphere(
    ...     paint_by="scalars",
    ...     scalars=lambda pts: pts[:, 2],
    ...     scalars_cmap="viridis",
    ... )
    """

    _DEFAULTS_FROZEN: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {**dict(OptsGlyph._DEFAULTS_FROZEN), "sides": 12}
    )


class PlotSphere(PlotGlyph):
    """
    Render one sphere at each input point.

    `PlotSphere` is the sphere-based concrete glyph class. It takes point
    coordinates as sphere centers and builds a sphere mesh for each point.
    This makes it useful whenever your geometry is naturally point-like but
    should still be shown with finite size rather than as abstract markers.

    Visual appearance is controlled through `opts`, explicit keyword
    arguments, or later updates with `act_commit(...)`. Most pointwise
    visual fields such as `radius`, `color`, `opacity`, and `scalars` can be
    provided as shared constants, per-point arrays, or callable resolvers.
    Callable resolvers use the source selected by `resolver_source`.

    Typical workflow:

    - provide point coordinates for the sphere centers
    - optionally attach the glyph to an existing figure or plotter so
      multiple objects share the same scene
    - optionally bind `bounds` and choose how clipping should work
    - set visual properties such as size, color, opacity, scalar coloring,
      or lighting
    - update the object later with `act_commit(...)`, edits on `sphere.opts`,
      or by reusing another prepared `opts` object

    Important readable attributes on `PlotSphere` include:

    - `opts` to access the paired `OptsSphere` object controlling rendering
    - `fig` to access the containing `PlotFigure`
    - `bounds` to inspect the currently bound clipping `Bounds` object
    - `raw_coords` to inspect the original sphere-center coordinates
    - `_calc_keep_index` to inspect which raw points remain after center-based
      clipping

    Parameters
    ----------
    coords
        Sphere-center coordinates with shape `(N, 3)`. Each row gives one
        sphere center. A single point given as shape `(3,)` is also accepted
        and treated as one sphere center.
    name
        Optional readable object name.
    name_replace
        Fallback name used when `name` is not provided.
    category
        Category label used when the object is registered in a figure.
        The default is `"sphere"`.
    figure
        Optional figure/container for this glyph. You may pass an existing
        `PlotFigure`, a `pyvistaqt.BackgroundPlotter`, or a `pyvista.Plotter`.
        Non-`PlotFigure` inputs are wrapped into a `PlotFigure` internally so
        this glyph can join an existing scene without extra setup. If `None`,
        a new figure is created automatically.
    opts
        Optional `OptsSphere` instance holding the visual configuration.
        You can also reuse an existing options object later with
        `sphere.act_commit(opts=other.opts)` to apply another object's
        current option settings directly. If both `opts` and explicit option
        keyword arguments are provided, the explicit keyword arguments are
        merged in and take precedence.
    clip_mode
        Controls how bounds clipping is applied.
        - `"center"`: decide whether to keep a sphere from its center point.
          This is the default setting.
        - `"mesh"`: build the sphere geometry first, then clip the resulting
          mesh against the bounds. Use this when you want the clipped sphere
          surface itself, for example to show a hemisphere or a 3/4 sphere.
    is_clip_inside
        Controls whether clipping keeps the region inside the active bounds
        (`True`) or outside it (`False`). This is a glyph/host setting, not
        an `OptsSphere` field.
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
        docstring of `OptsSphere` and its base option classes.

    Resolver Behavior
    -----------------
    When a visual field is provided as a callable, the callable input is
    chosen by `resolver_source`:

    - `"coords"`: the callable receives the raw sphere-center coordinates
    - `"u_percent"`: the callable receives point-index percentages from 0 to
      100 along the glyph ordering

    Interactive Behavior
    --------------------
    In an interactive figure window:

    - left double-click adds a numbered marker at the resolved picked
      location, or removes the nearest existing marker if you double-click
      near one
    - right click toggles the silhouette highlight of the picked sphere glyph
      and prints the object summary in the figure console
    - right double-click opens the sphere-specific interaction panel

    Examples
    --------
    Create spheres from point coordinates:

    >>> import numpy as np
    >>> pts = np.array([
    ...     [0.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.0],
    ...     [0.0, 1.0, 0.0],
    ... ])
    >>> spheres = PlotSphere(
    ...     pts,
    ...     radius=0.2,
    ...     color=(0.9, 0.2, 0.2),
    ...     opacity=0.8,
    ... )

    Update the appearance after creation:

    >>> spheres.act_commit(radius=0.35, color=(0.2, 0.4, 0.9))
    >>> spheres.opts.opacity = 1

    Reuse another options object:

    >>> spheres.act_commit(opts=other_spheres.opts)

    Resolve values from coordinates:

    >>> spheres.act_commit(
    ...     resolver_source="coords",
    ...     radius=lambda pts: 0.1 + 0.2 * np.linalg.norm(pts, axis=1),
    ...     color=lambda pts: np.column_stack([
    ...         np.clip(pts[:, 0], 0.0, 1.0),
    ...         np.clip(pts[:, 1], 0.0, 1.0),
    ...         np.full(len(pts), 0.4),
    ...     ]),
    ... )

    Resolve values from point-order percentage:

    >>> spheres.act_commit(
    ...     resolver_source="u_percent",
    ...     radius=lambda u: 0.08 + 0.04 * np.sin(u / 100 * np.pi),
    ... )

    Use scalar coloring instead of a fixed RGB color:

    >>> spheres.act_commit(
    ...     paint_by="scalars",
    ...     scalars=given_scalars_array,
    ...     scalars_cmap="viridis",
    ... )

    See Also
    --------
    OptsSphere
        Sphere-specific options.
    PlotGlyph
        Base glyph pipeline shared by drawable plot objects.
    PlotFigure
        Figure container that manages plotted objects.
    """

    __attrs__ = {
        **dict(PlotGlyph.__attrs__),
        "_calc_keep_index": "Indices of raw points kept after center-based point filtering.",
    }

    __slots__ = tuple(
        k
        for k, v in __attrs__.items()
        if not v.startswith("Property:") and k not in PlotGlyph.__slots__
    )

    # ==================== OVERRIDE ====================
    # PlotSphere overrides PlotGlyph.__init__ because it fixes the glyph family
    # to sphere rendering and installs the sphere-specific interaction panel.
    # ==================================================

    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        coords: np.ndarray,
        name: str | None = None,
        name_replace: str = "point",
        category: str = "sphere",
        figure: FigureData | None = None,
        opts: OptsSphere | None = None,
        bounds=None,
        clip_mode: str = "center",
        is_clip_inside: bool = True,
        opts_defaults_override: Mapping[str, Any] | None = None,
        logger=None,
        **kwargs,
    ):

        super().__init__(
            coords=coords,
            opts_type=OptsSphere,
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
        self._helper_init_end()
        self.act_set_interact_func(lambda: InteractSphere(self, self.fig).show())

    # ==================== OVERRIDE ====================
    # PlotSphere overrides PlotGlyph._helper_bound_coords because
    # sphere glyphs can center-clip by filtering raw points directly.
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
    # PlotSphere overrides PlotGlyph._helper_set_poly so center-based clipping
    # can directly filter pointwise visual data with the kept point indices.
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

    # ==================== OVERRIDE ====================
    # PlotSphere overrides PlotGlyph._helper_build_mesh to generate sphere
    # geometry for each input point using the resolved radius values.
    # ==================================================

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_build_mesh(self, logger=None):

        poly = self._calc_poly
        if poly.n_points == 0 or "radius" not in poly.point_data:
            return pv.PolyData()

        unit_sphere = pv.Sphere(
            theta_resolution=self.opts.sides, phi_resolution=self.opts.sides, radius=1.0
        )
        mesh = poly.glyph(geom=unit_sphere, scale="radius", orient=False)

        return mesh
