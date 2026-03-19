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
    Option container controlling how `PlotSphere` objects are rendered.

    `OptsSphere` stores the visual settings for sphere glyphs. It does not
    define where spheres are placed; the sphere centers come from the
    `coords` passed to `PlotSphere`. Instead, this class controls how those
    spheres look once they are drawn.

    You will usually use `OptsSphere` in one of three ways:

    - create an `OptsSphere(...)` instance and pass it into `PlotSphere`
    - modify fields on an existing `sphere.opts`
    - reuse one object's current settings on another object via
      `other_sphere.act_commit(opts=sphere.opts)`

    Many of the most important fields support several input styles. In
    particular, quantities such as `radius`, `color`, `opacity`, and `scalars`
    are commonly given as:

    - one shared value applied to every sphere
    - an array providing one value per sphere
    - a function that receives the point coordinates and computes values from
      them

    The most commonly adjusted fields are:

    - `radius`: sphere size
    - `color`: direct RGB coloring
    - `opacity`: transparency
    - `paint_by`: choose between direct color and scalar-colormap rendering
    - `scalars`: numeric values used when `paint_by="scalars"`
    - `sides`: sphere smoothness

    A few useful relationships to keep in mind:

    - `color` and `scalars` are different pipelines; `paint_by` decides which
      one is used for rendering
    - `scalars` are numeric data, not RGB colors
    - `is_clip_inside` matters when the owning glyph is clipped by bounds
    - lighting fields such as `ambient`, `diffuse`, `specular`, `metallic`,
      and `roughness` affect appearance but not sphere geometry

    For the shared glyph option model, validation behavior, and lower-level
    commit/update rules, see the docstrings of `OptsGlyph` and `OptsBase`.

    Examples
    --------
    Create reusable sphere options:

    >>> opts = OptsSphere(radius=0.2, color=(0.9, 0.2, 0.2), opacity=0.8)
    >>> spheres = PlotSphere(coords, opts=opts)

    Set one radius for all spheres:

    >>> opts = OptsSphere(radius=0.15)

    Set a different radius for each sphere:

    >>> opts = OptsSphere(radius=np.array([0.1, 0.2, 0.3]))

    Compute values from the point coordinates:

    >>> opts = OptsSphere(
    ...     radius=lambda pts: 0.1 + 0.2 * np.linalg.norm(pts, axis=1),
    ...     opacity=lambda pts: np.clip(pts[:, 2], 0.2, 1.0),
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


# Subclassing rules:
# - PlotSphere keeps the generic PlotGlyph pipeline and only specializes the
#   sphere geometry. Subclasses should preserve that separation of concerns.
# - If a subclass changes the mesh generation, keep radius handling compatible
#   with the glyph pipeline so point-wise radius data still maps cleanly onto
#   the generated geometry.
# - Keep the default interaction behavior aligned with sphere-specific tooling
#   unless there is a clear reason to expose a different interaction panel.


class PlotSphere(PlotGlyph):
    """
    Render one sphere at each input coordinate.

    `PlotSphere` is a concrete glyph class for visualizing a set of 3D points as
    spheres. The input `coords` defines the sphere centers, and the visual
    appearance of those spheres is controlled through `opts`, keyword
    arguments, or later updates via `act_commit(...)`.

    Each visual channel is resolved point-by-point. In particular, values such
    as `radius`, `color`, `opacity`, and `scalars` may be given as a single
    shared setting for all spheres, as explicit per-point data, or as a
    function that computes those values from the input point coordinates
    through the normal glyph option pipeline.

    Typical workflow:

    - provide an `N x 3` coordinate array
    - optionally attach the object to an existing figure or plotter so
      multiple objects share the same scene
    - optionally bind `bounds` to clip which or which parts of spheres are shown
    - choose visual settings such as `radius`, `color`, `opacity`, or
      scalar-based coloring, either as constants, arrays, or coordinate-based
      functions
    - update the object later with `act_commit(...)`, direct edits on
      `sphere.opts`, or by reusing `opts`

    Parameters
    ----------
    coords
        Sphere-center coordinates with shape `(N, 3)`. Each row gives the
        center of one sphere.
    name
        Optional readable object name.
    category
        Category label used when the object is registered in a figure.
        The default is `"sphere"`.
    figure
        Optional figure/container for this glyph. You may pass an existing
        `PlotFigure`, a `pyvistaqt.BackgroundPlotter`, or a `pyvista.Plotter`.
        Non-`PlotFigure` inputs are wrapped into a `PlotFigure` internally so
        this glyph can join an existing scene without extra setup.
    opts
        Optional `OptsSphere` instance holding the visual configuration.
        You can also reuse an existing options object later with
        `sphere.act_commit(opts=other.opts)` to apply another object's current
        option settings directly.
    clip_mode
        Controls how bounds clipping is applied.
        - `"center"`: keep or remove spheres according to whether their
          centers are inside the active bounds. Default setting.
        - `"mesh"`: build the sphere geometry first, then clip the resulting
          mesh against the bounds.
    bounds
        Optional clipping object forwarded through the underlying `PlotGlyph`
        interface.
    name_replace, opts_defaults_override, and other advanced keyword arguments
        These mostly affect default resolution and higher-level host/glyph
        behavior. New users can usually ignore them at first; see the
        docstring of `PlotGlyph` if you want the full forwarding model.
    **kwargs
        Additional option values forwarded into the glyph configuration
        pipeline. For the full list of supported visual options, see the
        docstring of `OptsSphere` and its base option classes.

    Interactive Behavior
    --------------------
    In an interactive figure window:

    - left double-click adds a numbered marker at the resolved picked location,
      or removes the nearest existing marker if you double-click near one
    - right click toggles the silhouette highlight of the picked sphere object
      and prints the object summary in the figure console
    - right double-click opens the sphere-specific interaction panel

    Examples
    --------
    Create a few spheres from point coordinates:

    >>> import numpy as np
    >>> pts = np.array([
    ...     [0.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.0],
    ...     [0.0, 1.0, 0.0],
    ... ])
    >>> spheres = PlotSphere(
    ...     pts,
    ...     radius=0.2,
    ...     color=(0.9, 0.2, 0.2), # RGB
    ...     opacity=0.8,
    ... )

    Update the appearance after creation:

    >>> spheres.act_commit(radius=0.35, color=(0.2, 0.4, 0.9))
    >>> spheres.opts.opacity = 1

    Reuse another opts:

    >>> spheres.act_commit(opts=other_spheres.opts)

    Compute per-point values from the coordinates:

    >>> spheres.act_commit(
    ...     radius=lambda pts: 0.1 + 0.2 * np.linalg.norm(pts, axis=1),
    ...     color=lambda pts: np.column_stack([
    ...         np.clip(pts[:, 0], 0.0, 1.0),
    ...         np.clip(pts[:, 1], 0.0, 1.0),
    ...         np.full(len(pts), 0.4),
    ...     ]),
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
        clip_mode: str = "center",
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
            clip_mode=clip_mode,
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
        mask_keep = mask_inside if self.opts.is_clip_inside else ~mask_inside
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
