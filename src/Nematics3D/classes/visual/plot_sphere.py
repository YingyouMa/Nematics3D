from dataclasses import dataclass
from typing import Any, Mapping, ClassVar
import numpy as np
import pyvista as pv
from types import MappingProxyType

from Nematics3D.logging_decorator import logging_and_warning_decorator
from .plot_figure import PlotFigure
from .glyph import OptsGlyph, PlotGlyph
from .qt.interact_sphere import InteractSphere


@dataclass(slots=True, repr=False)
class OptsSphere(OptsGlyph):

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
    Glyph subclass for rendering point-like objects as spheres.

    For most users, PlotSphere is the simplest concrete glyph family: each input
    point is rendered as a sphere whose appearance is controlled through
    `opts` or `act_commit(...)`.

    Typical usage:

    - create a sphere glyph from point coordinates
    - attach it to a `PlotFigure` or let it create one automatically
    - adjust visual settings such as color, radius, opacity, and scalar display
      through `sphere.opts`
    - use the built-in interaction panel for sphere-specific inspection and
      tuning when running interactively
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
        figure: PlotFigure | None = None,
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
