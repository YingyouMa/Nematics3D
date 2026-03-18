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
        opts_defaults_override: Mapping[str, Any] | None = None,
        logger=None,
        **kwargs
    ):

        super().__init__(
            coords=coords,
            opts_type=OptsSphere,
            category=category,
            name=name,
            name_replace=name_replace,
            opts=opts,
            figure=figure,
            opts_defaults_override=opts_defaults_override,
            **kwargs,
        )

        self._helper_init_end()
        self.act_set_interact_func(lambda: InteractSphere(self, self.fig).show())

    # ==================== OVERRIDE ====================
    # PlotSphere overrides PlotGlyph._helper_build_mesh to generate sphere
    # geometry for each input point using the resolved radius values.
    # ==================================================

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_build_mesh(self, logger=None):

        poly = self._calc_poly
        unit_sphere = pv.Sphere(
            theta_resolution=self.opts.sides, phi_resolution=self.opts.sides, radius=1.0
        )
        mesh = poly.glyph(geom=unit_sphere, scale="radius", orient=False)

        return mesh
