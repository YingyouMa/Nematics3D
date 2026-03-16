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

    _DEFAULTS_FROZEN: ClassVar[Mapping[str, Any]] = MappingProxyType({
        **dict(OptsGlyph._DEFAULTS_FROZEN),
        "sides":    12
    })
        

class PlotSphere(PlotGlyph):

    __attrs__ = {
        **dict(PlotGlyph.__attrs__),
    }
    
    __slots__ = tuple(
            k for k, v in __attrs__.items() 
            if not v.startswith("Property:") and k not in PlotGlyph.__slots__
        )
    
    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        coords: np.ndarray,
        name: str | None = None,
        name_replace: str = 'point',
        category: str = 'sphere',
        figure: PlotFigure | None = None,
        opts: OptsSphere | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        logger = None,
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
            
            
    @logging_and_warning_decorator(start_finish_level=5)    
    def _helper_build_mesh(self, logger=None):
        
        poly = self._calc_poly
        unit_sphere = pv.Sphere(theta_resolution=self.opts.sides, 
                                phi_resolution=self.opts.sides, 
                                radius=1.0)
        mesh = poly.glyph(geom=unit_sphere, scale="radius", orient=False)

        if self.opts.clip_geometry is not None:
            if isinstance(self.opts.clip_geometry, (list, tuple)) and len(self.opts.clip_geometry) == 6:
                mesh = mesh.clip_box(bounds=self.opts.clip_geometry, invert=False)
            elif hasattr(self.opts.clip_geometry, "points"):
                mesh = mesh.clip_surface(self.opts.clip_geometry, invert=False)

        return mesh
    
