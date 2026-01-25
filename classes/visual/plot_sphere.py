from dataclasses import dataclass
from typing import Any, Mapping, ClassVar
import numpy as np
import pyvista as pv
from types import MappingProxyType

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import as_str
from .plot_figure import PlotFigure
from .glyph import OptsGlyph, PlotGlyph


@dataclass(slots=True, repr=False)
class OptsSphere(OptsGlyph):

    _DEFAULTS_FROZEN: ClassVar[Mapping[str, Any]] = MappingProxyType({
        **dict(OptsGlyph._DEFAULTS_FROZEN),
        "sides":    12
    })
        

class PlotSphere(PlotGlyph):

    __descriptions__ = {
        **dict(PlotGlyph.__descriptions__),
    }
    
    __slots__ = tuple(__descriptions__.keys())  #+ ("__weakref__",)
    
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
        
        
        category = as_str(category, name="The category of the PlotTube object", replace="tube")
        object.__setattr__(self, 'raw_category', category)

        super().__init__(
            coords=coords,
            opts_type=OptsSphere,
            category=category,
            name=name,
            name_replace=name_replace,
            opts=opts,
            figure=figure,
            opts_defaults_override=opts_defaults_override,
            logger=logger,
            **kwargs,
        )

        self._helper_resolver_init()
        self._helper_make_figure()
        self._helper_init_end()
        
    def __setattr__(self, key, value):
        self._helper_setattr_glyph_basic(key, value, allowed_extra = ())
            
            
    @logging_and_warning_decorator(start_finish_level=5)    
    def _helper_build_mesh(self, logger=None):
        
        poly = self._calc_poly
            
        logger.detail("Applying sphere filter with dynamic radius scaling")
        unit_sphere = pv.Sphere(theta_resolution=self.opts.sides, 
                                phi_resolution=self.opts.sides, 
                                radius=1.0)
        mesh = poly.glyph(geom=unit_sphere, scale="radius", orient=False)

        if self.opts.clip_geometry is not None:
            logger.detail("Applying spatial clipping to sphere mesh")
            if isinstance(self.opts.clip_geometry, (list, tuple)) and len(self.opts.clip_geometry) == 6:
                mesh = mesh.clip_box(bounds=self.opts.clip_geometry, invert=False)
            elif hasattr(self.opts.clip_geometry, "points"):
                mesh = mesh.clip_surface(self.opts.clip_geometry, invert=False)

        object.__setattr__(self, "_calc_poly", poly)
        return mesh
    
