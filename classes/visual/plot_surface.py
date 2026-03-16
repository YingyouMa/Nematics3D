from dataclasses import dataclass
from typing import Callable, Any, Mapping, ClassVar
import numpy as np
import pyvista as pv
from types import MappingProxyType

from Nematics3D.logging_decorator import logging_and_warning_decorator
from .plot_figure import PlotFigure
from .glyph import OptsGlyph, PlotGlyph
from .qt.interact_surface import InteractSurface


@dataclass(slots=True, repr=False)
class OptsSurface(OptsGlyph):
    
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
        k: v
        for k, v in OptsGlyph._validators.items()
        if k not in ( "radius", "sides")
    }

    _DEFAULTS_FROZEN: ClassVar[Mapping[str, Any]] = MappingProxyType({
        **(OptsGlyph._DEFAULTS_FROZEN),
        "ambient": 0.5
    })

    
    
class PlotSurface(PlotGlyph):
    
    __attrs__: ClassVar[Mapping[str, str]] = {
        k: v
        for k, v in PlotGlyph.__attrs__.items()
        if k != "_calc_radius"
    }
    
    __slots__ = tuple(
            k for k, v in __attrs__.items() 
            if not v.startswith("Property:") and k not in PlotGlyph.__slots__
        )
    
    _pending_resolution_attrs = ['color', 'scalars', 'opacity']
    
    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        coords: np.ndarray,
        name: str | None = None,
        name_replace: str = 'surface',
        category: str = 'surface',
        figure: PlotFigure | None = None,
        opts: OptsSurface | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        logger = None,
        **kwargs
    ):
        
        

        super().__init__(
            coords=coords,
            opts_type=OptsSurface,
            category=category,
            name=name,
            name_replace=name_replace,
            opts=opts,
            figure=figure,
            opts_defaults_override=opts_defaults_override,
            **kwargs,
        )

        self.act_set_interact_func(lambda: InteractSurface(self, self.fig).show())

        self._helper_init_end()
        
            
            
    @logging_and_warning_decorator(start_finish_level=5)    
    def _helper_build_mesh(self, logger=None):
        
        poly = self._calc_poly  
        mesh = poly.delaunay_2d(alpha=0.0)

        if self.opts.clip_geometry is not None:
            if isinstance(self.opts.clip_geometry, (list, tuple)) and len(self.opts.clip_geometry) == 6:
                mesh = mesh.clip_box(bounds=self.opts.clip_geometry, invert=False)
            elif hasattr(self.opts.clip_geometry, "points"):
                mesh = mesh.clip_surface(self.opts.clip_geometry, invert=False)

        return mesh   
    

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
            color=(0,0,0),
            line_width=6,
            opacity=0.8,
        )
        actor_silhouette.visibility = False
        actor_silhouette.pickable = False
        
        object.__setattr__(self, "_entity_silhouette", actor_silhouette)








