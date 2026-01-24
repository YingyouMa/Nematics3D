from dataclasses import dataclass
from typing import Callable, Any, Mapping, ClassVar
import numpy as np
import pyvista as pv
from types import MappingProxyType

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import  as_str
from .plot_figure import PlotFigure
from .glyph import OptsGlyph, PlotGlyph


@dataclass(slots=True, repr=False)
class OptsDelaunay(OptsGlyph):

    __descriptions__: ClassVar[Mapping[str, str]] = {
        k: v
        for k, v in OptsGlyph.__descriptions__.items()
        if k != "radius"
    }

    _validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        k: v
        for k, v in OptsGlyph._validators.items()
        if k != "radius"
    }

    _DEFAULTS_FROZEN: ClassVar[Mapping[str, Any]] = MappingProxyType({
        **{
        k: v
        for k, v in OptsGlyph._DEFAULTS_FROZEN.items()
        if k != "radius"
    },
        "ambient": 0.5
    })

    _commit_level: ClassVar[Mapping[str, Any]] = {
        k: v
        for k, v in OptsGlyph._commit_level.items()
        if k != "radius"
    }
    
    
class PlotDelaunay(PlotGlyph):
    
    __descriptions__: ClassVar[Mapping[str, str]] = {
        k: v
        for k, v in PlotGlyph.__descriptions__.items()
        if k != "_calc_radius"
    }
    
    __slots__ = tuple(__descriptions__.keys())  #+ ("__weakref__",)
    
    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        coords: np.ndarray,
        name: str | None = None,
        name_replace: str = 'surface',
        category: str = 'surface',
        figure: PlotFigure | None = None,
        opts: OptsDelaunay | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        logger = None,
        **kwargs
    ):
        
        
        category = as_str(category, name="The category of the PlotDelaunay object", replace="surface")
        object.__setattr__(self, 'raw_category', category)

        super().__init__(
            coords=coords,
            opts_type=OptsDelaunay,
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
        
        points = self.raw_coords
        poly = pv.PolyData(points)

        if isinstance(self.opts.color, str) and self.opts.color == 'scalars':
            poly.point_data['opacity'] = self._calc_opacity
            poly.point_data['scalars'] = self._calc_scalars
        else:
            rgba_values = np.hstack([self._calc_color, self._calc_opacity.reshape(-1, 1)])
            poly.point_data['rgba'] = rgba_values 
            
        logger.detail("Creating the triangulation by Delaunay method.")
        mesh = poly.delaunay_2d(alpha=0.0)

        if self.opts.clip_geometry is not None:
            logger.detail("Applying spatial clipping to sphere mesh")
            if isinstance(self.opts.clip_geometry, (list, tuple)) and len(self.opts.clip_geometry) == 6:
                mesh = mesh.clip_box(bounds=self.opts.clip_geometry, invert=False)
            elif hasattr(self.opts.clip_geometry, "points"):
                mesh = mesh.clip_surface(self.opts.clip_geometry, invert=False)

        object.__setattr__(self, "_calc_poly", poly)
        # object.__setattr__(self, "_calc_mesh", mesh)
        return mesh   
    

    def _helper_add_silhouette(self):
    
        plotter = self.owner.pl

        silhouette_id = f"{self._internal_name_pv}__silhouette"
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
    
    
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_resolver_init(self, extra=[], logger=None):
        logger.detail("Resolving data for color, opacity and scalars")
        
        self._helper_resolver_spec('opacity')
        for attr in extra:
            self._helper_resolver_spec(attr)
        
        if isinstance(self.opts.color, str) and self.opts.color == 'scalars':
            self._helper_resolver_spec('scalars')
        else:
            self._helper_resolver_spec('color')
            
    def _helper_commit_apply(self, is_needs_remesh, **kwargs):
        return super()._helper_commit_apply(is_needs_remesh, is_radius=False, **kwargs)



