from dataclasses import dataclass
from typing import Callable, Sequence, Any, Mapping, ClassVar
import numpy as np
import pyvista as pv
from types import MappingProxyType

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import UNSET, Unset, as_bool, as_Number, as_str
from .plot_figure import PlotFigure
from .glyph import OptsGlyph, LEVEL_REMESH, PlotGlyph
from Nematics3D.general import pop_exclusive
from Nematics3D.datatypes import as_points

LengthMode = float | Callable | Sequence

@dataclass(slots=True, repr=False)
class OptsRod(OptsGlyph):

    # --- Geometry & Topology (Tube-specific) ---
    length:             LengthMode | Unset = UNSET


    __descriptions__: ClassVar[Mapping[str, str]] = {
        **dict(OptsGlyph.__descriptions__),
        "length":       "The length of rods"
    }

    _DEFAULTS_FROZEN: ClassVar[Mapping[str, Any]] = MappingProxyType({
        **dict(OptsGlyph._DEFAULTS_FROZEN),
        "length":       3,
        "radius":       0.3,
    })

    _commit_level: ClassVar[Mapping[str, Any]] = {
        **dict(OptsGlyph._commit_level),
        "length":        LEVEL_REMESH,
    }

        
class PlotRod(PlotGlyph):

    __descriptions__ = {
        **dict(PlotGlyph.__descriptions__),
        "raw_name":     "The name identifier of the PlotRod instance",
        "raw_orient":   "The orientation of rods",
        "_calc_length": "The resolved per-point length array used for rods length."
    }
    __slots__ = tuple(__descriptions__.keys())  #+ ("__weakref__",)
    
    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        coords: np.ndarray,
        orient: np.ndarray,
        name: str = 'rod',
        name_replace: str = 'rod',
        category: str = 'rods',
        figure: PlotFigure | None = None,
        opts: OptsRod | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        logger = None,
        **kwargs
    ):

        category = as_str(category, name="The category of the PlotTube object", replace="tube")
        object.__setattr__(self, 'raw_category', category)

        orient = as_points(orient, name="The orientation of PlotRod object") 
        object.__setattr__(self, "raw_orient", orient)

        super().__init__(
            coords=coords,
            opts_type=OptsRod,
            category=category,
            name=name,
            name_replace=name_replace,
            opts=opts,
            figure=figure,
            opts_defaults_override=opts_defaults_override,
            logger=logger,
            **kwargs,
        )
        
        if len(self.raw_orient) != len(self.raw_coords):
            raise ValueError(f"There are {len(self.raw_orient)} points for orientation, while {len(self.raw_coords)} points for positions.")
            
        object.__setattr__(self, "_internal_resolver_source", "raw_orient")

        # resolver + plot
        self._helper_resolver_init(extra=['length'])
        self._helper_make_figure()
        self._helper_init_end()

    def __setattr__(self, key, value):
        self._helper_setattr_glyph_basic(key, value, allowed_extra = ("raw_orient", "orient"))

    
    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if name in ["_calc_color", "_calc_opacity", "_calc_radius", "_calc_scalars"]:
            value = np.repeat(value, 2, axis=0)
        return value
        
        
    @logging_and_warning_decorator(start_finish_level=5)    
    def _helper_build_mesh(self, logger=None):
        
        points = self.raw_coords
        length = self._calc_length.reshape(-1, 1)
        orient = self.raw_orient
        
        orient_norm = np.linalg.norm(orient, axis=1, keepdims=True)
        mask = orient_norm.squeeze() > 1e-5
        if not np.all(mask):
            n_bad = np.count_nonzero(~mask)
            logger.warning(
                f"{n_bad} rod(s) have near-zero orientation norm (<= 1e-5). "
                "Their directions are left unnormalized, which may lead to degenerate or invisible rods."
            )
        orient[mask] /= orient_norm[mask]
        
        n_rods = points.shape[0]
        half = 0.5 * length
        p_minus = points - half * orient
        p_plus  = points + half * orient
        endpoints = np.empty((2 * n_rods, 3), dtype=p_minus.dtype)
        endpoints[0::2] = p_minus
        endpoints[1::2] = p_plus
        
        lines = np.empty((n_rods, 3), dtype=np.int64)
        lines[:, 0] = 2
        lines[:, 1] = 2 * np.arange(n_rods)
        lines[:, 2] = 2 * np.arange(n_rods) + 1
        
        poly = pv.PolyData(endpoints, lines=lines.ravel())

        poly.point_data['radius'] = self._calc_radius
        if isinstance(self.opts.color, str) and self.opts.color == 'scalars':
            poly.point_data['opacity'] = self._calc_opacity
            poly.point_data['scalars'] = self._calc_scalars
        else:
            rgba_values = np.hstack([self._calc_color, self._calc_opacity.reshape(-1, 1)])
            poly.point_data['rgba'] = rgba_values
            
        logger.detail("Applying tube filter with dynamic radius scaling")
        mesh = poly.tube(
            scalars='radius', 
            n_sides=self.opts.sides, 
            absolute=True 
        )

        if self.opts.clip_geometry is not None:
            logger.detail("Applying spatial clipping to tube mesh")
            if isinstance(self.opts.clip_geometry, (list, tuple)) and len(self.opts.clip_geometry) == 6:
                mesh = mesh.clip_box(bounds=self.opts.clip_geometry, invert=False)
            elif hasattr(self.opts.clip_geometry, "points"):
                mesh = mesh.clip_surface(self.opts.clip_geometry, invert=False)

        object.__setattr__(self, "_calc_poly", poly)
        return mesh
    

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_commit_pre_opts(self, logger=None, **kwargs):
        
        is_needs_remesh, kwargs = super()._helper_commit_pre_opts(**kwargs)
        
        found, orient, kwargs = pop_exclusive(kwargs, "orient", "raw_orient")
        if found:
            try:
                object.__setattr__(self, "raw_orient", as_points(orient))
                is_needs_remesh = True
            except:
                logger.exception("Invalid input of orient for PlotRod.")
                logger.recovery("Ignore this modification in the following")
                
        if len(self.raw_orient) != len(self.raw_coords):
            raise ValueError(f"There are {len(self.raw_orient)} points for orientation, while {len(self.raw_coords)} points for positions.")
                    
        return is_needs_remesh, kwargs
    
    
    def act_commit(self, opts=None, **kwargs):
        is_needs_remesh, kwargs = self._helper_commit_pre_opts(**kwargs)
        kwargs = self._helper_merge_opts_kwargs(opts=opts, **kwargs)
        self._helper_commit_apply(is_needs_remesh, attr_resolve_extra=['length'], **kwargs)
    

