from dataclasses import dataclass
from typing import Callable, Sequence, Any, Mapping, ClassVar
import numpy as np
import pyvista as pv
from types import MappingProxyType

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import UNSET, Unset, as_bool, as_Number, as_str
from .plot_figure import PlotFigure
from .glyph import OptsGlyph, PlotGlyph
from Nematics3D.general import closest_point_on_polyline, fmt_value
from .qt.interact_tube import InteractTube
from Nematics3D.classes.host_base import HostBase

#! clip_geometry
#! light dark pbr

#! info log extra attr
#1 del
#! orphan figure

#! test
#! color invalid


@dataclass(slots=True, repr=False)
class OptsTube(OptsGlyph):

    # --- Geometry & Topology (Tube-specific) ---
    is_capping:             bool | Unset = UNSET
    smooth_iter:            int | Unset  = UNSET


    __descriptions__: ClassVar[Mapping[str, str]] = {
        **dict(OptsGlyph.__descriptions__),
        "is_capping":        "Whether to close the ends of the tube.",
        "smooth_iter":       "Path smoothing iterations to remove jagged edges.",
    }

    _validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        **dict(OptsGlyph._validators),
        "is_capping":        lambda v, d: as_bool(v, name=d),
        "smooth_iter":       lambda v, d: as_Number(v, name=d, is_int=True, value_range=(0, 1000), bounded=True),
    }

    _DEFAULTS_FROZEN: ClassVar[Mapping[str, Any]] = MappingProxyType({
        **dict(OptsGlyph._DEFAULTS_FROZEN),
        "is_capping":        True,
        "smooth_iter":       0,
    })



        
        
class PlotTube(PlotGlyph):

    __descriptions__ = {
        **dict(PlotGlyph.__descriptions__),
        "raw_name":     "The name identifier of the PlotTube instance",
        "raw_line_index": "Optional polyline membership indices.",
    }
    
    __slots__ = tuple(
            k for k, v in __descriptions__.items() 
            if not v.startswith("Property:") and k not in PlotGlyph.__slots__
        )
    
    
    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        coords: np.ndarray,
        name: str | None = None,
        name_replace: str = 'line',
        category: str = 'tube',
        figure: PlotFigure | None = None,
        opts: OptsTube | None = None,
        line_index: Sequence | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        logger = None,
        **kwargs
    ):
    
        category = as_str(category, name="The category of the PlotTube object", replace="tube")
        object.__setattr__(self, 'raw_category', category)

        super().__init__(
            coords=coords,
            opts_type=OptsTube,
            category=category,
            name=name,
            name_replace=name_replace,
            opts=opts,
            figure=figure,
            opts_defaults_override=opts_defaults_override,
            logger=logger,
            **kwargs,
        )
        
        # tube-specific
        try:
            line_index = self._helper_check_index(
                line_index,
                self.__descriptions__["raw_line_index"]
            )
        except:
            logger.exception("Invalid `line_index` input")
            logger.recovery("Set line_index=None in the following (no stop points within the tube)")
            line_index = None
        object.__setattr__(self, "raw_line_index", line_index)

        self._helper_init_end()
        self.act_set_interact_func(lambda: InteractTube(self, self.fig).show())

        
    def _helper_check_index(self, line_index, name):
        if line_index is None:
            return None
        try:
            line_index = np.asarray(line_index, dtype=int)
            if line_index.ndim != 1 or len(line_index) != self.raw_coords.shape[0]:
                raise ValueError(
                    f"`line_index` is {name}. "
                    f"It must be a ({self.raw_coords.shape[0]},) array. "
                    f"Got shape {line_index.shape} instead."
                )
            return line_index
        except (ValueError, TypeError):
            raise
        
        
        
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_build_poly(self, logger=None): 
        
        points = self.raw_coords
        idx = getattr(self, "raw_line_index", None)
        
        # Decide whether to treat the input as a single continuous polyline
        is_use_multi = (idx is None) or (len(np.unique(idx)) == 1)
        if is_use_multi:
            poly = pv.MultipleLines(points)
        else:
            logger.detail('Searching run boundaries: each run corresponds to one disconnected polyline')
            breaks = np.nonzero(idx[1:] != idx[:-1])[0] + 1
            starts = np.r_[0, breaks]
            ends   = np.r_[breaks, len(idx)]
        
            chunks = []
            for s, e in zip(starts, ends):
                k = e - s
                if k < 2:
                    logger.warning(f"Detect one invalid line segment with only one point at index={s}."
                                   "This will not be plotted.")
                chunks.append(np.r_[k, np.arange(s, e, dtype=np.int64)])
        
            if len(chunks) == 0:
                raise ValueError("line_index produced no valid line segments (each segment needs >=2 points).")
        
            lines = np.concatenate(chunks).astype(np.int64)
            poly = pv.PolyData(points, lines=lines)
        
        if self.opts.smooth_iter > 0:
            logger.detail(f"Smoothing path with {self.opts.smooth_iter} iterations")
            poly = poly.smooth(n_iter=self.opts.smooth_iter)
            
        object.__setattr__(self, "_calc_poly", poly)
        self._helper_set_poly(poly)

        
    @logging_and_warning_decorator(start_finish_level=5)    
    def _helper_build_mesh(self, logger=None):
        """
        Internal: Create the PyVista PolyData, apply smoothing/clipping, 
        and generate tube with dynamic or static radius.
        """

        poly = self._calc_poly    

        logger.detail("Applying tube filter with dynamic radius scaling")
        mesh = poly.tube(
            scalars='radius', 
            n_sides=self.opts.sides, 
            capping=self.opts.is_capping,
            absolute=True 
        )

        if self.opts.clip_geometry is not None:
            logger.detail("Applying spatial clipping to tube mesh")
            if isinstance(self.opts.clip_geometry, (list, tuple)) and len(self.opts.clip_geometry) == 6:
                mesh = mesh.clip_box(bounds=self.opts.clip_geometry, invert=False)
            elif hasattr(self.opts.clip_geometry, "points"):
                mesh = mesh.clip_surface(self.opts.clip_geometry, invert=False)

        return mesh
    
    def _helper_commit_pre_opts(self, **kwargs):
        
        is_new_topology, kwargs = super()._helper_commit_pre_opts(**kwargs)
        is_new_topology2 = HostBase._helper_commit_pop_raw(
            self, kwargs, "line_index",
            validator=self._helper_check_index
        )
        is_new_topology = is_new_topology or is_new_topology2
                    
        return is_new_topology, kwargs
    
    # Rewrite _helper_resolve_pick
    # To privide more specific information about tube
    def _helper_resolve_pick(self, picked_point):
        
        pos_close, msg, idx = super()._helper_resolve_pick(picked_point)
        x_param = idx / len(self.raw_coords) * 100
        msg_head = (
            f"The closest point on the tube is {fmt_value(pos_close)}, where: \n"
            f"The normalized position along the tube is {x_param:2f} \n"
            )
        try:
            smooth = self.owner.owner
            tgt = smooth.act_calc_tgt(x_param)
            msg_head += f"Local tangent: {fmt_value(tgt)} \n"
        except:
            pass
        msg = msg_head + msg
        
        pos = closest_point_on_polyline(picked_point, self.raw_coords)
        
        return pos, msg, idx
        
        
        

