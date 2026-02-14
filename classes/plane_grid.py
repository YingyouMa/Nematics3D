import numpy as np
from dataclasses import dataclass
from typing import Literal, Any, Mapping
from types import MappingProxyType

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.field import generate_fixed_step_grid, apply_linear_transform
from Nematics3D.general import select_grid_in_box
from .opts import merge_opts_all
from .host_base import OptsBase, HostBase
from Nematics3D.datatypes import Number, as_Number, Vect, as_Vect, Tensor, as_Tensor, as_str, UNSET, Unset
from .opts import cover_value

from .visual.plot_extent import PlotExtent
from .visual.plot_tube import OptsTube
from .visual.plot_figure import PlotFigure, OptsFigure
from .visual.plot_sphere import PlotSphere, OptsSphere

#!!! grid unit
#!!! shape
#!!! asdict
#!!! axis normal figdemo

# --- Plane Options ---
@dataclass(slots=True, repr=False)
class OptsPlaneGrid(OptsBase):
    normal:                 Vect(3) | Unset                             = UNSET
    spacing:                Number | Unset                              = UNSET
    spacing_extra:          Number | Unset                              = UNSET
    size:                   Number | Unset                              = UNSET
    size_extra:             Number | Unset                              = UNSET
    shape:                  Literal["circle", "rectangle"] | Unset      = UNSET #!!!!!!!!!
    origin:                 Vect(3) | Unset                             = UNSET
    alignment:              Literal["center", "bottom-left"] | Unset    = UNSET
    axis1:                  Vect(3) | None | Unset                      = UNSET
    corners_limit:          Tensor((8, 3)) | None | UNSET               = UNSET
    grid_offset:            Vect(3) | Unset                             = UNSET
    grid_transform:         Tensor((3, 3)) | Unset                      = UNSET

    __descriptions__ = {
        **(OptsBase.__descriptions__),
        "normal":           "normal of plane",
        "spacing":          "grid spacing along axis1",
        "spacing_extra":    "grid spacing along axis2",
        "size":             "size of plane",
        "size_extra":       "size of plane along axis2",
        "origin":           "origin of plane",
        "alignment":        "Grid reference point to be placed at 'origin' ('center' for geometric middle, 'bottom-left' for the first grid point [0,0])",
        "axis1":            "first in-plane axis",
        "corners_limit":    "bounding box corners (8×3 array)",
        "grid_offset":      "grid translation offset to map lattice indices to real-space coordinates",
        "grid_transform":   "grid transform matrix to map lattice indices to real-space coordinates (3x3 orthogonal matrix)",
        "shape":            "plane shape (circle or rectangle) NOT VALID IN CURRENT VERSION",
    }

    _validators = {
        **(OptsBase._validators),
        "normal":           lambda v, d: as_Vect(v, name=d, is_norm=True),
        "spacing":          lambda v, d: as_Number(v, name=d),
        "spacing_extra":    lambda v, d: None if v is None else as_Number(v, name=d),
        "size":             lambda v, d: as_Number(v, name=d),
        "size_extra":       lambda v, d: None if v is None else as_Number(v, name=d),
        "shape":            lambda v, d: as_str(
                                v, name=d,
                                pool=("circle", "rectangle"),
                                ),
        "corners_limit":    lambda v, d: as_Tensor(v, (8, 3), name=d),
        "origin":           lambda v, d: as_Vect(v, name=d),
        "alignment":        lambda v, d: as_str(
                                v, name=d,
                                pool=("center", "bottom-left"),
                                ),
        "axis1":            lambda v, d: None if v is None else as_Vect(v, name=d, is_norm=True),
        "grid_offset":      lambda v, d: as_Vect(v, name=d),
        "grid_transform":   lambda v, d: as_Tensor(v, (3, 3), name=d),
    }
    
    _DEFAULTS_FROZEN = MappingProxyType({
        **(OptsBase._DEFAULTS_FROZEN),
        "tag":              "plane grid options",
        "shape":            "rectangle",
        "spacing_extra":    None,
        "size_extra":       None,
        "origin":           (0,0,0),
        "alignment":        "center",
        "axis1":            None,
        "corners_limit":    None,
        "grid_offset":      (0,0,0),
        "grid_transform":   np.diag((1,1,1))
    })
    

class PlaneGrid(HostBase):
    
    __descriptions__ = {
        **dict(HostBase.__descriptions__),

        # ========== generated grids ==========
        "_entity_grid": "Selected 3D grid points after applying transforms and optional bounding-box filtering (array of shape N×3)",
        "_entity_grid_all": "Complete 3D grid points before filtering, reshaped as (num1 × num2 × 3)",
        "_entity_grid_int": "Integer lattice indices corresponding to 2D grid positions (num1 × num2 × 3)",

        # ========== calc (derived quantities) ==========
        "_calc_axis2": "The second in-plane axis which normal to both axis1 and normal.",
        "_calc_offset_real": "Offset vector applied to center the plane grid at the specified origin (3-vector) in lattice units",
        "_calc_box_mask": "the flag indicating whether point in self._entity_grid_all is inside the corners limit",
        "_calc_size":   "The actual size calculated based on opts.size",
        "_calc_size_extra": "The actual size_extra calculated based on opts.size and opts.size_extra",
        
        
        "_impl_field_ref": ("Quantity field evaluated on the 2D plane grid."
                            "To assess it, use .field or ._impl_field."),
        
        # ========== visualization / diagnostic ==========
        "_entity_fig_demo": "Diagnostic plot showing the generated 2D grid points, axes, and normal vector for verification.",
    }

    __slots__ = tuple(
            k for k, v in __descriptions__.items() 
            if not v.startswith("Property:") and k not in HostBase.__slots__
        )

    def __init__(self,
                 name: str | None = None,
                 name_replace: str = "2d grid",
                 opts: OptsPlaneGrid | None = None,
                 opts_defaults_override: Mapping[str, Any] | None = None,
                 **kwargs):

        super().__init__(
            OptsPlaneGrid,
            opts,
            opts_defaults_override,
            name=name,
            name_replace=name_replace,
            **kwargs
            )
        
        object.__setattr__(self, '_entity_fig_demo', None)
        object.__setattr__(self, '_impl_field_ref', None)
        object.__setattr__(self, '_entity_fig_demo', None)

        for name, value in {
            "normal": self.opts.normal,
            "spacing": self.opts.spacing,
            "size": self.opts.size,
        }.items():
            if value is UNSET:
                raise ValueError(
                    f"Missing required variable {name!r} to generate plane_grid"
                )
        self.opts.act_finalize(defaults=self._opts_defaults)

        
        self._helper_commit_apply_opts()
    
    @logging_and_warning_decorator()
    def _helper_commit_apply_opts(self, logger=None, **kwargs):

        with self.opts._helper_internal_update():
            cover_value(self.opts,
                        is_allow_cover_target_set=True,
                        is_allow_unset_source=False,
                        **kwargs
                        )
        

        logger.debug("Start to generate a new 2D grid.")

        space1 = self.opts.spacing
        space2 = space1 if self.opts.spacing_extra is None else self.opts.spacing_extra
        size1 = self.opts.size
        size2 = size1 if self.opts.size_extra is None else self.opts.size_extra
        origin = self.opts.origin
        normal = self.opts.normal
        axis1 = self.opts.axis1
        corners_limit = self.opts.corners_limit
        grid_transform = self.opts.grid_transform
        grid_offset = self.opts.grid_offset
        shape = self.opts.shape
        alignment = self.opts.alignment
        
        if axis1 is not None:
            dot_product = normal @ axis1
            if not np.isclose(dot_product, 0, atol=1e-8): 
                old_axis1 = axis1.copy()
                axis1 = axis1 - dot_product * normal
                axis1 /= np.linalg.norm(axis1)
                logger.warning(
                    f"Invalid geometry: axis1 is not perpendicular to normal (dot product: {dot_product:.4e}). "
                    f"Projecting original axis1 {old_axis1} onto the plane defined by normal {normal}. "
                    f"New orthonormal axis1: {axis1}."
                )
        else:
            from Nematics3D.general import rotation_matrix_from_vectors
            _rotation_matrix = rotation_matrix_from_vectors((0,0,1), normal)
            axis1 = _rotation_matrix @ np.array([1,0,0])
            logger.debug(f"axis1 not provided. Automatically generated a reference axis1 {axis1} "
                         f"perpendicular to normal {normal}.")
        
        axis2 = np.cross(normal, axis1)
        axis_both = np.array([axis1, np.cross(normal, axis1)])
        logger.debug(f"axis2={axis2}")
        
        logger.detail("Start to generate coordinate grids.")
        grid, grid_int, sizes = generate_fixed_step_grid(size1, size2, space1, space2)
        size1, size2 = sizes
        target_shape = np.shape(grid)[:2]
        grid_int = np.reshape(grid_int, (-1, 2))
        grid = np.reshape(grid, (-1, 2))
        grid = np.einsum("ai, ib -> ab", grid, axis_both)
        
        logger.detail("Translate the grid according to the origin.")
        if alignment == "bottom-left":
            offset = origin
        else:
            offset = origin - np.average(grid, axis=0)
        grid = grid + offset
        
        logger.detail("Perform linear transform into real coordinates.")
        grid = apply_linear_transform(
            grid, transform=grid_transform, offset=grid_offset
        )

        logger.debug(f"Select the grids inside the corners limit {corners_limit}.")
        grid_select, mask = select_grid_in_box(grid, corners_limit, is_return_mask=True)

        object.__setattr__(self, '_entity_grid', grid_select)
        object.__setattr__(self, '_entity_grid_all', np.reshape(grid, (*target_shape, 3)))
        object.__setattr__(self, '_entity_grid_int', grid_int)
        object.__setattr__(self, '_calc_offset_real', offset)
        object.__setattr__(self, '_calc_axis2', axis2)
        object.__setattr__(self, '_calc_box_mask', mask)
        object.__setattr__(self, '_calc_size', size1)
        object.__setattr__(self, '_calc_size_extra', size2)
        object.__setattr__(self.opts, "axis1", axis1)
            
        if self._entity_fig_demo:
            self._entity_fig_demo['grid'].raw_coords = self._entity_grid
            self._entity_fig_demo['origin'].raw_coords = self.opts.origin
            
        if self.field:
            self.field._helper_commit()
            
        self._helper_trigger_sync_batch(**kwargs)
        
        
    
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        msg = f"{cls_name}, with normal={self.opts.normal}, axis1={self.opts.axis1}, origin={self.opts.origin} at {self.opts.alignment}"
        return msg
    
    def __iter__(self):
        return iter(self._entity_grid)
    
    def __getitem__(self, idx):
        return self._entity_grid[idx]
    
    def __array__(self, dtype=None):
        arr = self._entity_grid
        return np.asarray(arr, dtype=dtype) if dtype is not None else arr
    
    def __call__(self):
        return self._entity_grid
    
    @property
    def field(self):
        ref = self._impl_field_ref
        return ref() if ref is not None else None
    
    _impl_field = field
    
    def act_debug_plot(self,
                       opts_extent: OptsTube | None = None,
                       opts_points: OptsSphere | None = None,
                       opts_figure: OptsFigure | None = None,
                       opts_origin: OptsSphere | None = None,
                       **kwargs
                       ):
    
        if opts_extent is None:
            opts_extent = OptsTube(category="plane_grid_test", name="grid_extent")
        if opts_points is None:
            opts_points = OptsSphere(category="plane_grid_test", name="grid")
            opts_points.act_finalize()
        if opts_figure is None:
            opts_figure = OptsFigure()
        if opts_origin is None:
            opts_origin = OptsSphere(color=(1,0,0), category="plane_grid_test", name="origin")
            opts_origin.radius = 1.2 * opts_points.radius
            
        merge = merge_opts_all(
            {
                "figure_": opts_figure, 
                "point_": opts_points,
                "extent_": opts_extent,
                "origin_": opts_origin
            },
            kwargs, type(self).__name__)
        
        opts_figure = merge["figure_"]
        opts_points = merge["point_"]
        opts_extent = merge["extent_"]
        opts_origin = merge["origin_"]
        
        figure = PlotFigure(opts=opts_figure)
        PlotSphere(coords=self._entity_grid, opts=opts_points, figure=figure)
        PlotSphere(coords=self.opts.origin, opts=opts_origin, figure=figure)
        if self.opts.corners_limit is not None:
            PlotExtent(corners=self.opts.corners_limit, opts=opts_extent, figure=figure)
            
        self._entity_fig_demo = figure
            
        return figure
        
        
    # @logging_and_warning_decorator()
    # def act_log_parameters(self, is_return: bool = False, logger=None) -> None:
    #     """
    #     Log internal filter and output parameters for inspection.

    #     This is the standard logging interface used in this library, which
    #     can be redirected to console or to a file depending on the logger
    #     configuration and the behavior of ``logging_and_warning_decorator``.

    #     All attributes listed in ``__descriptions__`` are included,
    #     formatted in a single log entry with a clear separator.
    #     """
    #     lines = []
    #     lines.append("-------------- PlaneGrid Parameters --------------")

    #     lines.append("PlaneGrid parameters and results:")
    #     for attr in self.__slots__:
    #         desc = self.__descriptions__.get(attr, "(no description)")
    #         value = getattr(self, attr, None)

    #         if attr in ("opts.axis1", "opts.spacing", "opts.spacing_extra"):
    #             lines.append(f"  {attr}: {value!r}  # {desc} (derived final value)")
    #         else:
    #             lines.append(f"  {attr}: {value!r}  # {desc}")

    #     lines.append("-----------------------------------------------------")

    #     msg = "\n".join(lines)

    #     if is_return:
    #         return msg
    #     else:
    #         logger.info(msg)
            
    # def act_save(self, path: str = "save/PlaneGrid.json") -> None:
    #     import json
    #     import os
    #     data = asdict(self._opts_all)
    #     path = as_str(path, name="The path to save PlaneGrid")
    #     os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    #     with open(path, "w", encoding="utf-8") as f:
    #         json.dump(data, f, indent=2)
            
    # @classmethod
    # def act_load(cls, path: str = "save/PlaneGrid.json") -> "PlaneGrid":
    #     import json  
    #     path = as_str(path, name="The path to load PlaneGrid")
    #     with open(path, "r", encoding="utf-8") as f:
    #         data = json.load(f)
    #     opts = OptsPlaneGrid(**data)
    #     return cls(opts=opts)
    
    # def act_copy(self) -> "PlaneGrid":
    #     import copy
    #     opts_new = copy.deepcopy(self._opts_all)
    #     return self.__class__(opts=opts_new)
    
    
