import numpy as np
from dataclasses import dataclass, field, fields
from typing import Literal, Any, Mapping
import weakref
from contextlib import contextmanager
from types import MappingProxyType

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.field import generate_fixed_step_grid, apply_linear_transform
from Nematics3D.general import select_grid_in_box
from .opts import merge_opts_all, build_defaults_with_override
from Nematics3D.datatypes import Number, as_Number, Vect, as_Vect, Tensor, as_Tensor, as_str, UNSET, Unset
from .class_function import cover_value

from .visual.plot_extent import PlotExtent
from .visual.plot_tube import OptsTube
from .visual.plot_figure import PlotFigure, OptsFigure
from .visual.plot_sphere import PlotSphere, OptsSphere

#!!! grid unit
#!!! shape
#!!! asdict
#!!! axis normal figdemo

# --- Plane Options ---
@dataclass(slots=True)
class OptsPlaneGrid:
    name: str | Unset = UNSET
    normal: Vect(3) | Unset = UNSET
    spacing: Number | Unset = UNSET
    spacing_extra: Number | Unset = UNSET
    size: Number | Unset = UNSET
    size_extra: Number | Unset = UNSET
    shape: Literal["circle", "rectangle"] | Unset = UNSET #!!!!!!!!!
    origin: Vect(3) | Unset = UNSET
    alignment: Literal["center", "bottom-left"] | Unset = UNSET
    axis1: Vect(3) | None | Unset = UNSET
    corners_limit: Tensor((8, 3)) | None | UNSET = UNSET
    grid_offset: Vect(3) | Unset = UNSET
    grid_transform: Tensor((3, 3)) | Unset = UNSET
    
    _internal_owner_ref: weakref.ReferenceType | None = field(default=None, init=False, repr=False)
    _state_is_syncro: bool = field(default=False, init=False, repr=False)
    _state_functioning: bool = field(default=False, init=False, repr=False)
    _defaults: dict[str, Any] = field(init=False, repr=False)

    __descriptions__ = {
        "name": "name of this 2D grid",
        "normal": "normal of plane",
        "spacing": "grid spacing along axis1",
        "spacing_extra": "grid spacing along axis2",
        "size": "size of plane",
        "size_extra": "size of plane along axis2",
        "origin": "origin of plane",
        "alignment": "Grid reference point to be placed at 'origin' ('center' for geometric middle, 'bottom-left' for the first grid point [0,0])",
        "axis1": "first in-plane axis",
        "corners_limit": "bounding box corners (8×3 array)",
        "grid_offset": "grid translation offset to map lattice indices to real-space coordinates",
        "grid_transform": "grid transform matrix to map lattice indices to real-space coordinates (3x3 orthogonal matrix)",
        "shape": "plane shape (circle or rectangle) NOT VALID IN CURRENT VERSION",
    }

    _validators = {
        "name": lambda self, v, d: as_str(v, name=d),
        "normal": lambda self, v, d: as_Vect(v, name=d, is_norm=True),
        "spacing": lambda self, v, d: as_Number(v, name=d),
        "spacing_extra": lambda self, v, d: None if v is None else as_Number(v, name=d),
        "size": lambda self, v, d: as_Number(v, name=d),
        "size_extra": lambda self, v, d: None if v is None else as_Number(v, name=d),
        "corners_limit": lambda self, v, d: as_Tensor(v, (8, 3), name=d),
        "origin": lambda self, v, d: as_Vect(v, name=d),
        "alignment": lambda self, v, d: as_str(
            v, name=d,
            pool=("center", "bottom-left"),
        ),
        "axis1": lambda self, v, d: None if v is None else as_Vect(v, name=d, is_norm=True),
        "grid_offset": lambda self, v, d: as_Vect(v, name=d),
        "grid_transform": lambda self, v, d: as_Tensor(v, (3, 3), name=d),
    }
    
    _DEFAULTS_FROZEN = MappingProxyType({
        "name":                 "2d grid",
        "shape":                "rectangle",
        "spacing_extra":        None,
        "size_extra":           None,
        "origin":               (0,0,0),
        "alignment":            "center",
        "axis1":                None,
        "corners_limit":        None,
        "grid_offset":          (0,0,0),
        "grid_transform":       np.diag((1,1,1))
    })
    
    def __post_init__(self):
        object.__setattr__(self, "_defaults", dict(self._DEFAULTS_FROZEN))
    
    @logging_and_warning_decorator(start_finish_level=5)
    def __setattr__(self, key, value, logger=None):
        if value is not UNSET and key in self._validators:
            desc = f'{key!r}: {self.__descriptions__.get(key)}'
            try:
                value = self._validators[key](self, value, desc)
                object.__setattr__(self, key, value)
            except:
                logger.exception(f"Assignment to {key!r} failed")
                if self._state_functioning:
                    logger.recovery("Automatically ignore this modification")
                else:
                    logger.recovery("Reset this assignment to UNSET.")
                    object.__setattr__(self, key, UNSET)
        else:
            object.__setattr__(self, key, value)
                    
        if not key.startswith("_") and getattr(self, '_state_is_syncro', False):
            if hasattr(self, "_internal_owner_ref") and self._internal_owner_ref is not None:
                owner = self._internal_owner_ref()
                owner.act_commit()
                    
    @contextmanager
    def _helper_internal_update(self):
        state_current = getattr(self, '_state_is_syncro', False)
        object.__setattr__(self, "_state_is_syncro", False)
        try:
            yield
        finally:
            object.__setattr__(self, "_state_is_syncro", state_current)
            
    def act_asdict(self, is_include_UNSET=False):
        result = {}
        for key in self.__descriptions__.keys():
            value = getattr(self, key, UNSET)
            if not is_include_UNSET and value is UNSET:
                continue
            result[key] = getattr(self, key)
        return result
    
    def act_finalize(self, defaults: Mapping[str, Any] | None = None):
        """
        Resolve all UNSET fields using:
          1) the provided `defaults` mapping (higher priority), then
          2) the class-level `_DEFAULTS_FROZEN` mapping.

        This must be called before visualization. After finalization, the opts
        should be treated as ready-to-use (no more defaults resolution).
        """
        if getattr(self, "_state_functioning", False):
            raise RuntimeError("OptsTube has already been finalized.")

        defaults = {} if defaults is None else dict(defaults)

        for f in fields(self):
            k = f.name
            if k.startswith("_"):
                continue  # internal fields are not finalized

            if getattr(self, k) is UNSET:
                v = defaults.get(k, self._DEFAULTS_FROZEN.get(k, UNSET))
                if v is UNSET:
                    raise KeyError(f"Missing default for field {k!r}.")
                setattr(self, k, v)  # runs validators

        object.__setattr__(self, "_state_functioning", True)

class PlaneGrid:
    
    __descriptions__ = {
        # ========== options mirrored onto the instance ==========
        "opts": "Dataclass OptsPlaneGrid storing all user-specified and default options for the plane grid",
        "opts_defaults": "Some default option settings for the 2D plane grid",

        # ========== generated grids ==========
        "_entities_grid": "Selected 3D grid points after applying transforms and optional bounding-box filtering (array of shape N×3)",
        "_entities_grid_all": "Complete 3D grid points before filtering, reshaped as (num1 × num2 × 3)",
        "_entities_grid_int": "Integer lattice indices corresponding to 2D grid positions (num1 × num2 × 3)",

        # ========== calc (derived quantities) ==========
        "_calc_axis2": "The second in-plane axis which normal to both axis1 and normal.",
        "_calc_offset_real": "Offset vector applied to center the plane grid at the specified origin (3-vector) in lattice units",
        "_calc_box_mask": "the flag indicating whether point in self._entities_grid_all is inside the corners limit",
        
        # ========== visualization / diagnostic ==========
        "_entities_fig_demo": "Diagnostic plot showing the generated 2D grid points, axes, and normal vector for verification.",
        
        "_internal_owner_ref": ("A weak reference to the object associated with this grid."
                                "To access it, use .owner or ._internal_owner."),
    }

    __slots__ = tuple(__descriptions__.keys()) + ("__weakref__",)

    def __init__(self,
                 opts: OptsPlaneGrid | None = None,
                 opts_defaults_override: Mapping[str, Any] | None = None,
                 **kwargs):
        
        opts_defaults = build_defaults_with_override(
                            OptsTube._DEFAULTS_FROZEN,
                            opts_defaults_override,
                            name="OptsPlaneGrid",
                        )
        object.__setattr__(self, "opts_defaults", opts_defaults)
        
        if opts is None:
            opts = OptsPlaneGrid()
        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]

        for name, value in {
            "normal": opts.normal,
            "spacing": opts.spacing,
            "size": opts.size,
        }.items():
            if value is UNSET:
                raise ValueError(
                    f"Missing required variable {name} to generate plane_grid"
                )
        
        object.__setattr__(self, "opts", opts)
        self._entities_fig_demo = None

        self.act_commit()
    
    @logging_and_warning_decorator()
    def act_commit(self, 
                   opts: OptsPlaneGrid | None = None, 
                   logger=None, 
                   **kwargs):
        
        if opts is None:
            opts = OptsPlaneGrid()
        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
        cover_value(self.opts,
                    is_allow_cover_target_set=True,
                    is_allow_unset_source=False,
                    **opts.act_asdict()
                    )
        
        if not self.opts._state_functioning:
            self.opts.act_finalize()
            
        object.__setattr__(self.opts, "_internal_owner_ref", weakref.ref(self))
        object.__setattr__(self.opts, "_state_is_syncro", True)
        
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
        
        logger.detail("Transparent the grid to make its center at origin.")
        
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

        self._entities_grid = grid_select
        self._entities_grid_all = np.reshape(grid, (*target_shape, 3))
        self._entities_grid_int = grid_int
        self._calc_offset_real = offset
        self._calc_axis2 = axis2
        self._calc_box_mask = mask
        
        with self.opts._helper_internal_update():
            self.opts.size = size1
            self.opts.size_extra = size2 if self.opts.size_extra is not None else None
            self.opts.axis1 = axis1
            
        if self._entities_fig_demo:
            self._entities_fig_demo['grid'].raw_coords = self._entities_grid
            self._entities_fig_demo['origin'].raw_coords = self.opts.origin
            # self._entities_fig_demo['grid_extent'].raw_coords = 
            
        if hasattr(self, "_internal_owner_ref") and self.owner:
            self.owner._helper_commit()


    def __str__(self) -> str:
        header = f"<{self.__class__.__name__} object>"
        return header + "\n" + self.act_log_parameters(is_return=True) 
    
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        msg = f"{cls_name}, with normal={self.opts.normal}, axis1={self.opts.axis1}, origin={self.opts.origin} at {self.opts.alignment}"
        return msg
    
    def __iter__(self):
        return iter(self._entities_grid)
    
    def __getitem__(self, idx):
        return self._entities_grid[idx]
    
    def __array__(self, dtype=None):
        arr = self._entities_grid
        return np.asarray(arr, dtype=dtype) if dtype is not None else arr
    
    def __call__(self):
        return self._entities_grid
    
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
        PlotSphere(coords=self._entities_grid, opts=opts_points, figure=figure)
        PlotSphere(coords=self.opts.origin, opts=opts_origin, figure=figure)
        if self.opts.corners_limit is not None:
            PlotExtent(corners=self.opts.corners_limit, opts=opts_extent, figure=figure)
            
        self._entities_fig_demo = figure
            
        return figure
    
    @property
    def _internal_owner(self):
        return self._internal_owner_ref()
    
    @property
    def owner(self):
        return self._internal_owner_ref()
        
        
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
    
    
