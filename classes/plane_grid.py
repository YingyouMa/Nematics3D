import numpy as np
from dataclasses import asdict, dataclass, field
from typing import Optional, Literal

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.field import generate_fixed_step_grid, apply_linear_transform
from Nematics3D.general import select_grid_in_box
from .opts import merge_opts_all
from Nematics3D.datatypes import Number, as_Number, Vect, as_Vect, Tensor, as_Tensor, as_str


# --- Plane Options ---
@dataclass(slots=True)
class OptsPlaneGrid:
    normal: Vect(3) | None = None
    spacing: Number | None = None
    spacing_extra: Number | None = None
    size: Number | None = None
    size_extra: Number | None = None
    shape: Literal["circle", "rectangle"] = "rectangle" #!!!!!!!!!
    origin: Vect(3) = (0, 0, 0)
    axis1: Vect(3) | None = None
    corners_limit: Tensor((8, 3)) | None = None
    grid_offset: Vect(3) = (0, 0, 0)
    grid_transform: Tensor((3, 3)) = ((1,0,0),(0,1,0),(0,0,1))

    __descriptions__ = {
        "normal": "normal of plane",
        "spacing": "grid spacing along axis1",
        "spacing_extra": "grid spacing along axis2",
        "size": "size of plane",
        "size_extra": "size of plane along axis2",
        "origin": "origin of plane",
        "axis1": "first in-plane axis",
        "corners_limit": "bounding box corners (8×3 array)",
        "grid_offset": "grid translation offset to map lattice indices to real-space coordinates",
        "grid_transform": "grid transform matrix to map lattice indices to real-space coordinates (3x3 orthogonal matrix)",
        "shape": "plane shape (circle or rectangle)",
    }

    _validators = {
        "normal": lambda self, v: (
            None
            if v is None
            else as_Vect(v, name=self.__descriptions__["normal"], is_norm=True)
        ),
        "origin": lambda self, v: as_Vect(v, name=self.__descriptions__["origin"]),
        "grid_offset": lambda self, v: as_Vect(
            v, name=self.__descriptions__["grid_offset"]
        ),
        "grid_transform": lambda self, v: as_Tensor(
            v, (3, 3), name=self.__descriptions__["grid_transform"]
        ),
        "axis1": lambda self, v: (
            None
            if v is None
            else as_Vect(v, name=self.__descriptions__["axis1"], is_norm=True)
        ),
        "spacing": lambda self, v: (
            None if v is None else as_Number(v, name=self.__descriptions__["spacing"])
        ),
        "spacing_extra": lambda self, v: (
            None if v is None else as_Number(v, name=self.__descriptions__["spacing_extra"])
        ),
        "size": lambda self, v: (
            None if v is None else as_Number(v, name=self.__descriptions__["size"])
        ),
        "size_extra": lambda self, v: (
            None if v is None else as_Number(v, name=self.__descriptions__["size_extra"])
        ),
        "corners_limit": lambda self, v: (
            None
            if v is None
            else as_Tensor(v, (8, 3), name=self.__descriptions__["corners_limit"])
        ),
        "shape": lambda self, v: (
            v
            if v in ("circle", "rectangle")
            else (_ for _ in ()).throw(
                ValueError(
                    f"Invalid {self.__descriptions__['shape']}: {v!r}. "
                    f"Allowed values: 'circle', 'rectangle'"
                )
            )
        ),
    }

    def __setattr__(self, key, value):
        if key in self._validators:
            value = self._validators[key](self, value)
        object.__setattr__(self, key, value)

class PlaneGrid:
    
    __descriptions__ = {
        # ========== options mirrored onto the instance ==========
        "opts": "Dataclass OptsPlaneGrid storing all user-specified and default options for the plane grid",

        # ========== generated grids ==========
        "_entities_grid": "Selected 3D grid points after applying transforms and optional bounding-box filtering (array of shape N×3)",
        "_entities_grid_all": "Complete 3D grid points before filtering, reshaped as (num1 × num2 × 3)",
        "_entities_grid_int": "Integer lattice indices corresponding to 2D grid positions (array of shape N×2)",

        # ========== calc (derived quantities) ==========
        "_calc_offset_real": "Real-space offset vector applied to center the plane grid at the specified origin (3-vector)",
    }

    __slots__ = tuple(__descriptions__.keys())

    def __init__(self,
                 opts: OptsPlaneGrid | None = None,
                 **kwargs):
        
        if opts is None:
            opts = OptsPlaneGrid()
        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]

        for name, value in {
            "normal": opts.normal,
            "spacing": opts.spacing,
            "size": opts.size,
        }.items():
            if value is None:
                raise ValueError(
                    f"Missing required variable {name} to generate plane_grid"
                )
        
        if opts.spacing_extra == None:
            opts.spacing_extra = opts.spacing
        if opts.size_extra == None:
            opts.size_extra = opts.size
        
        object.__setattr__(self, "opts", opts)

        self.act_commit()
    
    @logging_and_warning_decorator()
    def act_commit(self, logger=None, **kwargs):
        
        logger.debug("Start to generate a new 2D grid.")
        
        self.opts = merge_opts_all({"": self.opts}, kwargs, type(self).__name__)[""]

        space1 = self.opts.spacing
        space2 = self.opts.spacing_extra
        size1 = self.opts.size
        size2 = self.opts.size_extra
        origin = self.opts.origin
        normal = self.opts.normal
        axis1 = self.opts.axis1
        corners_limit = self.opts.corners_limit
        grid_transform = self.opts.grid_transform
        grid_offset = self.opts.grid_offset
        shape = self.opts.shape
        
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

        if axis1 is None:
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
        grid_int = np.reshape(grid_int, (-1, 2))
        grid = np.reshape(grid, (-1, 2))
        grid = np.einsum("ai, ib -> ab", grid, axis_both)

        logger.detail("Transparent the grid to make its center at origin.")
        offset = -np.average(grid, axis=0) + origin
        grid = grid + offset
        
        logger.detail("Perform linear transform into real coordinates.")
        grid = apply_linear_transform(
            grid, transform=grid_transform, offset=grid_offset
        )

        if corners_limit is not None:
            logger.debug(f"Select the grids inside the corners limit {corners_limit}.")
            grid_select = select_grid_in_box(
                grid, corners_limit=corners_limit, logger=logger
            )
        else:
            grid_select = grid

        self._entities_grid = grid_select
        self._entities_grid_all = np.reshape(grid, (-1, 3))
        self._entities_grid_int = grid_int
        self._calc_offset_real = offset
        self.opts.size = size1
        self.opts.size_extra = size2
        self.opts.axis1 = axis1


    def __str__(self) -> str:
        header = f"<{self.__class__.__name__} object>"
        return header + "\n" + self.act_log_parameters(is_return=True) 
    
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        msg = f"{cls_name}, with normal={self.opts_normal}, axis1={self.opts_axis1}, origin={self.opts_origin}"
        return msg
    
    def __iter__(self):
        return iter(self._entities_grid)
    
    def __getitem__(self, idx):
        return self._entities_grid[idx]
    
    def __array__(self, dtype=None):
        arr = self._entities_grid
        return np.asarray(arr, dtype=dtype) if dtype is not None else arr
    
        
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

    #         if attr in ("opts_axis1", "opts_spacing", "opts_spacing_extra"):
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
    
    
