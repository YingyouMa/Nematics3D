import numpy as np
from dataclasses import asdict, dataclass, field
from typing import Optional, Literal

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.field import generate_coordinate_grid, apply_linear_transform
from Nematics3D.general import select_grid_in_box
from .opts import merge_opts_all
from Nematics3D.datatypes import Number, as_Number, Vect, as_Vect, Tensor, as_Tensor, as_str


# --- Plane Options ---
@dataclass(slots=True)
class OptsPlaneGrid:
    normal: Optional[Vect(3)] = None
    spacing1: Optional[Number] = None
    spacing2: Optional[Number] = None
    size: Optional[Number] = None
    shape: Literal["circle", "rectangle"] = "rectangle"
    origin: Vect(3) = (0, 0, 0)
    axis1: Optional[Vect(3)] = None
    corners_limit: Optional[np.ndarray] = None
    grid_offset: Vect(3) = (0, 0, 0)
    grid_transform: Tensor((3, 3)) = field(default_factory=lambda: np.eye(3))

    __descriptions__ = {
        "normal": "normal of plane",
        "spacing1": "grid spacing along axis1",
        "spacing2": "grid spacing along axis2",
        "size": "size of plane",
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
        "spacing1": lambda self, v: (
            None if v is None else as_Number(v, name=self.__descriptions__["spacing1"])
        ),
        "spacing2": lambda self, v: (
            None if v is None else as_Number(v, name=self.__descriptions__["spacing2"])
        ),
        "size": lambda self, v: (
            None if v is None else as_Number(v, name=self.__descriptions__["size"])
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
        "_opts_all": "Dataclass OptsPlaneGrid storing all user-specified and default options for the plane grid",
        "opts_spacing1": "Final spacing of grid along axis1 (float, computed from size and shape)",
        "opts_spacing2": "Final spacing of grid along axis2 (float, computed from size and shape)",
        "opts_axis1": "Normalized axis1 vector (3-vector), adjusted to be perpendicular to normal",
        "opts_normal": "Normal vector of the plane (3-vector)",
        "opts_origin": "Origin of the plane in real-space coordinates (3-vector)",
        "opts_size": "Extent (size) of the plane in real-space coordinates (float)",
        "opts_shape": "Shape of the plane grid ('circle' or 'rectangle')",
        "opts_corners_limit": "Optional bounding-box corners for grid filtering (array of shape 8×3)",
        "opts_grid_offset": "Translation offset applied to grid coordinates (3-vector)",
        "opts_grid_transform": "Linear transformation matrix applied to grid coordinates (3×3)",

        # ========== generated grids ==========
        "_entities_grid": "Selected 3D grid points after applying transforms and optional bounding-box filtering (array of shape N×3)",
        "_entities_grid_all": "Complete 3D grid points before filtering, reshaped as (num1 × num2 × 3)",
        "_entities_grid_int": "Integer lattice indices corresponding to 2D grid positions (array of shape N×2)",

        # ========== calc (derived quantities) ==========
        "_calc_offset_real": "Real-space offset vector applied to center the plane grid at the specified origin (3-vector)",
    }

    __slots__ = tuple(__descriptions__.keys())

    @logging_and_warning_decorator
    def __init__(self, opts=OptsPlaneGrid(), logger=None, **kwargs):

        for name, value in {
            "normal": opts.normal,
            "spacing1": opts.spacing1,
            "spacing2": opts.spacing2,
            "size": opts.size,
        }.items():
            if value is None:
                raise ValueError(
                    f"Missing required variable {name} to generate plane_grid"
                )

        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]

        self._opts_all = opts

        self.act_commit(logger=logger)

    def act_commit(
        self,
        logger=None,
        **kwargs,
    ):
        
        self._opts_all = merge_opts_all({"": self._opts_all}, kwargs, type(self).__name__)[""]

        for key, value in asdict(self._opts_all).items():
            setattr(self, f"opts_{key}", value)

        space1 = self._opts_all.spacing1
        space2 = self._opts_all.spacing2
        size = self._opts_all.size
        origin = self._opts_all.origin
        normal = self._opts_all.normal
        axis1 = self._opts_all.axis1
        corners_limit = self._opts_all.corners_limit
        grid_transform = self._opts_all.grid_transform
        grid_offset = self._opts_all.grid_offset
        shape = self._opts_all.shape

        num1 = int(size / space1)
        num2 = int(size / space2)

        # space1 = (size-1)/(num1-1)
        # space1 = (size-1)/(num2-1)

        if axis1 is not None:
            if normal @ axis1 != 0:
                axis1 = axis1 - axis1 @ normal * normal
                axis1 /= np.linalg.norm(axis1)
                msg = "normal must be perpendicular to axis1.\n"
                msg += f"Got {normal} and {axis1}. \n"
                msg += "Discard the component aligned with normal along axis1.\n"
                msg += f"Use axis1={axis1} in the following"
                logger.info(msg)
        if axis1 is None:
            from Nematics3D.general import rotation_matrix_from_vectors
            _rotation_matrix = rotation_matrix_from_vectors((0,0,1), normal)
            axis1 = _rotation_matrix @ np.array([1,0,0])
        
        axis_both = np.array([axis1, np.cross(normal, axis1)])

        source_shape = (size, size)
        target_shape = (num1, num2)

        grid, grid_int, spaces = generate_coordinate_grid(source_shape, target_shape)
        grid_int = np.reshape(grid_int, (-1, 2))
        grid = np.reshape(grid, (-1, 2))
        grid = np.einsum("ai, ib -> ab", grid, axis_both)

        offset = -np.average(grid, axis=0) + origin
        grid = grid + offset
        grid = apply_linear_transform(
            grid, transform=grid_transform, offset=grid_offset
        )

        if corners_limit is not None:
            grid_select = select_grid_in_box(
                grid, corners_limit=corners_limit, logger=logger
            )
        else:
            grid_select = grid

        self._entities_grid = [grid_select]
        self._entities_grid_all = [np.reshape(grid, (*target_shape, 3))]
        self._entities_grid_int = [grid_int]
        self._calc_offset_real = offset
        self.opts_spacing1 = spaces[0]
        self._opts_all.spacing1 = spaces[0]
        self.opts_spacing2 = spaces[1]
        self._opts_all.spacing2 = spaces[1]
        self.opts_axis1 = axis1
        self._opts_all.axis1 = axis1
        
    @logging_and_warning_decorator()
    def act_log_parameters(self, is_return: bool = False, logger=None) -> None:
        """
        Log internal filter and output parameters for inspection.

        This is the standard logging interface used in this library, which
        can be redirected to console or to a file depending on the logger
        configuration and the behavior of ``logging_and_warning_decorator``.

        All attributes listed in ``__descriptions__`` are included,
        formatted in a single log entry with a clear separator.
        """
        lines = []
        lines.append("-------------- PlaneGrid Parameters --------------")

        lines.append("PlaneGrid parameters and results:")
        for attr in self.__slots__:
            desc = self.__descriptions__.get(attr, "(no description)")
            value = getattr(self, attr, None)

            if attr in ("opts_axis1", "opts_spacing1", "opts_spacing1"):
                lines.append(f"  {attr}: {value!r}  # {desc} (derived final value)")
            else:
                lines.append(f"  {attr}: {value!r}  # {desc}")

        lines.append("-----------------------------------------------------")

        msg = "\n".join(lines)

        if is_return:
            return msg
        else:
            logger.info(msg)
            
    def act_save(self, path: str = "save/PlaneGrid.json") -> None:
        import json
        import os
        data = asdict(self._opts_all)
        path = as_str(path, name="The path to save PlaneGrid")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def act_load(cls, path: str = "save/PlaneGrid.json") -> "PlaneGrid":
        import json  
        path = as_str(path, name="The path to load PlaneGrid")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        opts = OptsPlaneGrid(**data)
        return cls(opts=opts)
    
    def act_copy(self) -> "PlaneGrid":
        import copy
        opts_new = copy.deepcopy(self._opts_all)
        return self.__class__(opts=opts_new)

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
