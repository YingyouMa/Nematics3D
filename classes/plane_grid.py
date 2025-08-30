import numpy as np
from dataclasses import asdict, dataclass, field
from typing import Optional, Literal

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.field import generate_coordinate_grid, apply_linear_transform
from Nematics3D.general import select_grid_in_box
from .opts import merge_opts_all
from Nematics3D.datatypes import Number, as_Number, Vect, as_Vect, Tensor, as_Tensor


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
