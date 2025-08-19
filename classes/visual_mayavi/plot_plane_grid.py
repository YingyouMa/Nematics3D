import numpy as np
from typing import Tuple, Optional, List, Literal

from Nematics3D.logging_decorator import logging_and_warning_decorator, Logger
from Nematics3D.datatypes import Vect3D, as_Vect3D, as_QField5
from Nematics3D.field import generate_coordinate_grid, apply_linear_transform
from Nematics3D.general import select_grid_in_box


class PlotPlaneGrid:

    @logging_and_warning_decorator
    def __init__(
        self,
        normal: Vect3D,
        space1: float,
        space2: float,
        size: float,
        shape: Literal["circle", "rectangle"] = "rectangle",
        origin: Vect3D = (0, 0, 0),
        axis1: Optional[Vect3D] = None,
        corners_limit: Optional[np.ndarray] = None,
        grid_offset: Vect3D = np.array([0, 0, 0]),
        grid_transform: np.ndarray = np.eye(3),
        logger=None,
    ):

        self.update_grid(
            normal,
            shape,
            space1,
            space2,
            size,
            origin=origin,
            axis1=axis1,
            corners_limit=corners_limit,
            grid_offset=grid_offset,
            grid_transform=grid_transform,
            logger=logger,
        )

    def update_grid(
        self,
        normal: Vect3D,
        shape: Literal["circle", "rectangle"],
        space1: float,
        space2: float,
        size: float,
        origin: Vect3D = (0, 0, 0),
        axis1: Optional[Vect3D] = None,
        corners_limit: Optional[np.ndarray] = None,
        grid_offset: Vect3D = np.array([0, 0, 0]),
        grid_transform: np.ndarray = np.eye(3),
        logger=None,
    ):

        space1 = float(space1)
        space2 = float(space2)

        num1 = int(size / space1)
        num2 = int(size / space2)

        # space1 = (size-1)/(num1-1)
        # space1 = (size-1)/(num2-1)

        origin = as_Vect3D(origin)

        if shape not in ["circle", "rectangle"]:
            msg = f">>> Input shape must either be 'circle' or 'rectangle'. Got {shape} instead.\n"
            msg += "Use 'rectangle' in the following."
            shape = "rectangle"
            logger.warning(msg)

        normal = as_Vect3D(normal, is_norm=True)

        if axis1 is not None:
            axis1 = as_Vect3D(axis1, is_norm=True)
            if normal @ axis1 != 0:
                msg = "normal must be perpendicular to axis1.\n"
                msg += "Discard the component aligned with normal along axis1 in the following."
                logger.info(msg)
        if axis1 is None:
            axis1 = np.random.randn(3)
            axis1 /= np.linalg.norm(axis1)

        axis1 = axis1 - axis1 @ normal * normal
        axis1 /= np.linalg.norm(axis1)
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

        grid_select = select_grid_in_box(
            grid, corners_limit=corners_limit, logger=logger
        )

        self._grid = grid_select
        self._axis1 = axis1
        self._normal = normal
        self._origin = origin
        self._shape = shape
        self._space1 = spaces[0]
        self._space2 = spaces[1]
        self._grid_all = np.reshape(grid, (*target_shape, 3))
        self._offset = offset
        self._grid_int = grid_int
        self._grid_transform = grid_transform
        self._grid_offset = grid_offset
