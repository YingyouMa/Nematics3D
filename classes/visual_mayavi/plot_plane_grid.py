import numpy as np
from typing import Tuple, Optional, List, Literal

from Nematics3D.logging_decorator import logging_and_warning_decorator, Logger
from Nematics3D.datatypes import (
    Vect3D,
    as_Vect3D,
    as_QField5
)
from Nematics3D.field import (
    generate_coordinate_grid
)

class PlotPlaneGrid():

    @logging_and_warning_decorator
    def __init__(self,
                 normal: Vect3D,
                 num1: int,
                 num2: int,
                 size: float,
                 shape: Literal["circle", "rectangle"] = "rectangle",
                 origin: Vect3D = (0,0,0),
                 axis1: Optional[Vect3D] = None,
                 corners_limit: Optional[np.ndarray] = None,
                 logger=None,
                 ):
        
        self.update_grid(
            normal,
            shape,
            num1,
            num2,
            size,
            origin=origin,
            axis1=axis1,
            corners_limit=corners_limit,
            logger=logger
        )

    def update_grid(self,
                 normal: Vect3D,
                 shape: Literal["circle", "rectangle"],
                 space1: float,
                 space2: float,
                 size: float,
                 origin: Vect3D = (0,0,0),
                 axis1: Optional[Vect3D] = None,
                 corners_limit: Optional[np.ndarray] = None,
                 logger=None,                    
                 ):
        
        space1 = int(space1)
        space2 = int(space2)
        
        num1 = int(size/space1)
        num2 = int(size/space2)
        
        origin = as_Vect3D(origin)

        if shape not in ["circle", "rectangle"]:
            msg = f">>> Input shape must either be 'circle' or 'rectangle'. Got {shape} instead.\n"
            msg += "Use 'rectangle' in the following."
            shape = 'rectangle'
            logger.warning(msg)
        
        normal = as_Vect3D(normal, is_norm=True)

        if axis1 is not None:
            axis1 = as_Vect3D(axis1, is_norm=True)
            if normal @ axis1 != 0:
                msg = 'normal must be perpendicular to axis1.\n'
                msg += 'Discard the component aligned with normal along axis1 in the following.'
                logger.info(msg)
        if axis1 is None:
            axis1 = np.random.randn(3)
            axis1 /= np.linalg.norm(axis1)

        axis1 = axis1 - axis1 @ normal * normal
        axis1 /= np.linalg.norm(axis1)
        axis_both = np.array([axis1, np.cross(normal, axis1)])
            
        source_shape = (size, size)
        target_shape = (num1, num2)

        grid = generate_coordinate_grid(source_shape, target_shape)
        grid = np.reshape(grid, (-1,2))
        grid = np.einsum('ai, ib -> ab', grid, axis_both)

        grid = grid - np.average(grid, axis=0) + origin
        grid = self.select_grid_in_box(grid, corners_limit=corners_limit, logger=logger)
        
        self._grid = grid
        self._axis1 = axis1
        self._normal = normal
        self._origin = origin
        self._shape = shape
        self._num1 = num1
        self._num2 = num2

    @staticmethod
    def select_grid_in_box(grid: np.ndarray, corners_limit: Optional[np.ndarray] = None, logger=None):
        if corners_limit is None:
            return grid
        else:
            box_axis1 = corners_limit[1] - corners_limit[0]
            box_axis2 = corners_limit[2] - corners_limit[0]
            box_axis3 = corners_limit[3] - corners_limit[0]
            
        L1, L2, L3 = np.linalg.norm(box_axis1), np.linalg.norm(box_axis2), np.linalg.norm(box_axis3)
        u1, u2, u3 = box_axis1 / L1, box_axis2 / L2, box_axis3 / L3

        rel = grid - corners_limit[0]
        x = rel @ u1
        y = rel @ u2
        z = rel @ u3
        
        tol = 1e-9
        mask = (
            (x >= -tol) & (x <= L1 + tol) &
            (y >= -tol) & (y <= L2 + tol) &
            (z >= -tol) & (z <= L3 + tol)
        )

        grid = grid[mask]
        if len(grid) == 0:
            msg = "No grid found in this box with corners_limit:\n"
            msg += f"{corners_limit}"
            logger.warning(msg)
            print(grid)
            
        return grid
    

    def add_Q(self, Q_values):
        
        Q_values = as_QField5(Q_values)
        Q_values.reshape(-1, 5)

        if np.shape(Q_values)[0] != np.shape(self._grid)[0]:
            raise ValueError(f"Got {np.shape(Q_values)[0]} points of Q_values and {np.shape(self._grid)[0]} points of grids.")
        
        from Nematics.field import diagonalizeQ
        self._Q = Q_values
        self._S, self._n = diagonalizeQ(Q_values)
        

    def add_values(self, name, values):
        
        name = str(name)

        if np.shape(values)[0] != np.shape(self._grid)[0]:
            raise ValueError(f"Got {np.shape(values)[0]} points of values and {np.shape(self._grid)[0]} points of grids.")

        setattr(self, name, values)