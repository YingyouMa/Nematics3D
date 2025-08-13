import numpy as np
from typing import Tuple, Optional, List, Literal

from ..logging_decorator import logging_and_warning_decorator, Logger
from ..datatypes import (
    Vect3D,
    as_Vect3D,
)
from ..field import (
    generate_coordinate_grid
)

class PlotPlaneGrid():

    @logging_and_warning_decorator
    def __init__(self,
                 normal: Vect3D,
                 shape: Literal["circle", "rectangle"],
                 num1: int,
                 num2: int,
                 size: float,
                 origin: Vect3D = (0,0,0),
                 axis1: Optional[Vect3D] = None,
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
            logger=logger
        )



    def update_grid(self,
                 normal: Vect3D,
                 shape: Literal["circle", "rectangle"],
                 num1: int,
                 num2: int,
                 size: float,
                 origin: Vect3D = (0,0,0),
                 axis1: Optional[Vect3D] = None,
                 logger=None,                    
                 ):
        
        num1 = int(num1)
        num2 = int(num2)
        
        origin = as_Vect3D(origin)

        if shape not in ["circle", "rectangle"]:
            msg = f">>> Input shape must either be 'circle' or 'rectangle'. Got {shape} instead.\n"
            msg += "Use 'rectangle' in the following."
            shape = 'rectangle'
            logger.warning(msg)
        
        normal = as_Vect3D(normal, is_norm=True)

        if axis1 is not None:
            if normal @ axis1 != 0:
                msg = 'normal must be perpendicular to axis1.\n'
                msg += 'Discard the component aligned with normal along axis1 in the following.'
                logger.info(msg)
        if axis1 is None:
            axis1 = np.random.randn(3)

        axis1 = axis1 - axis1 @ normal * normal
        axis1 /= np.linalg.norm(axis1)
        axis_both = np.array([normal, axis1])
            
        source_shape = (size, size)
        target_shape = (num1, num2)

        gird = generate_coordinate_grid(source_shape, target_shape)
        grid = np.reshape(-1,2)
        grid = np.einsum('ai, bi -> ab', gird, axis_both)

        grid = gird - np.average(gird, axis=0) + origin

        self._grid = grid
        self._axis1 = axis1
        self._normal = normal
        self._origin = origin
        self._shape = shape
        self._num1 = num1
        self._num2 = num2
