import numpy as np
from typing import Tuple, Optional, List, Literal

from ..logging_decorator import logging_and_warning_decorator, Logger
from ..datatypes import (
    Vect3D,
)
from ..field import (
    generate_coordinate_grid
)

class PlotPlaneGrid():

    def __init__(self,
                 normal,
                 shape: Literal["circle", "rectangle"],
                 num1: int,
                 num2: int,
                 size: float,
                 origin: Vect3D = (0,0,0),
                 axis1: Optional[Vect3D] = None,
                 ):

