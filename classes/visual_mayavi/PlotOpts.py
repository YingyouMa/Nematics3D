from dataclasses import dataclass
from typing import Tuple, Optional, Union, Literal, Callable, Sequence
import numpy as np

from Nematics3D.datatypes import ColorRGB, Vect3D, nField

# --- Scene Options ---
@dataclass
class SceneOpts:
    fig_size: Tuple[float, int] = (1920, 1360)
    bgcolor: Vect3D = (1.0, 1.0, 1.0)
    fgcolor: Vect3D = (0.0, 0.0, 0.0)

    def __post_init__(self):
        
        if not (isinstance(self.fig_size, Sequence) and len(self.fig_size) == 2):
            raise TypeError("fig_size must be a tuple of 2 ints")
            
        if not all(isinstance(x, (int, np.integer)) for x in self.fig_size):
            raise TypeError("fig_size values must be integers")
        for name, val in [("bgcolor", self.bgcolor), ("fgcolor", self.fgcolor)]:
            if not (isinstance(val, tuple) and len(val) == 3):
                raise TypeError(f"{name} must be a 3-tuple")
            if not all(isinstance(c, (float, int)) for c in val):
                raise TypeError(f"{name} values must be float or int")
            if not all(0.0 <= c <= 1.0 for c in val):
                raise ValueError(f"{name} values must be in [0,1]")

# --- Plane Options ---
@dataclass
class PlaneOpts:
    shape: Literal["circle", "rectangle"] = "rectangle"
    origin: Vect3D = (0,0,0)
    axis1: Optional[Vect3D] = None
    colors: Union[Callable[[nField], ColorRGB], ColorRGB] = (0.5, 0.5, 0.5)
    opacity: Union[Callable[[nField], np.ndarray], float] = 1.0
    length: float = 3.5
    radius: float = 0.5
    is_n_defect: bool = True
    defect_opacity: float = 1.0

    def __post_init__(self):
        if self.shape not in ("circle", "rectangle"):
            raise ValueError("shape must be 'circle' or 'rectangle'")
        if not isinstance(self.length, (float, int)):
            raise TypeError("length must be float")
        if not isinstance(self.radius, (float, int)):
            raise TypeError("radius must be float")
        if not (0 <= self.defect_opacity <= 1):
            raise ValueError("defect_opacity must be in [0,1]")

# --- Extent Options ---
@dataclass
class ExtentOpts:
    enabled: bool = True
    radius: float = 1.0
    opacity: float = 1.0

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool")
        if not isinstance(self.radius, (float, int)):
            raise TypeError("radius must be float")
        if not (0 <= self.opacity <= 1):
            raise ValueError("opacity must be in [0,1]")
