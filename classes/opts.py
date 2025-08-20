from dataclasses import dataclass, field
from typing import Tuple, Optional, Union, Literal, Callable
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from Nematics3D.datatypes import ColorRGB,as_ColorRGB, Vect, as_Vect, Tensor, as_Tensor, Number, as_Number, nField
from Nematics3D.field import n_color_immerse


@dataclass
class OptsSmoothen:
    window_ratio: Number = 3                    
    window_length: Optional[Number] = None       
    order: Number = 3                                
    N_out_ratio: Number = 3.0                      
    mode: Literal["interp", "wrap"] = "interp"
    min_line_length: int = 61
    name: str = 'None'

    def __post_init__(self):
        
        if not isinstance(self.min_line_length, int):
            raise TypeError("The minimum line length to be smoothened must be integer")

        self.window_ratio = as_Number(self.window_ratio, name="window_ratio of smoothening")
        if self.window_length is not None:
            self.window_length = as_Number(self.window_length, name="window_ratio of smoothening")
        self.order = as_Number(self.order, name="smoothing order")
        self.N_out_ratio = as_Number(self.N_out_ratio, name="ratio between # points of output and input in smoothening")

        if self.mode not in ("interp", "wrap"):
            raise ValueError("smoothing mode must be interp or wrap")


# --- Scene Options ---
@dataclass
class OptsScene:
    is_new: bool = True
    fig_size: Tuple[float, float] = (1920, 1360)
    bgcolor: ColorRGB = (1.0, 1.0, 1.0)
    fgcolor: ColorRGB = (0.0, 0.0, 0.0)
    name: str = "None"
    azimuth: Number = 45
    elevation: Number = 54.735610317245346
    roll: Number = 0
    distance: Optional[Number] = None
    focal_point: Optional[Vect(3)] = None
    name: Optional[str] = 'None'

    def __post_init__(self):
        self.fig_size = as_Vect(self.fig_size, dim=2)
        self.fig_size = tuple(self.fig_size)
        self.bgcolor = as_ColorRGB(self.bgcolor)
        self.bgcolor = as_ColorRGB(self.bgcolor)
        
        if not isinstance(self.name, str):
            raise TypeError(f"The name of scene must be str. Got {self.name} instead.")
            
        self.azimuth = as_Number(self.azimuth, name="Azimuth of position of scene camera")
        self.elevation = as_Number(self.elevation, name="Elevation of position of scene camera")
        self.roll = as_Number(self.roll, name="Roll of position of scene camera")
        
        if self.distance is not None:
            self.distance = as_Number(self.distance, name="Distance of scene camera from focal point")
            
        if self.focal_point is not None:
            self.focal_point = as_Number(self.focal_point, name="The focal point of scene camera")

        if not isinstance(self.name, str):
            raise TypeError("The name of the tube must be str")
        
        
# --- Plane Options ---
@dataclass
class OptsPlane:
    normal: Vect(3)
    spacing: Number
    size: Number
    shape: Literal["circle", "rectangle"] = "rectangle"
    origin: Vect(3) = (0, 0, 0)
    axis1: Optional[Vect(3)] = None
    corners_limit: Optional[np.ndarray] = None
    colors: Union[Callable[nField, ColorRGB], ColorRGB] = n_color_immerse
    opacity: Union[Callable[nField, np.ndarray], float] = 1
    length: Number = 3.5
    radius: Number = 0.5
    is_n_defect: bool = True
    defect_opacity: Union[Callable[nField, np.ndarray], float] = 1
    grid_offset: Vect(3) = field(default_factory=lambda: np.array([0, 0, 0]))
    grid_transform: Tensor((3,3)) = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        
        self.normal = as_Vect(self.normal, name='normal')
        self.spacing = as_Number(self.space, name='spacing')
        self.size = as_Number(self.size, name='size')
        
        if self.shape not in ("circle", "rectangle"):
            raise ValueError("Shape of a plane must be 'circle' or 'rectangle'")
        
        self.origin = as_Vect(self.origin, name='origin')
        if self.axis1 is not None:
            self.axis1 = as_Vect(self.axis1, name='axis1')
            
        self.corners_limit = as_Tensor(self.corners_limit, (8,3), name='box corners')
        
        if not isinstance(self.is_n_defect, bool):
            raise TypeError("is_n_defect must be a boolean value.")
            
        self.grid_offset = as_Vect(self.radius, name='grid_offset')
        self.grid_transform = as_Tensor(self.grid_transform, (3,3), name="grid_transform")


# --- nPlane Options ---
@dataclass
class OptsnPlane:
    QInterpolator: RegularGridInterpolator
    colors: Union[Callable[nField, ColorRGB], ColorRGB] = n_color_immerse
    opacity: Union[Callable[nField, np.ndarray], float] = 1
    length: Number = 3.5
    radius: Number = 0.5
    is_n_defect: bool = True
    defect_opacity: Union[Callable[nField, np.ndarray], float] = 1

    def __post_init__(self):
    
        self.length = as_Number(self.length, name='length')
        self.radius = as_Number(self.radius, name='radius')
        
        if not isinstance(self.is_n_defect, bool):
            raise TypeError("is_n_defect must be a boolean value.")


# --- Extent Options ---
@dataclass
class OptsExtent:
    corners: Optional[np.ndarray] = None
    radius: Number = 1.0
    sides: Number = 6
    opacity: Number = 1.0
    color: ColorRGB = (0,0,0)

    def __post_init__(self):
        if self.corners is not None:
            self.corners = as_Tensor(self.corners, (8,3), name="The array \'corners\' storing the positions of the 8 points.")
        self.radius = as_Number(self.radius, name='radius of extent tubes')
        self.sides = as_Number(self.sides, name='sides number of extent tubes')
        self.opacity = as_Number(self.opacity, name='opacity of extent tubes')
        self.color = as_ColorRGB(self.color)
        
# --- Tube Options ---
@dataclass
class OptsTube:
    radius: Number = 0.5
    opacity: Number = 1
    color: ColorRGB = (1.0, 1.0, 1.0)
    sides: Number = 6
    specular: Number = 1
    specular_color: ColorRGB = (1.0, 1.0, 1.0)
    specular_power: Number = 11
    name: str = 'None'
    
    def __post__init__(self):
            
        self.radius = as_Number(self.radius, name='radius of tube')
        self.opacity = as_Number(self.opacity, name='opacity of tube')
        self.color = as_ColorRGB(self.color)
        self.sides = as_Number(self.sides, name='number of sides of tube')
        self.specular = as_Number(self.specular, name='Strength of the tube specular highlight')
        self.specular_color = as_ColorRGB(self.specular_color)
        self.specular_power = as_Number(self.specular_power, name='Shininess of the tube specular highlight')
        
        if not isinstance(self.name, str):
            raise TypeError("The name of the tube must be str")
        
        
        
        