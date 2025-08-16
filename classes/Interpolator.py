import numpy as np

from Nematics3D.datatypes import Vect3D, as_Vect3D
from Nematics3D.field import apply_linear_transform

class Interpolator:
    
    def __init__(self, 
                 interpolator, 
                 transform: np.ndarray = np.eye(3), 
                 offset: Vect3D = np.array([0, 0, 0])
                 ):
        
        offset = as_Vect3D(offset)
        
        self._interpolator = interpolator
        self._transform = transform
        self._offset = offset
        
    def interpolate(self, points: np.ndarray, is_index=False):
        
        if not is_index:
            points = apply_linear_transform(
                points, 
                transform=np.linalg.inv(self._transform),
                offset=-self._origin
            )
        return self._interpolator(points)
        