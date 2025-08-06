from mayavi import mlab
import numpy as np

from Nematics3D.datatypes import Vect3D
from .visual_decorator import auto_properties


@auto_properties({
    'color': 'actor.actor.property.color',
    'opacity': 'actor.actor.property.opacity',
    'radius': 'actor.actor.property.tube_radius',
    'sides': 'actor.actor.property.tube_sides',
    'specular': 'actor.actor.property.specular',
    'specular_color': 'actor.actor.property.specular_color',
    'specular_power': 'actor.actor.property.specular_power',
    'x': 'actor.mlab_source.x',
    'y': 'actor.mlab_source.y',
    'z': 'actor.mlab_source.z',
})
class PlotTube:
    
    def __init__(
        self,
        coords: np.ndarray,
        color: Vect3D = (0, 0, 0),
        radius: float = 1,
        opacity: float = 1,
        sides: int = 6,
        specular: float = 1,
        specular_color: Vect3D = (1.0, 1.0, 1.0),
        specular_power: float = 11
    ) -> None:
        
        x, y, z = coords[:,0], coords[:,1], coords[:,2]
        
        self.actor = mlab.plot3d(
            x, y, z,
            color=color,
            tube_radius=radius,
            tube_sides=sides,
            opacity=opacity,
        )  
        
        prop = self.actor.actor.property
        prop.specular = self.specular
        prop.specular_color = self.specular_color
        prop.specular_power = self.specular_power
        
        
        

    
    
        
