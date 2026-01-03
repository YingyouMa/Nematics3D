from dataclasses import dataclass, field
import numpy as np
import pyvista as pv

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import (
    ColorRGB,
    as_ColorRGB,
    as_Number,
    as_str,
    as_bool,
    Vect,
    as_Vect
)
from ..opts import merge_opts_all


@dataclass(slots=True)
class OptsScene:
    azimuth: float = 0.0
    elevation: float = 0.0
    roll: float = 0.0
    distance: float = 10.0
    focal_point: Vect(3) = (0.0, 0.0, 0.0)
    size: Vect(2) = (1920, 1080)
    bg_color: ColorRGB = (1, 1, 1)
    
    _on_change: callable = field(default=None, repr=False, compare=False)
    
    __descriptions__ = {
        "azimuth": "The azimuthal angle (degrees) of the camera around the focal point.",
        "elevation": "The elevation angle (degrees) of the camera relative to the focal plane.",
        "roll": "The rotation (degrees) of the camera about the direction of projection.",
        "distance": "The distance from the camera position to the focal point.",
        "focal_point": "The point the camera is looking at (x, y, z).",
        "size": "The window size of figure",
        "bg_color": "The background color of figure"
    }
    
    _validators = {
        "azimuth": lambda self, v: as_Number(
            v, 
            name=self.__descriptions__["azimuth"], 
            value_range=(0, 360), 
            bounded=True
        ),
        "elevation": lambda self, v: as_Number(
            v, 
            name=self.__descriptions__["elevation"], 
            value_range=(-90, 90), 
            bounded=True
        ),
        "roll": lambda self, v: as_Number(
            v, 
            name=self.__descriptions__["roll"], 
            value_range=(-180,180), 
            bounded=True
        ),
        "distance": lambda self, v: as_Number(
            v, 
            name=self.__descriptions__["distance"], 
            value_range=(0, np.inf)
        ),
        "focal_point": lambda self, v: as_Vect(
            v,
            name=self.__descriptions__["focal_point"],
            dim=3
            ),
        "size": lambda self, v: as_Vect(
            v,
            name=self.__descriptions__["size"],
            dim=2
            ),
        "bg_color": lambda self, v: as_ColorRGB(
            v,
            name=self.__descriptions__["bg_color"],
            )
    }
    
    def __setattr__(self, key, value):
        is_camera = False
        if key in ["azimuth", "elevation", "roll", "distance", "focal_point"]:
            is_camera = True
            old_value = getattr(self, key, None)
        
        if key in self._validators:
            value = self._validators[key](self, value)
        
        object.__setattr__(self, key, value)
        
        if is_camera and old_value is not None and not np.allclose(old_value, value, atol=1e-7):
            if self._on_change:
                self._on_change(key, value)


class PlotFigure:
    
    __descriptions__ = {
        "name": (
            "Human-readable identifier of this PlotFigure."
            "In some contexts (e.g., when registered in a figure manager and used as a key), "
            "the name may become locked and thus read-only. Attempting to modify a locked "
            "name raises AttributeError."
        ),
        "opts": "The OptsScene object controlling the options beyond specific actors (glyphs)",
        "_entities_plotter": "The underlying PyVista Plotter instance that owns the VTK rendering pipeline. ",
        "_entities": "A registry (dict) for objects attached to this figure.",
        "_state_is_name_locked": (
            "Boolean flag indicating whether the figure name is currently locked. "
            "When True, the 'name' property becomes read-only to avoid breaking external "
            "bookkeeping (e.g., a figure manager that uses the name as a dictionary key)."
        ),
    }
    
    __slots__ = tuple(__descriptions__.keys()) # + ("__weakref__",)
    
    def __init__(self,
                 plotter: pv.Plotter | None = None,
                 name: str = 'figure'):
        
        if plotter is None:
            object.__setattr__(self, '_entities_plotter', pv.Plotter())
        else:
            object.__setattr__(self, '_entities_plotter', plotter)
        
        name = as_str(name, name="The name of PlotFigure", replace='figure')
        object.__setattr__(self, 'name', name)
        
        object.__setattr__(self, '_entities', {})
        object.__setattr__(self, '_state_is_name_locked', False)
        
    @property
    def opts_name(self):
        return self.name
    
    @opts_name.setter
    def opts_name(self, value):
        self.name = value
        
    @property
    def pl(self):
        return self._entities_plotter
    
    def __setattr__(self, key, value):
        
        if key == 'name' and self._state_is_name_locked:
            raise AttributeError(
                f"Name of PlotFigure {self.name!r} could not be modified"
                " because it is used as the key in figure manager")
        
        object.__setattr__(self, key, value)
    
    def _helper_register_entity(self, entity_instance, entity_category, is_reset_camera):
        if entity_category in self._entities.keys():
            self._entities[entity_category].append(entity_instance)
        else:
            self._entities[entity_category] = [entity_instance]
        if is_reset_camera:
            self._obj_camera._helper_sync_from_cam()
           
    def act_get_entities_names(self):
        names = [
            entity.opts.name
            for entity_list in self._entities.values()
            for entity in entity_list
            ]
        return names
    
    def act_check_is_alive(self):
        try:
            plotter = self._entities_plotter
            if plotter._closed:
                return False
    
            iren = plotter.iren
            return iren is not None and iren.initialized
        except Exception:
            return False
        
    def __bool__(self):
        return self.act_check_is_alive()