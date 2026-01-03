from dataclasses import dataclass
import numpy as np
import pyvista as pv

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import (
    ColorRGB,
    as_ColorRGB,
    as_ColorRGB_array,
    as_Number,
    as_str,
    as_bool,
    Vect,
    as_Vect
)
from ..opts import merge_opts_all
from .plot_figure_camera import FigureCamera


class PlotFigure:
    
    __descriptions__ = {
        # === Public / User-facing ===
        "name": (
            "Human-readable identifier of this PlotFigure."
            "In some contexts (e.g., when registered in a figure manager and used as a key), "
            "the name may become locked and thus read-only. Attempting to modify a locked "
            "name raises AttributeError."
        ),

        # === Internal State / Core Objects ===
        "_obj_plotter": "The underlying PyVista Plotter instance that owns the VTK rendering pipeline. ",
        
        "_obj_camera": (
            "Camera controller associated with '_obj_plotter'. "
            "This object is typically a thin wrapper (e.g., FigureCamera) that provides "
            "higher-level camera operations and state synchronization."
        ),

        # === Entity registry ===
        "_entities": (
            "A registry (dict) for objects attached to this figure."
        ),

        # === State flags (implied) ===
        "_state_is_name_locked": (
            "Boolean flag indicating whether the figure name is currently locked. "
            "When True, the 'name' property becomes read-only to avoid breaking external "
            "bookkeeping (e.g., a figure manager that uses the name as a dictionary key)."
        ),
    }
    
    __slots__ = tuple(__descriptions__.keys()) + ("__weakref__",)
    
    def __init__(self,
                 plotter: pv.Plotter | None = None,
                 name: str = 'figure'):
        
        if plotter is None:
            object.__setattr__(self, '_obj_plotter', pv.Plotter())
        else:
            object.__setattr__(self, '_obj_plotter', plotter)
        object.__setattr__(self, '_obj_camera', FigureCamera(self._obj_plotter))
        
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
            plotter = self._obj_plotter
            if plotter._closed:
                return False
    
            iren = plotter.iren
            return iren is not None and iren.initialized
        except Exception:
            return False
        
    def __bool__(self):
        return self.act_check_is_alive()