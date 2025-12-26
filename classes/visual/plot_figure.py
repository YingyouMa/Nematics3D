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
    
    def __init__(self):
        
        self.obj_plotter = pv.Plotter()
        self.obj_camera = FigureCamera(self.obj_plotter)
        self._entities = {}
    
    def _helper_register_entity(self, entity_instance, entity_category, is_reset_camera):
        if entity_category in self._entities.keys():
            self._entities[entity_category].append(entity_instance)
        else:
            self._entities[entity_category] = [entity_instance]
        if is_reset_camera:
            self.obj_camera._helper_sync_from_cam()
           
    def act_get_entities_names(self):
        names = [
            entity.opts.name
            for entity_list in self._entities.values()
            for entity in entity_list
            ]
        return names