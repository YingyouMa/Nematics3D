import pyvista as pv
import numpy as np
import weakref
from typing import Mapping, Any

from .Interpolator import Interpolator
from Nematics3D.logging_decorator import logging_and_warning_decorator
from .plane_grid import PlaneGrid, OptsPlaneGrid
from .plane_grid_polar import PlaneGridPolar, OptsPlaneGridPolar
from .class_base import ClassBase

#!!! class name

class InterpolatePlane(ClassBase):

    __descriptions__ = {
        **(ClassBase.__descriptions__),
        
        "raw_name": "The name identifier of this plane object",
        "_calc_result": "The interpolated value of the physics quantity on the 2D plane grid.",
        "_raw_interpolator": "Interpolator object for the physics quantity (class Interpolator)",
        "_entity_plane": "The PlaneGrid entity (coordinates of 2D lattice)",
    }

    __slots__ = tuple(
            k for k, v in __descriptions__.items() 
            if not v.startswith("Property:") and k not in ClassBase.__slots__
        )

    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        interpolator: Interpolator,
        name: str = "interpolate plane",
        grid: PlaneGrid | PlaneGridPolar | None = None,
        opts: OptsPlaneGrid | OptsPlaneGridPolar | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        logger=None,
        **kwargs,
    ):
        
        super().__init__(name=name, name_replace="interpolate plane")
        
        if grid:
            object.__setattr__(self, "_entity_plane", grid)
            self._entity_plane.act_commit(
                opts=opts, 
                name=self.name +"-grid",
                **kwargs
                )
        else:
            grid = PlaneGrid(
                opts=opts,
                opts_defaults_override=opts_defaults_override,
                name=self.name +"-grid",
                **kwargs)
            object.__setattr__(self, "_entity_plane", grid)
                
        object.__setattr__(
            self._entity_plane, "_impl_field_ref", weakref.ref(self)
        )

        if not isinstance(interpolator, Interpolator):
            raise TypeError(
                "Interpolator for InterplatePlane must be the class of Nematics3D.classes.Interpolator.Interpolator"
            )
        object.__setattr__(self, '_raw_interpolator', interpolator)
        
        self._helper_commit()
        

    @logging_and_warning_decorator()
    def _helper_commit(self, logger=None):

        plane_grid = self._entity_plane

        logger.detail("Retrieving the full grid in lattice index structure ...")
        grid_all = plane_grid._entity_grid_all
        grid_all_flatten = np.reshape(grid_all, (-1, 3))

        logger.detail("Interpolating ...")
        result = self._raw_interpolator.interpolate(grid_all_flatten)
        object.__setattr__(self, "_calc_result", result[plane_grid._calc_box_mask])


    @property
    def result(self):
        return self._calc_result

    @property
    def plane(self):
        return self._entity_plane
    
    def __setattr__(self, key, value):
        super()._helper_setattr_basic(key, value)
    
