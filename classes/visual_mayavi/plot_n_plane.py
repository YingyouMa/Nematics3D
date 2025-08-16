import numpy as np
from typing import Optional, Literal, Callable, List, Union

from .plot_plane_grid import PlotPlaneGrid
from Nematics3D.datatypes import Vect3D, nField, ColorRGB, as_ColorRGB
from Nematics3D.field import Q_diagonalize, n_color_immerse, n_visualize
from Nematics3D.logging_decorator import logging_and_warning_decorator

class PlotnPlane():
    
    @logging_and_warning_decorator
    def __init__(self,
                 normal: Vect3D,
                 space: float,
                 size: float,
                 QInterpolator,
                 transform: np.ndarray = np.eye(3),
                 offset: Vect3D = np.array([0, 0, 0]),
                 shape: Literal["circle", "rectangle"] = "rectangle",
                 origin: Vect3D = (0,0,0),
                 axis1: Optional[Vect3D] = None,
                 corners_limit: Optional[np.ndarray] = None,
                 colors: Union[Callable[nField,ColorRGB], ColorRGB] = n_color_immerse,
                 opacity: Union[Callable[nField, np.ndarray], float] = 1,
                 length: float = 3.5,
                 radius: float = 0.5,
                 is_n_defect: bool = True,
                 n_defect_opacity: float = 1,
                 logger=None,
                 ):
        
        
        self.make_figure(
                normal,
                space,
                size,
                QInterpolator,
                transform,
                offset,
                shape,
                origin,
                axis1,
                corners_limit,
                colors,
                opacity,
                length,
                radius,
                is_n_defect,
                n_defect_opacity,
                logger=logger)
        
        
    @logging_and_warning_decorator
    def make_figure(
            self,
            normal,
            space,
            size,
            QInterpolator,
            transform,
            offset,
            shape,
            origin,
            axis1,
            corners_limit,
            colors,
            opacity,
            length,
            radius,
            is_n_defect,
            n_defect_opacity,
            logger=None):
        
        self.plane = PlotPlaneGrid(
                            normal,
                            space,
                            space,
                            size,
                            shape=shape,
                            origin=origin,
                            axis1=axis1,
                            corners_limit=corners_limit,
                            logger=logger
                            )
        
        self.Q = QInterpolator(self.plane._grid)
        self.S, self.n = Q_diagonalize(self.Q)
        
        # if is_n_defect:
        #     shape_all = np.shape(self.plane._grid_all)
        #     grid_all_flatten = np.reshape(self.plane._grid_all, (-1,2))
        #     Q_all = QInterpolator(grid_all_flatten)
        #     _, n_all = Q_diagonalize(Q_all)
        #     n_all = np.reshape(n_all, (1, *shape_all, 3))
        #     defect_indices
        
        grid = self.plane._grid
        self.num_points = np.shape(grid)[0]
        
        self.colors_func = self.colors_check(colors)
        self.opacity_func = self.opacity_check(opacity)
             
        colors_out = self.colors_func(self.n)
        opacity_out = self.opacity_func(self.n)
        
        if hasattr(self, 'items'):
            self.items[0].remove()
            
        self.items = [n_visualize(
            grid[:,0],
            grid[:,1],
            grid[:,2],
            self.n[:,0],
            self.n[:,1],
            self.n[:,2],
            colors_out,
            opacity_out,
            length=length,
            radius=radius
            )]
        
        self.radius = radius
        self.axis1 = axis1
        self.normal = normal
        self.origin = origin
        self.shape = shape
        self.space = space
        self.size = size
        self.corners_limit = corners_limit
        self._QInterpolator = QInterpolator
            
    @logging_and_warning_decorator
    def colors_check(self, data, logger=None): 
        if isinstance(data, (tuple, list, np.ndarray)):
            data = as_ColorRGB(data)
            colors = lambda n: np.broadcast_to(data, (len(n), 3))
        elif not callable(data):
            msg = "Colors must be either callable function or a tuple of three elements.\n"
            msg = "Use default colormap in the following."
            logger.warning(msg)
            colors = n_color_immerse
        else:
            colors = data
        return colors
    
    @logging_and_warning_decorator
    def opacity_check(self, data, logger=None): 
        if isinstance(data, (int, float)):
            opacity = lambda n: np.broadcast_to(data, (len(n), 1))
        elif not callable(input):
            msg = "Opacity must be either callable function or a float.\n"
            msg = "Use 1 in the following."
            logger.warning(msg)
            opacity = lambda n: np.broadcast_to(1, (len(n), 1))
        else:
            opacity = data
        return opacity
    
 
    @property
    def length(self):
        return self.items[0].glyph.glyph_source.glyph_source.height
    
    @length.setter
    def length(self, value: float):
        self.items[0].glyph.glyph_source.glyph_source.height = float(value)
        
    @property
    def radius(self):
        return self.items[0].glyph.glyph_source.glyph_source.radius
    
    @radius.setter
    def radius(self, value: float):
        self.items[0].glyph.glyph_source.glyph_source.radius = float(value)
        
    @property
    def opacity(self):
        rgba = self.items[0].parent.parent.data.point_data.scalars
        return np.array(rgba)[:,3] / 255
    
    @opacity.setter
    def opacity(self, data):
        self.opacity_func = self.opacity_check(data)
        rgba = self.items[0].parent.parent.data.point_data.scalars
        opacity_out = self.opacity_func(self.n) * 255
        opacity_out = opacity_out[:,0]
        rgba = np.array(rgba)
        rgba[:,3] = opacity_out
        for i in range(self.num_points):
            self.items[0].parent.parent.data.point_data.scalars[i] = rgba[i]
        self.items[0].parent.parent.data.point_data.scalars.modified()
        
    @property
    def colors(self):
        rgba = self.items[0].parent.parent.data.point_data.scalars
        return np.array(rgba)[:,:3] / 255
    
    @colors.setter
    def colors(self, data):
        self.colors_func = self.colors_check(data)
        rgba = self.items[0].parent.parent.data.point_data.scalars
        colors_out = self.colors_func(self.n) * 255
        rgba = np.array(rgba)
        rgba[:,:3] = colors_out
        for i in range(self.num_points):
            self.items[0].parent.parent.data.point_data.scalars[i] = rgba[i]
        self.items[0].parent.parent.data.point_data.scalars.modified()
        
    @logging_and_warning_decorator    
    def update(self, logger=None, **changes):
        
        if not changes:
            return
        
        for k, v in changes.items():
            setattr(self, k, v)
        
        keys_rebuild = ['axis1', 'normal', 'origin', 'shape', 'space', 'size']
        
        for k in keys_rebuild:
            if k in changes:
                self.make_figure(
                        self.normal,
                        self.space,
                        self.size,
                        self._QInterpolator,
                        self.shape,
                        self.origin,
                        self.axis1,
                        self.corners_limit,
                        self.colors_func,
                        self.opacity_func,
                        self.length,
                        self.radius,
                        logger=logger)
                return
        
