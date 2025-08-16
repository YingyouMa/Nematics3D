import numpy as np
from typing import Optional, Literal, Callable, List, Union

from .plot_plane_grid import PlotPlaneGrid
from Nematics3D.datatypes import Vect3D, nField, ColorRGB, as_ColorRGB
from Nematics3D.field import diagonalizeQ, n_color_immerse
from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.general import calc_colors, calc_opacity

from tvtk.api import tvtk
from mayavi import mlab

class PlotnPlane():
    
    @logging_and_warning_decorator
    def __init__(self,
                 normal: Vect3D,
                 space: float,
                 size: float,
                 QInterpolator,
                 shape: Literal["circle", "rectangle"] = "rectangle",
                 origin: Vect3D = (0,0,0),
                 axis1: Optional[Vect3D] = None,
                 corners_limit: Optional[np.ndarray] = None,
                 colors: Union[Callable[nField,ColorRGB], ColorRGB] = n_color_immerse,
                 opacity: Union[Callable[nField, np.ndarray], float] = 1,
                 length: float = 3.5,
                 radius: float = 0.5,
                 logger=None,
                 ):
        
        
        self.make_figure(
                normal,
                space,
                size,
                QInterpolator,
                shape,
                origin,
                axis1,
                corners_limit,
                colors,
                opacity,
                length,
                radius,
                logger=logger)
        
        
    @logging_and_warning_decorator
    def make_figure(
            self,
            normal,
            space,
            size,
            QInterpolator,
            shape,
            origin,
            axis1,
            corners_limit,
            colors,
            opacity,
            length,
            radius,
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
        self.S, self.n = diagonalizeQ(self.Q)
        
        grid = self.plane._grid
        self.num_points = np.shape(grid)[0]
        
        self.colors_func = self.colors_check(colors)
        self.opacity_func = self.opacity_check(opacity)
             
        colors_out = self.colors_func(self.n)
        opacity_out = self.opacity_func(self.n)
        
        if hasattr(self, 'items'):
            self.items[0].remove()
        
        self.items = [self.quiver_with_direct_colors(
            grid[:,0],
            grid[:,1],
            grid[:,2],
            self.n[:,0],
            self.n[:,1],
            self.n[:,2],
            colors_out,
            opacity_out,
            length=length
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
    
    @staticmethod
    def quiver_with_direct_colors(x, y, z, u, v, w, colors, opacity, length=3.5, radius=0.5, mode='cylinder'):
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        z = np.asarray(z).ravel()
        u = np.asarray(u).ravel()
        v = np.asarray(v).ravel()
        w = np.asarray(w).ravel()
    
        # 点和向量
        pts = np.c_[x, y, z]
        vec = np.c_[u, v, w]
    
        # RGBA 转换
        colors = np.asarray(colors)
        colors = np.hstack([colors, opacity]) * 255
        colors = colors.astype(np.uint8)
    
        # PolyData
        poly = tvtk.PolyData(points=pts)
        poly.point_data.vectors = vec
        poly.point_data.vectors.name = 'vectors'
        poly.point_data.scalars = colors
        poly.point_data.scalars.name = 'rgba'
    
        # 管线
        src = mlab.pipeline.add_dataset(poly)
        g = mlab.pipeline.glyph(src, mode=mode, scale_factor=1)
        g.glyph.scale_mode = 'data_scaling_off'  # 固定缩放，不随标量变化
    
        # 直接颜色模式
        g.actor.mapper.scalar_visibility = True
        g.actor.mapper.color_mode = 'direct_scalars'
        
        g.glyph.glyph_source.glyph_source.height = length
        g.glyph.glyph_source.glyph_source.radius = radius
    
        return g
    
 
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
        
        keys_rebuild = ['axis1', 'normal', 'origin', 'shape', 'space']
        
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
        
