import numpy as np
from typing import Optional, Literal

from .plot_plane_grid import PlotPlaneGrid
from Nematics3D.datatypes import Vect3D
from Nematics3D.field import diagonalizeQ

from tvtk.api import tvtk
from mayavi import mlab

class PlotnPlane():
    
    def __init__(self,
                 normal: Vect3D,
                 num: int,
                 size: float,
                 QIntegrator,
                 shape: Literal["circle", "rectangle"] = "rectangle",
                 origin: Vect3D = (0,0,0),
                 axis1: Optional[Vect3D] = None,
                 corners_limit: Optional[np.ndarray] = None,
                 colors: Optional[np.ndarray] = None,
                 opacity: Optional[np.ndarray] = None,
                 length: float = 3.5,
                 logger=None,
                 ):
        
        self._plane = PlotPlaneGrid(
                            normal,
                            num,
                            num,
                            size,
                            shape=shape,
                            origin=origin,
                            axis1=axis1,
                            corners_limit=corners_limit,
                            logger=logger
                            )
        
        self._Q = QIntegrator(self._plane._grid)
        self._S, self._n = diagonalizeQ(self._Q)
        
        grid = self._plane._grid
        num_points = np.shape(grid)[0]
        
        if colors is None:
            colors = np.ones((num_points, 3))
        if opacity is None:
            opacity = np.ones((num_points,1))
        
        coord0 = grid[:,0] - self._n[:,0]/2
        coord1 = grid[:,1] - self._n[:,1]/2
        coord2 = grid[:,2] - self._n[:,2]/2
        
        self.quiver_with_direct_colors(
            coord0,
            coord1,
            coord2,
            self._n[:,0],
            self._n[:,1],
            self._n[:,2],
            colors,
            opacity,
            length
            )
        
    
    @staticmethod
    def quiver_with_direct_colors(x, y, z, u, v, w, colors, opacity, length=3.5, mode='cylinder'):
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
        g = mlab.pipeline.glyph(src, mode=mode, scale_factor=length)
        g.glyph.scale_mode = 'data_scaling_off'  # 固定缩放，不随标量变化
    
        # 直接颜色模式
        g.actor.mapper.scalar_visibility = True
        g.actor.mapper.color_mode = 'default'
    
        return g