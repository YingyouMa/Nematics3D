import time
import numpy as np
import pyvistaqt as pyqt
import logging
# import matplotlib.pyplot as plt

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import Nematics3D

from Nematics3D.debug.debug_store import DEBUG_VARS

# index_max =  128
# n = np.load( 'data/n_example_global.npy')[0:index_max, 0:index_max, 0:index_max]
# S = np.load( 'data/S_example_global.npy')[0:index_max, 0:index_max, 0:index_max]

# Q = Nematics3D.QFieldObject(S=S, n=n, box_periodic_flag=index_max >= 128, name="testQ")
# Q.act_lines_smooth()
# # Q.act_visualize_disclination_lines(is_wrap=False, extent_color=(0.5,0.5,0.5))
# Q.act_visualize_disclination_lines(is_new=True, is_wrap=True, line_shading_type='pbr', figure_azimuth=0, figure_elevation=45)




# index_max =  60
# n = np.load( 'data/n_example_global.npy')[0:index_max, 0:index_max, 0:index_max]
# S = np.load( 'data/S_example_global.npy')[0:index_max, 0:index_max, 0:index_max]

# Q = Nematics3D.QFieldObject(S=S, n=n, box_periodic_flag=index_max >= 128)
# Q.act_lines_smooth(window_length=21, min_line_length=40)
# Q.act_visualize_disclination_lines(is_wrap=True, line_color=(0.5, 0.5, 0.5), extent_radius=0.05, min_line_length=50, line_radius=0.4)

# trans = 7.5
# Q.act_visualize_n_plane(
#     is_extent=False,
#     grid_spacing=3,
#     grid_origin=(index_max/2-trans,index_max/2-trans,index_max/2-trans),
#     grid_normal=(1,1,1),
#     figure_azimuth=90,
#     figure_elevation=0
# )

   
figure = Nematics3D.PlotFigure(is_off_screen=False)  
# figure = pyqt.BackgroundPlotter()
   
index_max =  60
n = np.load( 'data/n_example_global.npy')[0:index_max, 0:index_max, 0:index_max]
S = np.load( 'data/S_example_global.npy')[0:index_max, 0:index_max, 0:index_max]

Q = Nematics3D.QFieldObject(S=S, n=n, box_periodic_flag=index_max >= 128)
Q.act_lines_smooth(window_length=21, min_line_length=40)
Q.act_visualize_disclination_lines(is_wrap=True, line_color=(0.5, 0.5, 0.5), extent_radius=0.05, min_line_length=50, line_radius=0.4, figure=figure)
    
trans = 7.5
spacing = 2.5

# Q.act_visualize_n_plane(grid_normal=(1,1,1), grid_spacing=spacing, grid_size=100, grid_origin=(index_max/2-trans,index_max/2-trans,index_max/2-trans), grid_corners_limit=Q._calc_corners, is_extent=False)
# Q.figs.active_fig.act_view_yz()
# Q.figs.active_fig.opts.azimuth = 90

Q.act_visualize_n_near_defect(75, Q.lines[3].smooth)


# testGrid = Nematics3D.PlaneGrid(normal=(1,1,1), spacing=spacing, size=100, origin=(index_max/2-trans,index_max/2-trans,index_max/2-trans), corners_limit=Q._calc_corners)
# testnPlane = Nematics3D.QPlane(Q._calc_interpolator, grid=testGrid)

# testnPlane.act_visualize_n(figure=figure)
# # testnPlane.act_visualize_S(figure=figure)
# figure.act_view_yz()
# figure.opts.azimuth = 90

# figure.act_savefig('test.png')
# figure.pl.screenshot('test.png', transparent_background=False)


# # origin = (41.36, 40.36, 29.75)
# origin = (21.58, 23.31, 23.08)
# # origin = (42.12, 46.13, 23.22)
# figure2 = Nematics3D.PlotFigure()  
# Q.act_visualize_disclination_lines(is_wrap=True, line_color=(0.5, 0.5, 0.5), extent_radius=0.05, min_line_length=50, line_radius=0.4, figure=figure2)
# testPolarGrid = Nematics3D.PlaneGridPolar(normal=(1,1,1), dr=2.5, R_max=36, origin=origin, corners_limit=Q._calc_corners)
# testnPolar = Nematics3D.QPlanePolar(Q._calc_interpolator, grid=testPolarGrid)
# testnPolar.act_visualize_n(figure=figure2)
# # testnPlane.act_visualize_S(figure=figure)
# figure2.act_view_yz()
# figure2.opts.azimuth = 90






