import time
import numpy as np
from mayavi import mlab
import re
import glob
import logging
# import matplotlib.pyplot as plt

import sys
# sys.path.insert(0, 'D:/Document/GitHub/3D-active-nematics/simulation')
sys.path.insert(0, 'D:/Document/GitHub/')
import Nematics3D


L = 200

#Q = np.load( 'data/Q_example_local.npy')

# S, n = Nematics3D.diagonalizeQ(Q)

index_max =  80
n = np.load( 'data/n_example_global.npy')[0:index_max, 0:index_max, 0:index_max]
S = np.load( 'data/S_example_global.npy')[0:index_max, 0:index_max, 0:index_max]

Q = Nematics3D.QFieldObject(S=S, n=n, box_periodic_flag=index_max >= 128)

# @Nematics3D.logging_and_warning_decorator
# def example_visualize(Q, logger=None):
#     Q.update_defects(logger=logger)
#     Q.update_lines_classify(logger=logger)
#     Q.update_lines_smoothen(logger=logger)
#     Q.visualize_disclination_lines(logger=logger)
    
# # example_visualize(Q, log_level=logging.DEBUG, show_timestamp=True)
# example_visualize(Q, log_level=logging.DEBUG, show_timestamp=True, log_mode='none')


Q.update_defects()
Q.update_lines_classify()
Q.update_lines_smoothen()
Q.visualize_disclination_lines(is_wrap=True, lines_color_input_all=(0,0,0), extent_radius=0.05)

# Q.update_defects()
# mlab.figure()
# mlab.points3d(*(Q._defect_indices).T, scale_factor=0.5)


'''
Q.update_corners()
extent = Nematics3D.PlotExtent(Q._corners)
test = Nematics3D.PlotPlaneGrid((1,1,1), 100, 100, 200, corners_limit=Q._corners, origin=(64,64,64))

Nematics3D.PlotExtent(Q._corners)
mlab.points3d(*(test._grid.T))
'''
Q.update_corners()
# extent = Nematics3D.PlotExtent(Q._corners, radius=0.02)
# plane = Nematics3D.PlotPlaneGrid((1,1,1), 10, 10, index_max, corners_limit=Q._corners, origin=(index_max/2, index_max/2, index_max/2))
# plane = Nematics3D.PlotPlaneGrid((1,0,0), 10, 10, 200, axis1=(0,1,0), corners_limit=Q._corners, origin=(64,64,64))
# mlab.points3d(*(plane._grid.T))
interpolator = Q.update_integrator()
# Nematics3D.PlotnPlane((1,1,1), 5, 200, Q._interpolator, corners_limit=Q._corners, origin=(64,64,64))
# Nematics3D.PlotnPlane((1,0,0), 3, index_max, Q._interpolator, axis1=(0,1,0), corners_limit=Q._corners, origin=(0,index_max/2,index_max/2))
test = Nematics3D.PlotnPlane((0,0,1), 3, 0.85*index_max, Q._interpolator, axis1=(1,0,0), corners_limit=Q._corners, origin=(index_max/2,index_max/2,0), length=3, opacity=0.5)
g = test.items[0]








# space_index_ratio = 128 / np.array(np.shape(n)[:-1])

# # Nematics3D.visualize_nematics_field(n=n, plotn=True, plotdefects=True, plotS=False, defect_indices=defect_indices,
# #                                     space_index_ratio=space_index_ratio,
# #                                     n_opacity=0.1, defect_n_opacity=1, n_plane_index=[[0],[0],[0]], n_interval=15,
# #                                     S_is_colorbar=False, n_is_colorbar=False,
# #                                     n_is_color_immerse=False)
# # mlab.view(distance=450)

# defect_indices = Nematics3D.defect_detect(n, is_boundary_periodic=1)
# lines = Nematics3D.disclination.defect_classify_into_lines(defect_indices, box_size_periodic = (128, 128, 128))

