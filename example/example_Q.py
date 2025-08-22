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


# index_max =  128
# n = np.load( 'data/n_example_global.npy')[0:index_max, 0:index_max, 0:index_max]
# S = np.load( 'data/S_example_global.npy')[0:index_max, 0:index_max, 0:index_max]

# Q = Nematics3D.QFieldObject(S=S, n=n, box_periodic_flag=index_max >= 128)

# Q.update_defects()
# Q.update_lines_classify()
# Q.update_lines_smoothen()
# Q.visualize_disclination_lines(is_wrap=False, extent_color=(0.5,0.5,0.5))

# Q.update_lines_smoothen(min_line_length=30, window_length=21)
# Q.visualize_disclination_lines(is_wrap=True, min_line_length=10)
# extent = Q.figures[1].objects['extent'][0]
# extent.opts_opacity = 0.5
# extent.opts_radius = 0.2
# extent.opts_color = [1,0,0]
# scene = Q.figures[1].scene
# scene.azimuth = 90
# scene.elevation = 30
# scene.roll = 30
# scene.bgcolor = [0.5,0.5,0.5]
# for line in Q.figures[1].objects['lines']:
#     line.opts_specular_power = 20
#     line.opts_specular_color = (1,0,0)
#     line.opts_radius = 3
#     line.opts_sides = 20
    
    
index_max =  60
n = np.load( 'data/n_example_global.npy')[0:index_max, 0:index_max, 0:index_max]
S = np.load( 'data/S_example_global.npy')[0:index_max, 0:index_max, 0:index_max]

Q = Nematics3D.QFieldObject(S=S, n=n, box_periodic_flag=index_max >= 128)

Q.update_defects()
Q.update_lines_classify()
Q.update_lines_smoothen(window_length=16)
Q.visualize_disclination_lines(is_wrap=True, line_color=(0.5, 0.5, 0.5), extent_radius=0.05, min_line_length=30, radius=0.2)
    
trans = 7.5
spacing = 2.5
Q.visualize_n_in_Q(plane_normal=(1,1,1), plane_spacing=spacing, plane_size=100, 
                   plane_origin=(index_max/2-trans,index_max/2-trans,index_max/2-trans), 
                   n_length=spacing, n_opacity=0.2, n_radius=0.3, is_new=False, is_extent=False)
Q.figures[0].scene.azimuth = 90
Q.figures[0].scene.elevation = 90




# @Nematics3D.logging_and_warning_decorator
# def example_visualize(Q, logger=None):
#     Q.update_defects(logger=logger)
#     Q.update_lines_classify(logger=logger)
#     Q.update_lines_smoothen(logger=logger)
#     Q.visualize_disclination_lines(logger=logger)
    
# # example_visualize(Q, log_level=logging.DEBUG, show_timestamp=True)
# example_visualize(Q, log_level=logging.DEBUG, show_timestamp=True, log_mode='none')

# Q.update_defects()
# Q.update_lines_classify()
# opts_smoothen = Nematics3D.OptsSmoothen(min_line_length=50, window_length=41)
# Q.update_lines_smoothen(opts=opts_smoothen)
# Q.update_lines_smoothen(smoothen_min_line_length=50, smoothen_window_length=41)
# Q.update_lines_smoothen()
# Q.visualize_disclination_lines(is_wrap=False, extent_color=(0.5,0.5,0.5))

# for line in Q._lines:
#     if line._defect_num > 50:
#         line.visualize(is_wrap=False, opts=Nematics3D.OptsTube(color=(1,0,0)))

# trans = 0
# Q.update_defects()
# Q.update_lines_classify()
# Q.update_lines_smoothen(min_line_length=30, window_length=21)
# Q.visualize_disclination_lines(is_wrap=True, lines_color_input_all=(0,0,0), extent_radius=0.05, min_line_length=30, radius=0.2)
# # Q.visualize_n_in_Q((1,1,1), 1.5, 100, origin=(index_max/2-trans,index_max/2-trans,index_max/2-trans), length=1.5, opacity=0.2, radius=0.2, is_new=False, is_extent=False)
# Q.visualize_n_in_Q((1,0,0), 1.5, index_max, axis1=(0,1,0), origin=(0,index_max/2,index_max/2), length=1.5, opacity=0.2, radius=0.2, is_new=False, is_extent=False)
# Q.visualize_n_in_Q((0,1,0), 1.5, index_max, axis1=(0,0,1), origin=(index_max/2,0,index_max/2), length=1.5, opacity=0.2, radius=0.2, is_new=False, is_extent=False)
# Q.visualize_n_in_Q((0,0,1), 1.5, index_max, axis1=(1,0,0), origin=(index_max/2,index_max/2,0), length=1.5, opacity=0.2, radius=0.2, is_new=False, is_extent=False)


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
# Q.update_corners()
# extent = Nematics3D.PlotExtent(Q._corners, radius=0.02)
# plane = Nematics3D.PlotPlaneGrid((1,1,1), 10, 10, index_max, corners_limit=Q._corners, origin=(index_max/2, index_max/2, index_max/2))
# plane = Nematics3D.PlotPlaneGrid((1,0,0), 10, 10, 200, axis1=(0,1,0), corners_limit=Q._corners, origin=(64,64,64))
# mlab.points3d(*(plane._grid.T))
# interpolator = Q.update_interpolator()
# Nematics3D.PlotnPlane((1,1,1), 5, 200, Q._interpolator, corners_limit=Q._corners, origin=(64,64,64))
# Nematics3D.PlotnPlane((1,0,0), 3, index_max, Q._interpolator, axis1=(0,1,0), corners_limit=Q._corners, origin=(0,index_max/2,index_max/2))
# test = Nematics3D.PlotnPlane((0,0,1), 2, 0.9*index_max, Q._interpolator, axis1=(1,0,0), corners_limit=Q._corners, origin=(index_max/2,index_max/2,index_max/2), length=2, opacity=0.2, radius=0.25)

# test.update(normal=(1,1,1), size=100)
# normal = test.plane._normal
# axis1 = test.plane._axis1
# axis_both = np.array([axis1, np.cross(normal, axis1)])
# shape_all = np.shape(test.plane._grid_all)[:2]
# grid_all_flatten = np.reshape(test.plane._grid_all, (-1,3))
# Q_all = test._QInterpolator.interpolate(grid_all_flatten)
# _, n_all = Nematics3D.Q_diagonalize(Q_all)
# n_all = np.reshape(n_all, (1, *shape_all, 3))
# defect_plane_index = Nematics3D.defect_detect(n_all, planes=(True, False, False))
# defect_vicinity_index = Nematics3D.defect_vicinity_grid(defect_plane_index, num_shell=1).reshape((-1,3))[:, 1:]

# others_index, defect_vicinity_index = Nematics3D.split_points(test.plane._grid_i  nt, defect_vicinity_index)

# defect_vicinity = np.einsum('ai, ib -> ab', defect_vicinity_index, axis_both) * test.plane._space1 + test.plane._offset
# defect_vicinity = Nematics3D.select_grid_in_box(defect_vicinity, Q._corners)
# others = np.einsum('ai, ib -> ab', others_index, axis_both) * test.plane._space1 + test.plane._offset
# others = Nematics3D.select_grid_in_box(others, Q._corners)

# # defect_vicinity = Nematics3D.select_grid_in_box(defect_vicinity, Q._corners)
# mlab.points3d(*(defect_vicinity.T), scale_factor=2, color=(1,0,0))
# mlab.points3d(*(others.T), scale_factor=2, color=(0,0,1))
# mlab.points3d(*(Q._defect_indices.T), scale_factor=1)
# # temp = np.einsum('ai, ib -> ab', defect_plane_index[:,1:], axis_both) * test.plane._space1
# # temp = temp + test.plane._offset
# # temp = Nematics3D.select_grid_in_box(temp, Q._corners)
# # mlab.points3d(*(temp.T), scale_factor=2, color=(1,0,0))
# # mlab.points3d(*(Q._defect_indices.T), scale_factor=1)







# space_index_ratio = 128 / np.array(np.shape(n)[:-1])

# # Nematics3D.visualize_nematics_field(n=n, plotn=True, plotdefects=True, plotS=False, defect_indices=defect_indices,
# #                                     space_index_ratio=space_index_ratio,
# #                                     n_opacity=0.1, defect_n_opacity=1, n_plane_index=[[0],[0],[0]], n_interval=15,
# #                                     S_is_colorbar=False, n_is_colorbar=False,
# #                                     n_is_color_immerse=False)
# # mlab.view(distance=450)

# defect_indices = Nematics3D.defect_detect(n, is_boundary_periodic=1)
# lines = Nematics3D.disclination.defect_classify_into_lines(defect_indices, box_size_periodic = (128, 128, 128))

