import time
import numpy as np
from mayavi import mlab
import re
import glob
# import matplotlib.pyplot as plt

import sys
# sys.path.insert(0, 'D:/Document/GitHub/3D-active-nematics/simulation')
sys.path.insert(0, 'D:/Document/GitHub/')
import Nematics3D


L = 200

#Q = np.load( 'data/Q_example_local.npy')

# S, n = Nematics3D.diagonalizeQ(Q)

n = np.load( 'data/n_example_global.npy')
S = np.load( 'data/S_example_global.npy')

Q = Nematics3D.QFieldObject(S=S, n=n, box_size_periodic=128)

Q.update_defects()
Q.update_lines_classify()
Q.update_lines_smoothen()
Q.visualize_lines()









# space_index_ratio = 128 / np.array(np.shape(n)[:-1])

# # Nematics3D.visualize_nematics_field(n=n, plotn=True, plotdefects=True, plotS=False, defect_indices=defect_indices,
# #                                     space_index_ratio=space_index_ratio,
# #                                     n_opacity=0.1, defect_n_opacity=1, n_plane_index=[[0],[0],[0]], n_interval=15,
# #                                     S_is_colorbar=False, n_is_colorbar=False,
# #                                     n_is_color_immerse=False)
# # mlab.view(distance=450)

# defect_indices = Nematics3D.defect_detect(n, is_boundary_periodic=1)
# lines = Nematics3D.disclination.defect_classify_into_lines(defect_indices, box_size_periodic = (128, 128, 128))

