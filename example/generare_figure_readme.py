import numpy as np

import sys
# sys.path.insert(0, 'D:/Document/GitHub/3D-active-nematics/simulation')
sys.path.insert(0, 'D:/Document/GitHub/')
import Nematics3D

n = np.load( 'data/n_example_global.npy')
S = np.load( 'data/S_example_global.npy')

Q = Nematics3D.QFieldObject(S=S, n=n, box_periodic_flag=True)

Q.act_lines_classify()
Q.act_lines_smooth()

Q.act_visualize_disclination_lines()
Q.figs[0].save('figures/lines.png')