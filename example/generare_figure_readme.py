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
Q.figs[0].save('figures', 'lines.png')

Q.act_visualize_disclination_lines(is_wrap=False)
Q.figs[1].save('figures', 'lines_unwrap.png')

Q.act_visualize_disclination_lines(is_wrap=True, min_line_length=40)
extent = Q.figs[1].objects['extent'][0]
extent.opacity = 0.5
extent.radius = 0.2
extent.color = [1,0,0]
scene = Q.figs[1].scene
scene.azimuth = 90
scene.elevation = 30
scene.roll = 30
scene.bgcolor = [0.5,0.5,0.5]
for line in Q.figs[1].objects['lines']:
    line.specular_power = 20
    line.specular_color = (1,0,0)
    line.radius = 2
    line.sides = 20



 






