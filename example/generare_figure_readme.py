import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import Nematics3D

n = np.load( 'data/n_example_global.npy')
S = np.load( 'data/S_example_global.npy')

Q = Nematics3D.QFieldObject(S=S, n=n, box_periodic_flag=True)

Q.act_lines_classify()
Q.act_lines_smooth()

Q.act_visualize_disclination_lines()
Q.figs[0].save('figures/lines.png')

Q.act_visualize_disclination_lines(is_wrap=False)
Q.figs[1].save('figures/lines_unwrap.png')

Q.act_visualize_disclination_lines()
extent = Q.figs[2].objects['extent'][0]
extent.opacity = 0.5
extent.radius = 0.2
extent.color = [1,0,0]
scene = Q.figs[2].scene
scene.azimuth = 90
scene.elevation = 30
scene.roll = 30
scene.bgcolor = [0.5,0.5,0.5]
for line in Q.figs[2].objects['lines']:
    line.specular_power = 20
    line.specular_color = (1,0,0)
    line.radius = 2
    line.sides = 20
Q.figs[2].save('figures/lines_modified.png')


index_max =  64
n = np.load( 'data/n_example_global.npy')[0:index_max, 0:index_max, 0:index_max]
S = np.load( 'data/S_example_global.npy')[0:index_max, 0:index_max, 0:index_max]
Q = Nematics3D.QFieldObject(S=S, n=n)
Q.act_lines_classify()
Q.act_lines_smooth()

Q.act_visualize_disclination_lines(line_color=(0.5, 0.5, 0.5), extent_radius=0.1, line_radius=0.4)

n_length = 2.5
n_radius = 0.3
spacing = 2.5

Q.act_visualize_n_in_Q(plane_normal=(0,0,1), plane_spacing=spacing, plane_size=0.95*index_max, plane_origin=(int(index_max/2), int(index_max)/2,0), 
                       n_length=2.5, n_radius=0.3, n_opacity=1, n_opacity_defect=1,
                       is_new=False, is_extent=False)
Q.figs[0].save('figures/PlotnPlaneZ.png')

Q.act_visualize_disclination_lines(line_color=(0.5, 0.5, 0.5), extent_radius=0.1, line_radius=0.4)
Q.act_visualize_n_in_Q(plane_normal=(0,0,1), plane_spacing=spacing, plane_size=0.95*index_max, plane_origin=(int(index_max/2), int(index_max)/2,0), 
                       n_length=2.5, n_opacity=0.2, n_radius=0.3, n_is_n_defect=True,
                       is_new=False, is_extent=False)
Q.act_visualize_n_in_Q(plane_normal=(0,1,0), plane_spacing=spacing, plane_size=0.95*index_max, plane_origin=(int(index_max/2), 0, int(index_max)/2), 
                       n_length=2.5, n_opacity=0.2, n_radius=0.3, n_is_n_defect=True,
                       is_new=False, is_extent=False)
Q.act_visualize_n_in_Q(plane_normal=(1,0,0), plane_spacing=spacing, plane_size=0.95*index_max, plane_origin=(0, int(index_max/2), int(index_max)/2), 
                       n_length=2.5, n_opacity=0.2, n_radius=0.3, n_is_n_defect=True,
                       is_new=False, is_extent=False)
Q.figs[1].save('figures/PlotnPlaneXYZ.png')

Q.act_visualize_disclination_lines(line_color=(0.5, 0.5, 0.5), extent_radius=0.1, line_radius=0.4, min_line_length=1, is_smooth=False)
Q.act_visualize_n_in_Q(plane_normal=(0,0,1), plane_spacing=spacing, plane_size=0.95*index_max, plane_origin=(int(index_max/2), int(index_max)/2,0), 
                       n_length=2.5, n_opacity=0.2, n_radius=0.3, n_is_n_defect=True,
                       is_new=False, is_extent=False)
Q.act_visualize_n_in_Q(plane_normal=(0,1,0), plane_spacing=spacing, plane_size=0.95*index_max, plane_origin=(int(index_max/2), 0, int(index_max)/2), 
                       n_length=2.5, n_opacity=0.2, n_radius=0.3, n_is_n_defect=True,
                       is_new=False, is_extent=False)
Q.act_visualize_n_in_Q(plane_normal=(1,0,0), plane_spacing=spacing, plane_size=0.95*index_max, plane_origin=(0, int(index_max/2), int(index_max)/2), 
                       n_length=2.5, n_opacity=0.2, n_radius=0.3, n_is_n_defect=True,
                       is_new=False, is_extent=False)
Q.figs[2].save('figures/PlotnPlaneXYZall.png')

Q.act_visualize_disclination_lines(line_color=(0.5, 0.5, 0.5), extent_radius=0.1, line_radius=0.4)
trans = 7.5
Q.act_visualize_n_in_Q(plane_normal=(1,1,1), plane_spacing=spacing, plane_size=100, plane_origin=(index_max/2-trans,index_max/2-trans,index_max/2-trans), 
                       n_length=spacing, n_opacity=0.2, n_radius=0.3, 
                       scene_azimuth=90, scene_elevation=90, scene_distance=190,
                       is_new=False, is_extent=False)
Q.figs[3].save('figures/PlotnPlaneTilt.png')

# Q.figs[3].objects["nPlanes"][0].act_commit(opts_spacing=3, length=3, origin=(30,30,30), opts_normal=(1,1,0), opts_radius=0.1, sda=1)
# Q.figs[3].objects["nPlanes"][0].colors = (1,0,0)

 






