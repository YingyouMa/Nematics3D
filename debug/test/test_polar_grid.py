import sys
sys.path.insert(0, 'D:/Document/GitHub/')
import Nematics3D

# test = Nematics3D.PlaneGridPolar(normal=(0,0,1), dr=0.4, arc_dist=0.4, R_max=10, origin=(0,0,0), start_layer=1 )
origin = (41.36, 40.36, 29.75)
test = Nematics3D.PlaneGridPolar(normal=(1,1,1), dr=1, R_max=40, origin=origin)

spheres = Nematics3D.PlotSphere(test._entity_grid, radius=0.1)     