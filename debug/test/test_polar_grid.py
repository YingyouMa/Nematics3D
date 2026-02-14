import sys
sys.path.insert(0, 'D:/Document/GitHub/')
import Nematics3D

test = Nematics3D.PlaneGridPolar(normal=(0,0,1), dr=0.4, R_max=20, origin=(0,0,0) )

spheres = Nematics3D.PlotSphere(test._entity_grid, radius=0.1)     