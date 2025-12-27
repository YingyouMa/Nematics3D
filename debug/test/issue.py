import numpy as np
import pyvista as pv

x = np.linspace(0, 6, 50)
coords = np.column_stack((0*x, 0*x, x))

scalars = 0.2 * np.abs(np.sin(coords[:, 2]))
opacity = np.abs(np.sin(coords[:, 2]))


poly = pv.MultipleLines(coords)
poly.point_data['scalars'] = scalars
poly.point_data['opacity'] = opacity
mesh = poly.tube()

pl = pv.Plotter()

my_actor = pl.add_mesh(mesh, scalars='scalars', opacity='opacity')
# my_actor = pl.add_mesh(mesh, scalars='scalars')
pl.show(interactive_update=True)

opacity_mesh = my_actor.mapper.dataset
my_actor.mapper.lookup_table.cmap = 'plasma'

# my_actor.mapper.set_scalars(my_actor.mapper.dataset['scalars'], 'scalars', custom_opac=True, opacity=my_actor.mapper.dataset['opacity'], cmap='plasma')