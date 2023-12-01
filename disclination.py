# -----------------------------------------------
# Basic detection and processing of disclinations
# Yingyou Ma, Physics @ Brandeis, 2023
# -----------------------------------------------

import numpy as np
import time

# -------------------------------------------------
# Detect the disclinations in the 3D director field
# -------------------------------------------------

def find_defect(n, threshold=0, print_time=False):

  now = time.time()

  N, M, L = np.shape(n)[:3]

  here = n[:, 1:, :-1]
  if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[:, :-1, :-1], here))
  here = np.einsum('lmn, lmni -> lmni',if_parallel, n[:, 1:, :-1])
  if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[:, 1:, 1:], here))
  here = np.einsum('lmn, lmni -> lmni',if_parallel, n[:, 1:, 1:])
  if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[:, :-1, 1:], here))
  here = np.einsum('lmn, lmni -> lmni',if_parallel, n[:, :-1, 1:])
  test = np.einsum('lmni, lmni -> lmn', n[:, :-1, :-1], here)
  defect_indices = np.array(np.where(test<threshold)).transpose().astype(float)
  defect_indices[:,1:] = defect_indices[:,1:]+0.5
  if print_time == True:
    print('finish x-direction, with', str(round(time.time()-now,2))+'s')
  now = time.time()

  here = n[1:, :, :-1]
  if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[:-1,:, :-1], here))
  here = np.einsum('lmn, lmni -> lmni',if_parallel, n[1:, :, :-1])
  if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[1:, :, 1:], here))
  here = np.einsum('lmn, lmni -> lmni',if_parallel, n[1:, :, 1:])
  if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[:-1, :, 1:], here))
  here = np.einsum('lmn, lmni -> lmni',if_parallel, n[:-1, :, 1:])
  test = np.einsum('lmni, lmni -> lmn', n[:-1, :, :-1], here)
  temp = np.array(np.where(test<threshold)).transpose().astype(float)
  temp[:, [0,2]] = temp[:, [0,2]]+0.5
  defect_indices = np.concatenate([ defect_indices, temp ])
  if print_time == True:
    print('finish y-direction, with', str(round(time.time()-now,2))+'s')
  now = time.time()

  here = n[1:, :-1]
  if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[:-1, :-1], here))
  here = np.einsum('lmn, lmni -> lmni',if_parallel, n[1:, :-1])
  if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[1:, 1:], here))
  here = np.einsum('lmn, lmni -> lmni',if_parallel, n[1:, 1:])
  if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[:-1, 1:], here))
  here = np.einsum('lmn, lmni -> lmni',if_parallel, n[:-1, 1:])
  test = np.einsum('lmni, lmni -> lmn', n[:-1, :-1], here)
  temp = np.array(np.where(test<threshold)).transpose().astype(float)
  temp[:, :-1] = temp[:, :-1]+0.5
  defect_indices = np.concatenate([ defect_indices, temp ])
  if print_time == True:
    print('finish z-direction, with', str(round(time.time()-now,2))+'s')
  now = time.time()

  return defect_indices

# -------------------------------------------------------------------------
# Sort the index of defect points within a disclination loop
# Minimize the distance between the pair of points with neighboring indices
# -------------------------------------------------------------------------

def sort_loop_indices(defect_indices):
  loop_indices = defect_indices[nearest_neighbor_order(defect_indices)]
  return loop_indices

def nearest_neighbor_order(points):
    from scipy.spatial.distance import cdist
    num_points = len(points)
    dist = cdist(points, points) 

    visited = np.zeros(num_points, dtype=bool)
    visited[0] = True
    order = [0]  

    for i in range(num_points - 1):
        current_point = order[-1]
        nearest_neighbor = np.argmin(dist[current_point, :] + visited * np.max(dist))
        order.append(nearest_neighbor)
        visited[nearest_neighbor] = True

    return order




# -------------------------------------------------------------------------
# Visualize the disclination loop with directors lying on one cross section
# -------------------------------------------------------------------------

def show_loop_plane(
                    loop_box_indices, n_whole, 
                    width=0, margin_ratio=0.6, upper=0, down=0, norm_index=0, 
                    tube_radius=0.25, tube_opacity=0.5, scale_n=0.5,
                    print_load_mayavi=False
                    ):
    
    if print_load_mayavi == True:
        now = time.time()
        from mayavi import mlab
        print(f'loading mayavi cost {round(time.time()-now, 2)}s')
    else:
        from mayavi import mlab
    from .field import select_subbox, local_box_diagonalize

    def SLP_plot_plane(upper, down, d_box, grid, norm_vec, n_box, scale_n):

        index = (d_box<upper) * (d_box>down)
        index = np.where(index == True)
        n_plane = n_box[index]
        scalars = np.abs(np.einsum('ij, j -> i', n_plane, norm_vec))

        X, Y, Z = grid
        cord1 = X[index] - n_plane[:,0]/2
        cord2 = Y[index] - n_plane[:,1]/2
        cord3 = Z[index] - n_plane[:,2]/2

        vector = mlab.quiver3d(
                cord1, cord2, cord3,
                n_plane[:,0], n_plane[:,1], n_plane[:,2],
                mode = '2ddash',
                scalars = scalars,
                scale_factor=scale_n,
                opacity = 1
                )
        vector.glyph.color_mode = 'color_by_scalar'
        lut_manager = mlab.colorbar(object=vector)
        lut_manager.data_range=(0,1)

    def SLP_plot_loop(n_box, origin, N, width, tube_radius, tube_opacity):

        loop_indices = find_defect(n_box)
        if len(loop_indices) > 0:
            loop_indices = loop_indices + np.tile(origin, (np.shape(loop_indices)[0],1) )
            loop_coord = sort_loop_indices(loop_indices)/N*width
            loop_coord = np.concatenate([loop_coord, [loop_coord[0]]])
            mlab.plot3d(*(loop_coord.T), tube_radius=tube_radius, opacity=tube_opacity)

    N = np.shape(n_whole)[0]
    if width == 0:
        width = N

    # Find the region enclosing the loop. The size of the region is controlled by margin_ratio
    sl0, sl1, sl2, _ = select_subbox(loop_box_indices, 
                                [N, N, N], 
                                margin_ratio=margin_ratio
                                )

    # Select the local n around the loop
    n_box = n_whole[sl0,sl1,sl2]

    eigvec, eigval = local_box_diagonalize(n_box)

    # The directors within one cross section of the loop will be shown
    # Select the cross section by its norm vector
    # The norm of the principle plane is the eigenvector corresponding to the smallest eigenvalue
    norm_vec = eigvec[norm_index]

    # Build the grid for visualization
    x = np.arange( loop_box_indices[0][0], loop_box_indices[0][-1]+1 )/N*width
    y = np.arange( loop_box_indices[1][0], loop_box_indices[1][-1]+1 )/N*width
    z = np.arange( loop_box_indices[2][0], loop_box_indices[2][-1]+1 )/N*width
    grid = np.meshgrid(x,y,z, indexing='ij')

    # Find the height of the middle cross section: dmean
    d_box = np.einsum('iabc, i -> abc', grid, norm_vec)
    dmean = np.average(d_box)

    down, upper = np.sort([down, upper])
    if upper==down:
        upper = dmean + 0.5
        down  = dmean - 0.5
    else:
        upper = dmean + upper
        down  = dmean + down

    mlab.figure(bgcolor=(0,0,0))
    SLP_plot_plane(upper, down, d_box, grid, norm_vec, n_box, scale_n)
    SLP_plot_loop(
                  n_box, loop_box_indices[:,0], 
                  N, width, tube_radius, tube_opacity
                  )

    return dmean, eigvec, eigval


    
  
  
  
