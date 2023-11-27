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

def interpolate_box(origin, axes, num, ratio, loop_box, n, S, margin_ratio=2):

  from itertools import product

  numx, numy, numz = np.array(num) * np.array(ratio)

  e1 = np.array(axes[0]) / ratio[0]
  e2 = np.array(axes[1]) / ratio[1]
  e3 = np.array(axes[2]) / ratio[2]

  box = np.zeros(((numx+1)*(numy+1)*(numz+1), 3))
  box = np.array(list(product(np.arange(numx+1), np.arange(numy+1), np.arange(numz+1))))
  box = np.einsum('ai, ij -> aj', box[:,:3], np.array([e1,e2,e3])) + origin



  n_box = n[sl0,sl1,sl2]
  S_box = S[sl0,sl1,sl2]

    
  
  
  
