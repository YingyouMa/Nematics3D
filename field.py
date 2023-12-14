# ------------------------------------
# Analysis of Q field in 3D
# Yingyou Ma, Physics @ Brandeis, 2023
# ------------------------------------

import numpy as np

# --------------------------------------------------------
# Diagonalization of Q tensor in 3D nematics.
# --------------------------------------------------------

def diagonalizeQ(qtensor):
    """
    Diagonalization of Q tensor in 3D nematics.
    Currently it onply provides the uniaxial information.
    Will be updated to derive biaxial analysis in the future.
    Algorythm provided by Matthew Peterson:
    https://github.com/YingyouMa/3D-active-nematics/blob/405c8d54d797cc39c1f14c82112cb43d304ef16c/reference/order_parameter_calculation.pdf

    Parameters
    ----------
    qtensor:  numpy array, tensor order parameter Q of each grid
              shape: (N, M, L, 5), where N, M and L are the number of grids in each dimension.
              qtensor[..., 0] = Q_xx, qtensor[..., 1] = Q_xy, and so on. 

    Returns
    -------
    S:  numpy array, the biggest eigenvalue as the scalar order parameter of each grid
        shape: (N, M, L)
    n:  numpy array, the eigenvector corresponding to the biggest eigenvalue, as the director, of each grid.
        shape: (N, M, L)
    """
    
    # derive Q field and calculate it with np.einsum() and np.linalg.det()

    N, M, L = np.shape(qtensor)[:3]

    if np.shape(qtensor) == (N, M, L, 3, 3):
        Q = qtensor
    elif np.shape(qtensor) == (N, M, L, 5):
        Q = np.zeros( (N, M, L, 3, 3)  )
        Q[..., 0,0] = qtensor[..., 0]
        Q[..., 0,1] = qtensor[..., 1]
        Q[..., 0,2] = qtensor[..., 2]
        Q[..., 1,0] = qtensor[..., 1]
        Q[..., 1,1] = qtensor[..., 3]
        Q[..., 1,2] = qtensor[..., 4]
        Q[..., 2,0] = qtensor[..., 2]
        Q[..., 2,1] = qtensor[..., 4]
        Q[..., 2,2] = - Q[..., 0,0] - Q[..., 1,1]
    else:
        raise NameError(
            "The dimension of qtensor would be (N, M, L, 3, 3) or (N, M, L, 5)"
            )

    p = 0.5 * np.einsum('ijkab, ijkba -> ijk', Q, Q)
    q = np.linalg.det(Q)
    r = 2 * np.sqrt( p / 3 )

    # derive S and n
    temp = 4 * q / r**3
    temp[temp>1]  =  1
    temp[temp<-1] = -1
    S = r * np.cos( 1/3 * np.arccos( temp ) )
    temp = np.array( [
        Q[..., 0,2] * ( Q[..., 1,1] - S ) - Q[..., 0,1] * Q[..., 1,2] ,
        Q[..., 1,2] * ( Q[..., 0,0] - S ) - Q[..., 0,1] * Q[..., 0,2] ,
        Q[..., 0,1]**2 - ( Q[..., 0,0] - S ) * ( Q[..., 1,1] - S  )
        ] )
    n = temp / np.linalg.norm(temp, axis = 0)
    n = n.transpose((1,2,3,0))
    S = S * 1.5

    return S, n

# ----------------------------------------------------------------------
# With the indices of a pair of opposite vertices of sub-orthogonal-box,
# select n and S within that subbox
# ----------------------------------------------------------------------

def select_subbox(subbox_indices, box_grid_size, 
                  margin_ratio=0):

    subbox  = subbox_indices
    N, M, L = box_grid_size

    if margin_ratio != 0:
        xrange, yrange, zrange = subbox_indices[:,1] - subbox_indices[:,0]
        margin = ( np.array([xrange, yrange, zrange]) * margin_ratio/2 ).astype(int)
        subbox[:,0] -= margin
        subbox[:,1] += margin

    xmin, ymin, zmin = subbox[:,0]
    xmax, ymax, zmax = subbox[:,1]

    sl0 = np.array(range(xmin, xmax+1)).reshape(-1,1, 1)%N
    sl1 = np.array(range(ymin, ymax+1)).reshape(1,-1, 1)%M
    sl2 = np.array(range(zmin, zmax+1)).reshape(1,1,-1)%L

    return sl0, sl1, sl2, subbox


# ----------------------------------------------------
# The biaxial analysis of directors within a local box
# ----------------------------------------------------

def local_box_diagonalize(n_box):

    # Derive and take the average of the local Q tensor with the director field around the loop
    Q = np.einsum('abci, abcj -> abcij', n_box, n_box)
    Q = np.average(Q, axis=(0,1,2))
    Q = Q - np.diag((1,1,1))/3

    # Diagonalisation and sort the eigenvalues.
    eigval, eigvec = np.linalg.eig(Q)
    eigvec = np.transpose(eigvec)
    eigidx = np.argsort(eigval)
    eigval = eigval[eigidx]
    eigvec = eigvec[eigidx]

    return eigvec, eigval

# ---------------------------------------------------------------------------
# Interpolate the directors within a local box containing a disclination loop
# The box is given by its unit vector and vertex coordinates
# ---------------------------------------------------------------------------

def interpolate_subbox(vertex_indices, axes_unit, loop_box, n, S, whole_box_grid_size,
                        margin_ratio=2, num_min=20, ratio=[1,1,1]):
    
    from itertools import product

    diagnal = vertex_indices[1] - vertex_indices[0]
    num_origin = np.einsum('i, ji -> j', diagnal, axes_unit)
    axes = np.einsum('i, ij -> ij', np.abs(num_origin) / num_origin, axes_unit)
    num_origin = np.abs(num_origin)
    num_scale = num_min / np.min(num_origin) * np.array(ratio)
    numx, numy, numz = np.round(num_scale * num_origin).astype(int)

    box = list(product(np.arange(numx+1), np.arange(numy+1), np.arange(numz+1)))
    box = np.array(box)
    box = np.einsum('ai, ij -> aj', box[:,:3], (axes.T/num_scale).T)
    box = box + vertex_indices[0]

    sl0, sl1, sl2, ortho_box = select_subbox(loop_box, whole_box_grid_size,
                                              margin_ratio=margin_ratio)
    n_box = n[sl0,sl1,sl2]
    S_box = S[sl0,sl1,sl2]

    Q_box = np.einsum('abci, abcj -> abcij', n_box, n_box)
    Q_box = Q_box - np.eye(3)/3
    Q_box = np.einsum('abc, abcij -> abcij', S_box, Q_box)

    xmin, ymin, zmin = ortho_box[:,0]
    xmax, ymax, zmax = ortho_box[:,1]
    points = (
        np.arange(xmin, xmax+1),
        np.arange(ymin, ymax+1),
        np.arange(zmin, zmax+1)
        )

    from scipy.interpolate import interpn
    def interp_Q(index1, index2):
      result = interpn(points, Q_box[..., index1, index2], box)
      result = np.reshape( result, (numx+1, numy+1, numz+1))
      return result

    Q_out = np.zeros( (numx+1, numy+1, numz+1, 3, 3)  )
    Q_out[..., 0,0] = interp_Q(0, 0)
    Q_out[..., 0,1] = interp_Q(0, 1)
    Q_out[..., 0,2] = interp_Q(0, 2)
    Q_out[..., 1,1] = interp_Q(1, 1)
    Q_out[..., 1,2] = interp_Q(1, 2)
    Q_out[..., 1,0] = Q_out[..., 0,1]
    Q_out[..., 2,0] = Q_out[..., 0,2]
    Q_out[..., 2,1] = Q_out[..., 1,2]
    Q_out[..., 2,2] = - Q_out[..., 0,0] - Q_out[..., 1,1]

    Q_out = np.einsum('ab, ijkbc, dc -> ijkad', axes_unit, Q_out, axes_unit)

    return Q_out






  
  
  
