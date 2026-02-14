import time
import numpy as np
import pyvistaqt as pyqt
import logging
# import matplotlib.pyplot as plt

import sys
# sys.path.insert(0, 'D:/Document/GitHub/3D-active-nematics/simulation')
sys.path.insert(0, 'D:/Document/GitHub/')
import Nematics3D

from Nematics3D.debug.debug_store import DEBUG_VARS

# index_max =  128
# n = np.load( 'data/n_example_global.npy')[0:index_max, 0:index_max, 0:index_max]
# S = np.load( 'data/S_example_global.npy')[0:index_max, 0:index_max, 0:index_max]

# Q = Nematics3D.QFieldObject(S=S, n=n, box_periodic_flag=index_max >= 128, name="testQ")
# Q.act_lines_smooth()
# Q.act_visualize_disclination_lines(is_wrap=False, extent_color=(0.5,0.5,0.5))
# Q.act_visualize_disclination_lines(is_new=True, is_wrap=True, line_shading_type='pbr', figure_azimuth=0, figure_elevation=45)




# index_max =  60
# n = np.load( 'data/n_example_global.npy')[0:index_max, 0:index_max, 0:index_max]
# S = np.load( 'data/S_example_global.npy')[0:index_max, 0:index_max, 0:index_max]

# Q = Nematics3D.QFieldObject(S=S, n=n, box_periodic_flag=index_max >= 128)
# Q.act_lines_smooth(window_length=21, min_line_length=40)
# Q.act_visualize_disclination_lines(is_wrap=True, line_color=(0.5, 0.5, 0.5), extent_radius=0.05, min_line_length=50, line_radius=0.4)

# trans = 7.5
# Q.act_visualize_n_plane(
#     is_extent=False,
#     grid_spacing=3,
#     grid_origin=(index_max/2-trans,index_max/2-trans,index_max/2-trans),
#     grid_normal=(1,1,1),
#     figure_azimuth=90,
#     figure_elevation=0
# )

   
figure = Nematics3D.PlotFigure(is_off_screen=False)  
# figure = pyqt.BackgroundPlotter()
   
index_max =  60
n = np.load( 'data/n_example_global.npy')[0:index_max, 0:index_max, 0:index_max]
S = np.load( 'data/S_example_global.npy')[0:index_max, 0:index_max, 0:index_max]

Q = Nematics3D.QFieldObject(S=S, n=n, box_periodic_flag=index_max >= 128)
Q.act_lines_smooth(window_length=21, min_line_length=40)
Q.act_visualize_disclination_lines(is_wrap=True, line_color=(0.5, 0.5, 0.5), extent_radius=0.05, min_line_length=50, line_radius=0.4, figure=figure)
    
trans = 7.5
spacing = 2.5

testGrid = Nematics3D.PlaneGrid(normal=(1,1,1), spacing=spacing, size=100, origin=(index_max/2-trans,index_max/2-trans,index_max/2-trans), corners_limit=Q._calc_corners)
testnPlane = Nematics3D.QPlane(Q._calc_interpolator, grid=testGrid)

testnPlane.act_visualize_n(figure=figure)
# testnPlane.act_visualize_S(figure=figure)
figure.act_view_yz()
figure.opts.azimuth = 90

figure.act_savefig('test.png')
# figure.pl.screenshot('test.png', transparent_background=False)


# origin = (41.36, 40.36, 29.75)
origin = (21.58, 23.31, 23.08)
# origin = (42.12, 46.13, 23.22)
figure2 = Nematics3D.PlotFigure()  
Q.act_visualize_disclination_lines(is_wrap=True, line_color=(0.5, 0.5, 0.5), extent_radius=0.05, min_line_length=50, line_radius=0.4, figure=figure2)
testPolar = Nematics3D.PlaneGridPolar(normal=(1,1,1), dr=0.5, R_max=5, origin=origin )
# Nematics3D.PlotSphere(testPolar._entity_grid, figure=figure2)
QPolar = Q._calc_interpolator.interpolate(testPolar._entity_grid)
_, nPolar = Nematics3D.Q_diagonalize(QPolar)
rods = Nematics3D.PlotRod(coords=testPolar._entity_grid, orient=nPolar, figure=figure2, color=Nematics3D.n_color_immerse)

figure2.act_view_yz()
figure2.opts.azimuth = 90

def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def defect_detects_polar_flat(
    points: np.ndarray,          # (N, 3)
    polar: np.ndarray,           # (N, 2) [r, theta]
    ring_offsets: np.ndarray,    # (n_rings+1,)
    directors: np.ndarray,       # (N, 3)
    threshold: float=0,
):
    """
    Detect defects on flattened polar-ring sampling and return:
      - defect_centers: (K,3) mean of loop vertices (a,b,c,d) for flagged loops
      - adjacent_mask: (N,) bool, True for points participating in any flagged loop
    """
    from Nematics3D.field import align_directors  # use your existing definition

    points = np.asarray(points)
    polar = np.asarray(polar)
    ring_offsets = np.asarray(ring_offsets, dtype=np.int64)
    directors = np.asarray(directors)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3). Got {points.shape}.")
    if polar.ndim != 2 or polar.shape[1] != 2:
        raise ValueError(f"polar must be (N,2). Got {polar.shape}.")
    if directors.shape != points.shape:
        raise ValueError(f"directors must match points shape (N,3). Got {directors.shape} vs {points.shape}.")
    if ring_offsets.ndim != 1 or ring_offsets.shape[0] < 2:
        raise ValueError("ring_offsets must be 1D with length >= 2.")
    if ring_offsets[0] != 0 or ring_offsets[-1] != points.shape[0]:
        raise ValueError("ring_offsets must start at 0 and end at N (points.shape[0]).")

    n_rings = ring_offsets.shape[0] - 1
    adjacent_mask = np.zeros((points.shape[0],), dtype=bool)
    defect_centers_chunks = []

    # If ring 0 is origin block (size 1 at r=0), skip it for quad loops
    start_ring = 0
    if n_rings >= 1:
        s0, e0 = ring_offsets[0], ring_offsets[1]
        if (e0 - s0) == 1 and np.isclose(polar[s0, 0], 0.0):
            start_ring = 1

    # ring-pair loop: traverse from outer ring -> inner ring
    # For each pair (k, k+1), we take:
    #   outer = k+1, inner = k
    for k in range(start_ring, n_rings - 1):
        s_inner, e_inner = ring_offsets[k], ring_offsets[k + 1]
        s_outer, e_outer = ring_offsets[k + 1], ring_offsets[k + 2]

        n_inner = e_inner - s_inner
        n_outer = e_outer - s_outer
        if n_inner < 2 or n_outer < 2:
            continue

        theta_inner = polar[s_inner:e_inner, 1]    # (n_inner,)
        theta_outer = polar[s_outer:e_outer, 1]    # (n_outer,) may NOT be sorted

        # local indices on OUTER ring (base edge lives on outer ring)
        j = np.arange(n_outer, dtype=np.int64)
        jn = (j + 1) % n_outer

        idx_a = s_outer + j     # a on outer ring
        idx_b = s_outer + jn    # b on outer ring (next theta)

        # ----------------------------
        # Build the polar "cell" as (outer -> inner):
        #   a=(outer, j)
        #   b=(outer, j+1)
        #   c=(inner, nearest(theta_b))
        #   d=(inner, neighbor-of-c that is closer to theta_a)
        #
        # Here "neighbor-of-c" is defined by sorting theta_inner to form the circular
        # adjacency on the INNER ring (because c,d lie on the inner ring).
        # ----------------------------

        theta_a = theta_outer[j]     # (n_outer,)
        theta_b = theta_outer[jn]    # (n_outer,)

        # (1) c: nearest on INNER ring to theta_b (no sorted assumption)
        diff_b = _wrap_to_pi(theta_inner[None, :] - theta_b[:, None])        # (n_outer, n_inner)
        c_local = np.argmin(np.abs(diff_b), axis=1).astype(np.int64)         # (n_outer,)

        # (2) define INNER-ring adjacency by sorting theta_inner to form a circular order
        order = np.argsort(theta_inner)                                      # (n_inner,)
        rank_of = np.empty_like(order)
        rank_of[order] = np.arange(n_inner, dtype=np.int64)                  # local_index -> rank

        c_rank = rank_of[c_local]                                            # (n_outer,)
        prev_rank = (c_rank - 1) % n_inner
        next_rank = (c_rank + 1) % n_inner

        prev_local = order[prev_rank]                                        # (n_outer,)
        next_local = order[next_rank]                                        # (n_outer,)

        # (3) pick d as the neighbor of c closer to theta_a (circular distance on INNER ring)
        d_prev = np.abs(_wrap_to_pi(theta_inner[prev_local] - theta_a))      # (n_outer,)
        d_next = np.abs(_wrap_to_pi(theta_inner[next_local] - theta_a))      # (n_outer,)
        d_local = np.where(d_prev <= d_next, prev_local, next_local).astype(np.int64)

        idx_c = s_inner + c_local   # c on inner ring
        idx_d = s_inner + d_local   # d on inner ring

        # loop vertex coords
        pa = points[idx_a]
        pb = points[idx_b]
        pc = points[idx_c]
        pd = points[idx_d]

        # loop directors
        a = directors[idx_a]
        b_raw = directors[idx_b]
        c_raw = directors[idx_c]
        d_raw = directors[idx_d]

        # align along loop: a -> b -> c -> d
        b = align_directors(a, b_raw)
        c = align_directors(b, c_raw)
        d = align_directors(c, d_raw)

        test = np.einsum("...i,...i->...", a, d)  # (n_outer,)
        hit = test < threshold
        if not np.any(hit):
            continue

        centers = (pa + pb + pc + pd) * 0.25
        defect_centers_chunks.append(centers[hit])

        # adjacency mask: mark all points participating in hit loops
        adjacent_mask[idx_a[hit]] = True
        adjacent_mask[idx_b[hit]] = True
        inner_idx = np.unique(np.concatenate([idx_c[hit], idx_d[hit]]))
        adjacent_mask[inner_idx] = True

    defect_centers = (
        np.concatenate(defect_centers_chunks, axis=0).astype(float)
        if defect_centers_chunks
        else np.zeros((0, 3), dtype=float)
    )

    return defect_centers, adjacent_mask

test_defect_centers, test_adjacent_mask = defect_detects_polar_flat(
    testPolar._entity_grid,
    testPolar._entity_polar,
    testPolar._calc_ring_offsets,
    nPolar
    )

opacity = np.zeros(len(testPolar._entity_grid)) + 0.2
opacity[test_adjacent_mask] = 1
rods.opts.opacity = opacity



