import numpy as np
import time
from typing import Union, Sequence, Optional, List, Tuple

# ----------------------------------------------------------
# Functions which are being used and general.
# General means the code is for general nematics analysis.
# Not general means the code is specifically for my project.
# ----------------------------------------------------------

from .datatypes import (
    Vect,
    nField,
    DimensionPeriodicInput,
    DimensionFlagInput,
    as_dimension_info,
    DefectIndex,
    check_Sn,
)
from .grid import GRID_TRANSFORM_IDENTITY, as_grid_transform
from .logging_decorator import logging_and_warning_decorator
from .field import align_stack, add_periodic_boundary, Q_diagonalize

# from .debug.debug_store import DEBUG_VARS


DEFECT_NEIGHBOR = np.zeros((10, 3))
DEFECT_NEIGHBOR[0] = (1, 0, 0)
DEFECT_NEIGHBOR[1] = (-1, 0, 0)
DEFECT_NEIGHBOR[2] = (0.5, 0.5, 0)
DEFECT_NEIGHBOR[3] = (0.5, -0.5, 0)
DEFECT_NEIGHBOR[4] = (0.5, 0, 0.5)
DEFECT_NEIGHBOR[5] = (0.5, 0, -0.5)
DEFECT_NEIGHBOR[6] = (-0.5, 0.5, 0)
DEFECT_NEIGHBOR[7] = (-0.5, -0.5, 0)
DEFECT_NEIGHBOR[8] = (-0.5, 0, 0.5)
DEFECT_NEIGHBOR[9] = (-0.5, 0, -0.5)


def defect_detects_xyplane(n: np.ndarray, threshold: float) -> np.ndarray:
    """
    Detect defects in xy-plane of a reoriented director field (z as loop normal).

    Parameters
    ----------
    n : nField, np.ndarray
        Director field of shape (A, B, C, 3), where C is the loop-normal axis.

    threshold : float
        Threshold for defect detection.

    Returns
    -------
    coords : np.ndarray
        Coordinates of detected defects in reoriented space.
    """

    n = check_Sn(n, "n")

    a_orig = n[:-1, :-1]
    b_orig = n[1:, :-1]
    c_orig = n[1:, 1:]
    d_orig = n[:-1, 1:]
    stack = np.stack([a_orig, b_orig, c_orig, d_orig], axis=0)
    aligned_stack = align_stack(stack)
    a, b, c, d = aligned_stack

    test = np.einsum("...i,...i->...", a, d)

    coords = np.array(np.where(test < threshold)).T.astype(float)
    coords[:, [0, 1]] += 0.5

    return coords


@logging_and_warning_decorator()
def defect_detect(
    n_origin: nField,
    threshold: float = 0,
    is_boundary_periodic: DimensionFlagInput = 0,
    planes: DimensionFlagInput = 1,
    logger=None,
) -> DefectIndex:
    """
    Detect defects in a 3D director field.
    For each small loop formed by four neighoring grid points,
    calculate the inner product between the beginning and end director,
    where we enforce the successive directors have the similar orientation to handle the nematic symmetry.
    The indices of defect will be represented by one integer and two half-integers.
    A detailed introduction of this algorithm with illustration is elaborated in the FIG. 1 of the following paper:
    Coexistence of Defect Morphologies in Three-Dimensional Active Nematics, PRL


    Parameters
    ----------
    n_origin : nField
        Director field of shape (Nx, Ny, Nz, 3).
        Must be a float array representing unit vectors at each grid point.

    threshold : float, optional
        Threshold for detecting a defect. A defect is identified if the inner product
        between the starting and ending directors around a loop is less than this value.
        Default is 0.

    is_boundary_periodic : DimensionFlagInput, optional
        Accepts a bool or a sequence of 3 bools.
        Whether to apply periodic boundary conditions in each dimension.
        Default is 0 (no periodicity).

    planes : DimensionFlagInput, optional
        Accepts a bool or a sequence of 3 bools.
        Axes along which to compute loop windings. Each index indicates whether
        to consider plaquettes normal to x-, y-, or z-direction respectively.
        For example, planes=[1,0,0] analyzes only yz-planes (perpendicular to x).
        Default is [1, 1, 1].

    logger : Logger, optional
        Logger object used for internal messages.
        Automatically handled by decorator logging_and_warning_decorator().

    Returns
    -------
    defect_indices : DefectIndex
        Array of shape (N_defects, 3), where each row represents the index of a detected defect.
        Each index has one integer component and two half-integer components.
        The geometrical meaning of these components is explained in the definition of `DefectIndex`
        in `datatype.py`.
    """

    n_origin = check_Sn(n_origin, "n")

    is_boundary_periodic = as_dimension_info(is_boundary_periodic)
    planes = as_dimension_info(planes)

    logger.debug(
        "Start to defect defects. \n"
        f"Periodic boundary flags: {is_boundary_periodic}. \n"
        f"Threshold of the inner product between the first and last director is {threshold}."
    )

    n = add_periodic_boundary(n_origin, is_boundary_periodic)
    defect_indices = np.empty((0, 3), dtype=float)

    axis_permutations = {
        0: (2, 1, 0),  # x-direction → move axis 0 to back
        1: (0, 2, 1),  # y-direction → move axis 1 to back
        2: (0, 1, 2),  # z-direction → identity
    }

    now = time.time()

    for axis in range(3):
        if not planes[axis]:
            continue

        perm = axis_permutations[axis]
        n_rot = np.moveaxis(n, [0, 1, 2], perm)  # shape (A, B, C, 3)

        coords = defect_detects_xyplane(n_rot, threshold)

        # Restore original axis order
        inv_perm = np.argsort(perm)
        coords = coords[:, inv_perm]

        defect_indices = np.vstack((defect_indices, coords))
        logger.debug(
            f"Finished axis {axis}-direction in {round(time.time() - now, 2)}s"
        )
        now = time.time()

    # Wrap indices under periodic conditions
    for i, periodic in enumerate(is_boundary_periodic):
        if periodic:
            defect_indices[:, i] %= n_origin.shape[i]

    defect_indices, _ = np.unique(defect_indices, axis=0, return_index=True)

    return defect_indices


@logging_and_warning_decorator()
def defect_detect_surface(
    surface,
    ndata,
    threshold: float = 0,
    *,
    is_simplify: bool = True,
    is_return_mask: bool = False,
    logger=None,
) -> np.ndarray:
    """
    Detect surface defects on a triangulated PolyData surface.

    For each internal edge of the surface triangulation, the two adjacent
    triangles form a quadrilateral. The four directors at the quad vertices
    are walked around the loop with nematic sign-alignment, and a defect is
    flagged when the inner product of the first and last aligned director
    falls below ``threshold``.

    Parameters
    ----------
    surface : pyvista.PolyData
        A triangulated surface (e.g. ``SurfaceSampling.calc_surface_clean``).
        Every cell must be a triangle.

    ndata : GridInterpolator or np.ndarray, shape (V, 3)
        Source of the director field at the surface vertices.  Pass a
        ``GridInterpolator`` to evaluate the Q-tensor and derive directors
        internally, or pass a pre-computed director array directly to skip
        the interpolation step.  The array must have the same number of rows
        as the surface has vertices.

    threshold : float, optional
        Inner-product threshold below which a quad loop is classified as a
        defect. Default is 0.

    is_simplify : bool, optional
        When ``True``, consolidate redundant detections caused by a single
        defect triggering all three edges of the same triangle.  Any triangle
        whose three edges all produce defect quads is replaced by a single
        point at the triangle centroid; the three individual quad centroids
        are dropped.  Quads that do not belong to such a fully-defective
        triangle are kept as-is.  Default is ``True``.

    is_return_mask : bool, optional
        When ``True``, a second return value is added: a boolean array of
        shape ``(V,)`` marking which surface vertices participate in any
        detected defect quad or defect triangle.  The caller can index the
        surface vertex coordinates or interpolated directors directly with
        this mask.  Default is ``False``.

    logger : Logger, optional
        Automatically handled by ``logging_and_warning_decorator``.

    Returns
    -------
    defect_coords : np.ndarray, shape (D, 3)
        Physical coordinates of detected defect centres.
    near_defect_mask : np.ndarray of bool, shape (V,)
        Only returned when ``is_return_mask=True``.  ``True`` for each surface
        vertex that belongs to a defect quad or defect triangle.
    """
    # ------------------------------------------------------------------ #
    # 1. Evaluate director field at surface vertices                       #
    # ------------------------------------------------------------------ #
    if isinstance(ndata, np.ndarray):
        # Surface is assumed to already be a clean triangulated mesh whose
        # vertex order matches the supplied director array.  Skipping
        # .clean() avoids vertex reordering that would break the mapping.
        surface = surface.triangulate()
        vertices = np.asarray(surface.points, dtype=float)   # (V, 3)
        n = np.asarray(ndata, dtype=float)
        if n.shape != (len(vertices), 3):
            raise ValueError(
                f"`ndata` array has shape {n.shape} but surface has "
                f"{len(vertices)} vertices; expected shape ({len(vertices)}, 3)."
            )
    else:
        surface = surface.triangulate().clean()
        vertices = np.asarray(surface.points, dtype=float)   # (V, 3)
        Q = ndata.interpolate(vertices)
        _, n = Q_diagonalize(Q)                              # (V, 3)

    logger.debug(
        f"Evaluated director at {len(vertices)} surface vertices."
    )

    # ------------------------------------------------------------------ #
    # 2. Build edge → [tri_idx, opposite_vertex] mapping                  #
    # ------------------------------------------------------------------ #
    faces = np.asarray(surface.faces, dtype=int).reshape(-1, 4)
    triangles = faces[:, 1:]                             # (T, 3)

    # Each internal edge is shared by exactly two triangles.
    # Map (min_v, max_v) → list of (tri_idx, opposite_vertex_idx)
    edge_map = {}
    for tri_idx, (a, b, c) in enumerate(triangles):
        for edge, opp in (
            ((min(a, b), max(a, b)), c),
            ((min(b, c), max(b, c)), a),
            ((min(a, c), max(a, c)), b),
        ):
            if edge not in edge_map:
                edge_map[edge] = []
            edge_map[edge].append((tri_idx, opp))

    # ------------------------------------------------------------------ #
    # 3. Collect quads from internal edges                                 #
    # ------------------------------------------------------------------ #
    # For each internal edge (i, j) with opposing vertices k and l,
    # sort the four points by polar angle in the local tangent plane so
    # they form a proper (non-self-intersecting) loop.
    quad_list = []
    quad_tri_pairs = []                                  # (tri_idx_0, tri_idx_1) per quad
    for (vi, vj), entries in edge_map.items():
        if len(entries) != 2:
            continue                                     # boundary edge
        tri_idx_0, vk = entries[0]
        tri_idx_1, vl = entries[1]
        four = np.array([vi, vj, vk, vl], dtype=int)

        # Project to local tangent plane centred at the quad centroid.
        pts4 = vertices[four]                            # (4, 3)
        centroid = pts4.mean(axis=0)
        rel = pts4 - centroid                            # (4, 3)

        # Build an orthonormal 2-D frame from the first relative vector.
        u = rel[0]
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-14:
            continue
        u = u / u_norm

        # Normal from cross product of two independent edge vectors.
        normal = np.cross(rel[0], rel[1])
        n_norm = np.linalg.norm(normal)
        if n_norm < 1e-14:
            continue
        normal = normal / n_norm
        v = np.cross(normal, u)

        # Polar angles and sort.
        px = rel @ u
        py = rel @ v
        order = np.argsort(np.arctan2(py, px))
        quad_list.append(four[order])
        quad_tri_pairs.append((tri_idx_0, tri_idx_1))

    if not quad_list:
        logger.debug("No internal edges found; returning empty result.")
        empty_coords = np.empty((0, 3), dtype=float)
        if is_return_mask:
            return empty_coords, np.zeros(len(vertices), dtype=bool)
        return empty_coords

    quads = np.array(quad_list, dtype=int)               # (Q, 4)
    logger.debug(f"Formed {len(quads)} quad loops from internal edges.")

    # ------------------------------------------------------------------ #
    # 4. Vectorised loop integrals                                         #
    # ------------------------------------------------------------------ #
    # stack shape: (4, Q, 3)
    directors = np.stack([n[quads[:, k]] for k in range(4)], axis=0).copy()
    aligned = align_stack(directors)

    dot_first_last = np.einsum("qi,qi->q", aligned[0], aligned[3])
    defect_mask = dot_first_last < threshold

    logger.debug(
        f"Detected {defect_mask.sum()} defect quads "
        f"(threshold={threshold})."
    )

    if not is_simplify:
        defect_coords = vertices[quads[defect_mask]].mean(axis=1)
        if is_return_mask:
            near_defect_mask = np.zeros(len(vertices), dtype=bool)
            near_defect_mask[quads[defect_mask].ravel()] = True
            return defect_coords, near_defect_mask
        return defect_coords

    # ------------------------------------------------------------------ #
    # 5. Simplify: replace fully-defective triangles with their centroid  #
    # ------------------------------------------------------------------ #
    # Count how many defect quads each triangle participates in.
    defect_quad_indices = np.where(defect_mask)[0]
    tri_defect_edge_count = np.zeros(len(triangles), dtype=int)
    for q_idx in defect_quad_indices:
        t0, t1 = quad_tri_pairs[q_idx]
        tri_defect_edge_count[t0] += 1
        tri_defect_edge_count[t1] += 1

    # A triangle is "fully defective" when all three of its edges are
    # defect edges (count == 3).
    fully_defective_tris = np.where(tri_defect_edge_count == 3)[0]
    fully_defective_set = set(fully_defective_tris.tolist())

    # Collect triangle centroids for fully-defective triangles.
    tri_coords = []
    for t_idx in fully_defective_tris:
        tri_coords.append(vertices[triangles[t_idx]].mean(axis=0))

    # Keep quad centroids for defect quads that do NOT belong to any
    # fully-defective triangle.
    quad_coords = []
    for q_idx in defect_quad_indices:
        t0, t1 = quad_tri_pairs[q_idx]
        if t0 not in fully_defective_set and t1 not in fully_defective_set:
            quad_coords.append(vertices[quads[q_idx]].mean(axis=0))

    parts = []
    if tri_coords:
        parts.append(np.array(tri_coords, dtype=float))
    if quad_coords:
        parts.append(np.array(quad_coords, dtype=float))

    if not parts:
        empty_coords = np.empty((0, 3), dtype=float)
        if is_return_mask:
            return empty_coords, np.zeros(len(vertices), dtype=bool)
        return empty_coords

    defect_coords = np.vstack(parts)
    logger.debug(
        f"After simplification: {len(fully_defective_tris)} triangle centres "
        f"+ {len(quad_coords)} residual quad centres = {len(defect_coords)} total."
    )

    if not is_return_mask:
        return defect_coords

    near_defect_mask = np.zeros(len(vertices), dtype=bool)
    for t_idx in fully_defective_tris:
        near_defect_mask[triangles[t_idx]] = True
    for q_idx in defect_quad_indices:
        t0, t1 = quad_tri_pairs[q_idx]
        if t0 not in fully_defective_set and t1 not in fully_defective_set:
            near_defect_mask[quads[q_idx]] = True
    return defect_coords, near_defect_mask


@logging_and_warning_decorator()
def defect_classify_into_lines(
    defect_indices: DefectIndex,
    box_size_periodic: DimensionFlagInput = np.inf,
    grid_offset: Optional[Vect(3)] = None,
    grid_transform=GRID_TRANSFORM_IDENTITY,
    logger=None,
) -> List["DisclinationLine"]:
    """
    Group defect points into disclination lines based on graph connectivity.

    This function treats each defect point as a graph node, and forms edges between
    spatially adjacent nodes (using `defect_neighbor_possible_get` and periodicity).
    The resulting undirected graph is decomposed into connected components,
    each representing a disclination line.

    Each line is then:
    - Unwrapped across periodic boundaries
    - Transformed into physical coordinates via `transform` and `offset`
    - Encapsulated as a `DisclinationLine` object

    Parameters
    ----------
    defect_indices : DefectIndex, np.ndarray of shape (N_defects, 3)
        Grid indices of all the defects composing the line.
        Each point should contain one integer and two half-integers (e.g., [1, 3.5, 7.5]).
        The geometrical meaning of these components is explained in the definition of `DefectIndex`
        in `datatype.py`.

    box_size_periodic : DimensionPeriodic,
        array_like of 3 ints or a single int
        Grid size in each dimension, used to infer periodicity.
        If a single float `x` is provided, it is interpreted as (x, x, x).
        Use `np.inf` for non-periodic directions.
        Example: [128, 128, np.inf] indicates periodicity in x and y only.

    offset : Vect(3), array_like of 3 floats, optional
        Global offset added to all coordinates after transformation.
        Useful for shifting lines in real space.
        Default is None (no shift).

    transform : np.ndarray of shape (3, 3), optional
        Linear transformation matrix applied to the defect indices
        to convert from grid space to physical space (e.g., for anisotropic grids).
        Default is the canonical identity transform.

    logger : Logger object, optional
        Used internally by the logging decorator: logging_and_warning_decorator()

    Returns
    -------
    lines : list of DisclinationLine
        A list of disclination line objects, each representing one connected component
        (i.e., one continuous defect trajectory).
    """

    from .classes.graph import Graph
    from .classes.disclination_line import DisclinationLine
    from .grid import unwrap_trajectory
    from .general import make_hash_table, search_in_reservoir

    box_size_periodic = as_dimension_info(box_size_periodic)
    grid_transform = as_grid_transform(grid_transform)

    logger.debug(
        "Start line classfication. \n"
        f"box_size_periodic: {box_size_periodic}."
    )

    defect_indices_hash = make_hash_table(defect_indices)

    graph = Graph()

    for idx1, defect in enumerate(defect_indices):
        neighbor = defect_neighbor_possible_get(
            defect, box_size_periodic=box_size_periodic
        )
        search = search_in_reservoir(
            neighbor, defect_indices_hash, is_reservoir_hash=True
        )
        search = search[~np.isnan(search)].astype(int)
        for idx2 in search:
            graph.add_edge(idx1, idx2)

    paths = graph.find_path()
    paths = [
        unwrap_trajectory(defect_indices[path], box_size_periodic=box_size_periodic)
        for path in paths
    ]
    logger.debug("Done!")

    lines = [
        DisclinationLine(
            defect_indices=path,
            box_size_periodic_index=box_size_periodic,
            grid_offset=grid_offset,
            grid_transform=grid_transform,
            is_sorted=True,
        )
        for path in paths
    ]

    return lines


def defect_neighbor_possible_get(
    defect_index: Union[Sequence[float], np.ndarray],
    box_size_periodic: DimensionPeriodicInput = np.inf,
) -> np.ndarray:
    """
    Compute all possible neighboring defect indices of a given defect in a 3D grid,
    and apply periodic boundary conditions by generating mirror points if necessary.

    Each defect index is represented as a tuple of three floats. One of them is an integer (the "layer" dimension),
    and the other two are half-integers (the pixel centers on that layer).

    The 10 possible neighbors include:
    - 2 direct neighbors along the layer axis
    - 4 diagonal neighbors shifting one half along one pixel axis
    - 4 diagonal neighbors shifting one half along the other pixel axis

    If the defect lies near a periodic boundary, the mirror images of neighbors are also included.

    Parameters
    ----------
    defect_index : array-like of 3 floats
        Defect position, where exactly one coordinate is integer (the layer),
        and the other two are half-integers.

    box_size_periodic : float or array-like of 3 floats, optional
        Size of the periodic domain in each direction. Use `np.inf` for non-periodic boundaries.
        If a single float is provided, it is broadcasted to all three dimensions.
        For example:
            [X+1, Y+1, np.inf] means periodic in x and y, open in z.
        Default is [np.inf, np.inf, np.inf], i.e., no periodicity.

    Returns
    -------
    result : np.ndarray of shape (10, 3) or more
        Neighboring defect positions, with additional mirrored points if periodic and near boundary.

    Raises
    ------
    ValueError
        If input shape is not (3,) or if the "layer" dimension cannot be identified.
    """

    from .grid import generate_mirror_point_periodic_boundary

    defect_index = np.asarray(defect_index, dtype=np.float64)
    if defect_index.shape != (3,):
        raise ValueError(
            f"defect_index must be a 3-element vector, got shape {defect_index.shape}"
        )

    # Standardize box_size format
    box_size_periodic = as_dimension_info(box_size_periodic)

    # Copy neighbor offset vectors: shape (10, 3)
    neighbor = DEFECT_NEIGHBOR.copy()

    # Identify the integer-valued index (i.e., the layer direction)
    layer_index = np.where(defect_index % 1 == 0)[0]
    if len(layer_index) != 1:
        raise ValueError(
            f"Exactly one coordinate must be integer (the layer). Got {defect_index}"
        )
    layer_index = layer_index[0]

    # If layer is not axis 0, swap axes to make math easier
    if layer_index != 0:
        neighbor[:, (0, layer_index)] = neighbor[:, (layer_index, 0)]

    # Shift base defect by all 10 neighbor directions
    result = np.tile(defect_index, (10, 1)) + neighbor

    # Determine if periodic mirror points are needed
    periodic_mask = box_size_periodic != np.inf
    if np.any(periodic_mask):
        coord_in_periodic = defect_index[periodic_mask]
        box_size_in_periodic = box_size_periodic[periodic_mask]

        # Near boundary condition check: if defect is close to periodic edge
        near_boundary = np.min(coord_in_periodic) <= 1 or np.any(
            coord_in_periodic >= box_size_in_periodic - 2
        )
        if near_boundary:
            result = [
                generate_mirror_point_periodic_boundary(
                    point, box_size_periodic=box_size_periodic
                )
                for point in result
            ]
            result = np.vstack(result)

    return result


def defect_vicinity_grid(defect_indices, num_shell=2):
    """
    Generate square-shell neighborhoods around lattice-aligned defect points.

    This function constructs integer grid coordinates forming square shells
    (with odd side lengths 1, 3, 5, ... up to `2*num_shell-1`) around defect
    positions that lie close to integer lattice planes in x, y, or z.
    For each such defect, the neighborhood points are generated on the
    plane perpendicular to the corresponding axis.

    Parameters
    ----------
    defect_indices : ndarray of shape (N, 3)
        Array of defect positions in 3D (floating-point coordinates).

    num_shell : int, default=2
        Number of square shells around each defect.
        The side lengths of the shells will be 1, 3, 5, ..., (2*num_shell-1).

    Returns
    -------
    result : ndarray of shape (N, 4*num_shell**2, 3), dtype=int
        Integer lattice coordinates of neighborhood points for each defect.
        Defects not aligned to a lattice plane remain filled with zeros.

    Notes
    -----
    - The function separates defects into three groups depending on whether
      their x, y, or z coordinate is closest to an integer (within tolerance).
    - For each group, square neighborhoods are constructed on the
      corresponding orthogonal plane.
    - The neighborhood size grows quadratically with `num_shell`.

    Examples
    --------
    >>> defects = np.array([[1.0, 2.5, 3.0], [4.0, 5.0, 6.5]])
    >>> grid = defect_vicinity_grid(defects, num_shell=2)
    >>> grid.shape
    (2, 16, 3)
    """

    defect_indices = np.asarray(defect_indices)
    if defect_indices.size == 0:
        return np.empty((0, 3), dtype=int)

    square_size_list = np.arange(1, 2 * num_shell + 1, 2)
    square_num_list = square_size_list + 1

    square_origin_list = np.arange(-0.5, -num_shell - 0.5, -1)
    square_origin_list = np.broadcast_to(square_origin_list, (2, num_shell)).T
    square_origin_list = np.hstack([np.zeros((num_shell, 1)), square_origin_list])

    length = 4 * num_shell**2

    result = np.zeros((np.shape(defect_indices)[0], length, 3))

    indexx = np.isclose(defect_indices[:, 0], np.round(defect_indices[:, 0]))
    indexy = np.isclose(defect_indices[:, 1], np.round(defect_indices[:, 1]))
    indexz = np.isclose(defect_indices[:, 2], np.round(defect_indices[:, 2]))

    defectx = defect_indices[indexx]
    defecty = defect_indices[indexy]
    defectz = defect_indices[indexz]

    from .general import get_square

    squarex = get_square(
        square_size_list, square_num_list, origin_list=square_origin_list, dim=3
    )
    squarey = squarex.copy()
    squarey[:, [0, 1]] = squarey[:, [1, 0]]
    squarez = squarex.copy()
    squarez[:, [0, 1]] = squarez[:, [1, 0]]
    squarez[:, [1, 2]] = squarez[:, [2, 1]]

    defectx = np.repeat(defectx, length, axis=0).reshape(
        np.shape(defectx)[0], length, 3
    )
    defecty = np.repeat(defecty, length, axis=0).reshape(
        np.shape(defecty)[0], length, 3
    )
    defectz = np.repeat(defectz, length, axis=0).reshape(
        np.shape(defectz)[0], length, 3
    )

    defectx = defectx + np.broadcast_to(squarex, (np.shape(defectx)[0], length, 3))
    defecty = defecty + np.broadcast_to(squarey, (np.shape(defecty)[0], length, 3))
    defectz = defectz + np.broadcast_to(squarez, (np.shape(defectz)[0], length, 3))

    result[indexx] = defectx
    result[indexy] = defecty
    result[indexz] = defectz

    result = result.astype(int)

    return result
