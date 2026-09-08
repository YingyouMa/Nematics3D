"""Surface detection, mask validity, and vicinity helpers for disclinations."""

import numpy as np

from ...q_field.diagonalization import q_diagonalize
from ...datatypes import (
    DefectIndex,
    DimensionInfo,
    MaskField,
    as_bool,
    as_defect_index,
    as_dimension_info,
    as_director_field,
    as_lattice_mask,
)
from ...field import align_director_stack
from ...logging_decorator import logging_and_warning_decorator
from .line import get_square


def defect_validity_from_mask(
    defect_indices: DefectIndex,
    mask: MaskField,
    is_boundary_periodic: DimensionInfo = 0,
) -> np.ndarray:
    """
    Judge which defects are fully supported by valid voxels.

    Each defect index has one integer and two half-integer components, so the
    defect sits at the center of a plaquette of four neighboring grid points.
    A defect is valid only if all four corner voxels are valid in ``mask``;
    touching even one invalid voxel marks the defect as invalid, because the
    winding number computed from undefined directors carries no physical
    meaning.

    Parameters
    ----------
    defect_indices : DefectIndex
        Array of shape (N_defects, 3) in lattice-index coordinates, with one
        integer and two half-integer components per row.

    mask : MaskField
        Boolean validity field of shape (Nx, Ny, Nz). True marks voxels whose
        director data is physically meaningful.

    is_boundary_periodic : DimensionInfo, optional
        Accepts a bool or a sequence of 3 bools.
        Whether to apply periodic boundary conditions in each dimension.
        Along periodic dimensions the plaquette corner indices wrap around;
        along non-periodic dimensions out-of-range corners raise an error.
        Default is 0 (no periodicity).

    Returns
    -------
    validity : np.ndarray, shape (N_defects,), dtype bool
        True for each defect whose four supporting voxels are all valid. The
        result preserves the input row order. Input arrays are not modified.
    """
    defect_indices = as_defect_index(defect_indices)
    mask = as_lattice_mask(mask, name="defect validity mask")
    is_boundary_periodic = as_dimension_info(
        is_boundary_periodic,
        name="is_boundary_periodic",
        is_bool=True,
    )

    lower = np.floor(defect_indices).astype(int)
    upper = np.ceil(defect_indices).astype(int)

    shape = np.array(mask.shape)
    for axis in range(3):
        if is_boundary_periodic[axis]:
            lower[:, axis] %= shape[axis]
            upper[:, axis] %= shape[axis]

    out_of_bounds_axes = [
        axis
        for axis in range(3)
        if np.any(lower[:, axis] < 0) or np.any(upper[:, axis] >= shape[axis])
    ]
    if out_of_bounds_axes:
        details = "; ".join(
            f"axis {axis}: corner range "
            f"[{lower[:, axis].min()}, {upper[:, axis].max()}], valid range "
            f"[0, {shape[axis] - 1}]"
            for axis in out_of_bounds_axes
        )
        raise ValueError(
            "Defect indices reach outside the mask along a non-periodic "
            f"dimension. Mask shape is {mask.shape}; {details}."
        )

    # The plaquette corners are every lower/upper combination per axis. The
    # integer component has lower == upper, so the 8 combinations collapse to
    # the 4 distinct corner voxels of the loop. Keeping the small repeated
    # reads avoids allocating a separate (N, 4, 3) corner-index array.
    validity = np.ones(len(defect_indices), dtype=bool)
    corners = (lower, upper)
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                validity &= mask[
                    corners[ix][:, 0],
                    corners[iy][:, 1],
                    corners[iz][:, 2],
                ]

    return validity


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
    is_simplify = as_bool(is_simplify, name="is_simplify")
    is_return_mask = as_bool(is_return_mask, name="is_return_mask")
    if not np.isscalar(threshold) or not np.isfinite(threshold):
        raise ValueError("`threshold` must be a finite scalar.")

    # ------------------------------------------------------------------ #
    # 1. Evaluate director field at surface vertices                       #
    # ------------------------------------------------------------------ #
    if isinstance(ndata, np.ndarray):
        # Surface is assumed to already be a clean triangulated mesh whose
        # vertex order matches the supplied director array.  Skipping
        # .clean() avoids vertex reordering that would break the mapping.
        surface = surface.triangulate()
        vertices = np.asarray(surface.points, dtype=float)  # (V, 3)
        n = as_director_field(
            ndata, name="ndata", is_normalized=True, is_zero_allowed=False
        )
        if n.shape != (len(vertices), 3):
            raise ValueError(
                f"`ndata` array has shape {n.shape} but surface has "
                f"{len(vertices)} vertices; expected shape ({len(vertices)}, 3)."
            )
    else:
        surface = surface.triangulate().clean()
        vertices = np.asarray(surface.points, dtype=float)  # (V, 3)
        Q = ndata.interpolate(vertices)
        n = q_diagonalize(Q).n  # (V, 3)

    logger.debug(f"Evaluated director at {len(vertices)} surface vertices.")

    # ------------------------------------------------------------------ #
    # 2. Build edge → [tri_idx, opposite_vertex] mapping                  #
    # ------------------------------------------------------------------ #
    faces = np.asarray(surface.faces, dtype=int).reshape(-1, 4)
    triangles = faces[:, 1:]  # (T, 3)

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
    # For each internal edge (i, j) with opposing vertices k and l, the
    # quadrilateral boundary is k -> i -> l -> j. This follows directly from
    # triangle connectivity and avoids fragile geometric angle sorting.
    quad_list = []
    quad_tri_pairs = []  # (tri_idx_0, tri_idx_1) per quad
    for (vi, vj), entries in edge_map.items():
        if len(entries) == 1:
            continue  # boundary edge
        if len(entries) != 2:
            raise ValueError(
                "`surface` must be edge-manifold; an edge is shared by "
                f"{len(entries)} triangles."
            )
        tri_idx_0, vk = entries[0]
        tri_idx_1, vl = entries[1]
        quad_list.append(np.array([vk, vi, vl, vj], dtype=int))
        quad_tri_pairs.append((tri_idx_0, tri_idx_1))

    if not quad_list:
        logger.debug("No internal edges found; returning empty result.")
        empty_coords = np.empty((0, 3), dtype=float)
        if is_return_mask:
            return empty_coords, np.zeros(len(vertices), dtype=bool)
        return empty_coords

    quads = np.array(quad_list, dtype=int)  # (Q, 4)
    logger.debug(f"Formed {len(quads)} quad loops from internal edges.")

    # ------------------------------------------------------------------ #
    # 4. Vectorised loop integrals                                         #
    # ------------------------------------------------------------------ #
    # stack shape: (4, Q, 3)
    directors = np.stack([n[quads[:, k]] for k in range(4)], axis=0).copy()
    aligned = align_director_stack(directors)

    dot_first_last = np.einsum("qi,qi->q", aligned[0], aligned[3])
    defect_mask = dot_first_last < threshold

    logger.debug(
        f"Detected {defect_mask.sum()} defect quads " f"(threshold={threshold})."
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

    defect_indices = as_defect_index(defect_indices)
    if isinstance(num_shell, bool) or not isinstance(num_shell, (int, np.integer)):
        raise TypeError("`num_shell` must be a positive integer.")
    if num_shell <= 0:
        raise ValueError("`num_shell` must be a positive integer.")
    if defect_indices.size == 0:
        return np.empty((0, 4 * num_shell**2, 3), dtype=int)

    square_size_list = np.arange(1, 2 * num_shell + 1, 2)
    square_num_list = square_size_list + 1

    square_origin_list = np.arange(-0.5, -num_shell - 0.5, -1)
    square_origin_list = np.broadcast_to(square_origin_list, (2, num_shell)).T
    square_origin_list = np.hstack([np.zeros((num_shell, 1)), square_origin_list])

    length = 4 * num_shell**2

    result = np.empty((len(defect_indices), length, 3), dtype=int)

    indexx = np.isclose(defect_indices[:, 0], np.round(defect_indices[:, 0]))
    indexy = np.isclose(defect_indices[:, 1], np.round(defect_indices[:, 1]))
    indexz = np.isclose(defect_indices[:, 2], np.round(defect_indices[:, 2]))

    defectx = defect_indices[indexx]
    defecty = defect_indices[indexy]
    defectz = defect_indices[indexz]

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

    result[indexx] = defectx.astype(int)
    result[indexy] = defecty.astype(int)
    result[indexz] = defectz.astype(int)

    return result
