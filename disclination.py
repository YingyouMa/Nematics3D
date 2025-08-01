import numpy as np
import time
from typing import Union, Sequence, Optional, List, Tuple

# ----------------------------------------------------------
# Functions which are being used and general.
# General means the code is for general nematics analysis.
# Not general means the code is specifically for my project.
# ----------------------------------------------------------

from .datatypes import (
    Vect3D,
    nField,
    DimensionPeriodicInput,
    DimensionFlagInput,
    as_dimension_info,
    DefectIndex,
    check_Sn
)
from .logging_decorator import logging_and_warning_decorator


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


def detect_defects_xyplane(n: np.ndarray, threshold: float) -> np.ndarray:
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
    
    n = check_Sn(n, 'n')
    
    from .field import align_directors
    a = n[:-1, :-1]
    b = align_directors(a, n[1:, :-1])
    c = align_directors(b, n[1:, 1:])
    d = align_directors(c, n[:-1, 1:])
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
    
    n_origin = check_Sn(n_origin, 'n')
    
    from .field import add_periodic_boundary
    
    is_boundary_periodic = as_dimension_info(is_boundary_periodic)
    planes = as_dimension_info(planes)
    
    logger.info("Start to defect defects")
    logger.debug(f"Periodic boundary flags: {is_boundary_periodic}")
    logger.debug(f"Planes selected for detection: {planes}")

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

        coords = detect_defects_xyplane(n_rot, threshold)

        # Restore original axis order
        inv_perm = np.argsort(perm)
        coords = coords[:, inv_perm]

        defect_indices = np.vstack((defect_indices, coords))
        logger.info(f"Finished axis {axis}-direction in {round(time.time() - now, 2)}s")
        now = time.time()

    # Wrap indices under periodic conditions
    for i, periodic in enumerate(is_boundary_periodic):
        if periodic:
            defect_indices[:, i] %= n_origin.shape[i]

    defect_indices, _ = np.unique(defect_indices, axis=0, return_index=True)
    return defect_indices


@logging_and_warning_decorator()
def defect_classify_into_lines(
    defect_indices: DefectIndex,
    box_size_periodic: DimensionFlagInput = np.inf,
    offset: Optional[Vect3D] = None,
    transform: Optional[np.ndarray] = None,
    logger=None
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

    offset : Vect3D, array_like of 3 floats, optional
        Global offset added to all coordinates after transformation.
        Useful for shifting lines in real space.
        Default is None (no shift).

    transform : np.ndarray of shape (3, 3), optional
        Linear transformation matrix applied to the defect indices
        to convert from grid space to physical space (e.g., for anisotropic grids).
        Default is None (identity transform).

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
    from .field import unwrap_trajectory
    from .general import make_hash_table, search_in_reservoir

    box_size_periodic = as_dimension_info(box_size_periodic)
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

    lines = [
        DisclinationLine(
            path, box_size_periodic, offset=offset, transform=transform
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

    from .field import generate_mirror_point_periodic_boundary

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

# @logging_and_warning_decorator()
# def draw_multiple_disclination_lines(
#     lines: List["DisclinationLine"],
#     is_new: bool = True,
#     fig_size: Tuple[int, int] = (1920, 1360),
#     bgcolor: Tuple[float, float, float] = (1.0, 1.0, 1.0),
#     is_wrap: bool = True,
#     is_smooth: bool = True,
#     color_input: Optional[Tuple[float, float, float]] = None,
#     tube_radius: float = 0.5,
#     tube_opacity: float = 1,
#     tube_specular: float = 1,
#     tube_specular_col: Tuple[float, float, float] = (1.0, 1.0, 1.0),
#     tube_specular_pow: float = 11,
#     outline_corners: Optional[np.ndarray] = None,
#     outline_radius: float = 3,
#     logger=None
# ):
    
#     from mayavi import mlab
#     from .general import blue_red_in_white_bg, sample_far
    
#     if color_input is None:
#         color_map = blue_red_in_white_bg()
#         color_map_length = np.shape(color_map)[0] - 1
#         lines_color = color_map[ (sample_far(len(lines))*color_map_length).astype(int)  ]
#     else:
#         lines_color = [color_input for line in lines_color]

#     if is_new:
#         mlab.figure(bgcolor=bgcolor, size=fig_size)

#     for i, line in enumerate(lines):
#         line.figure_init(tube_color=tuple(lines_color[i]), 
#                          is_new=False, 
#                          is_wrap=is_wrap,
#                          is_smooth=is_smooth,
#                          tube_opacity=tube_opacity, 
#                          tube_radius=tube_radius)
#         line.figure_update(tube_spec=tube_specular, 
#                            tube_spec_col=tube_specular_col, 
#                            tube_spec_pow=tube_specular_pow)
        
        
#     if outline_corners is not None:
#         try:
#             from .field import draw_box_from_corners
#             draw_box_from_corners(outline_corners)
#         except:
#             logger.exception("Corner error is caught")
#             logger.recovery("Discarded outline")
        
        
            
        
    
    




# def add_mid_points_disclination(line, is_loop=False):
#     #! defect_indices half integer
#     #! add one more point if the line is a loop
#     '''
#     Add mid-points into the disclination lines.

#     Parameters
#     ----------
#     line : array, (defect_num,3)
#            The array that includes all the indices of defects.
#            The defects must be sorted, as the neighboring defects have the minimum distance.
#            For each defect, one of the indices should be integer and the rest should be half-integer.
#            Usually defect_indices are generated by defect_defect() and smoothen_line() in this module.

#     is_loop : bool, optional
#               If this disclination line is a closed loop.
#               If so, this function will add one more point between the start and the end of this loop.
#               Default is False.

#     Returns
#     -------
#     line_new : array, ( 2*defect_num-1 , 3 ) or ( 2*defect_num , 3 ), for a crossing line or a loop
#                The new array that includes all the indices of defects, with mid-points added

#     Dependencies
#     ------------
#     - NumPy: 1.22.0

#     Called by
#     ---------
#     - Disclination_line
#     '''

#     if is_loop == True:
#         line = np.vstack([line, line[0]])

#     line_new = np.zeros((2*len(line)-1,3))
#     line_new[0::2] = line
#     defect_diff = line[1:] - line[:-1]
#     defect_diff_mid_value = np.sign(defect_diff[np.where( line[:-1]%1 == 0 )]) * 0.5
#     defect_diff_mid_orient = (line[:-1]%1 == 0).astype(int)
#     line_new[1::2] = line_new[0:-1:2] + np.array([defect_diff_mid_value]).T * defect_diff_mid_orient

#     if is_loop == True:
#         line = line[:-1]

#     return line_new

# def defect_vinicity_grid(defect_indices, num_shell=2):

#     square_size_list = np.arange(1, 2*num_shell+1, 2)
#     square_num_list  = square_size_list + 1

#     square_origin_list = np.arange(-0.5, -num_shell-0.5, -1)
#     square_origin_list = np.broadcast_to(square_origin_list, (2,num_shell)).T
#     square_origin_list = np.hstack([ np.zeros((num_shell, 1)), square_origin_list ])

#     length = 4 * num_shell**2

#     result = np.zeros( (np.shape(defect_indices)[0], length, 3) )

#     indexx = np.isclose(defect_indices[:, 0], np.round(defect_indices[:, 0]))
#     indexy = np.isclose(defect_indices[:, 1], np.round(defect_indices[:, 1]))
#     indexz = np.isclose(defect_indices[:, 2], np.round(defect_indices[:, 2]))

#     defectx = defect_indices[indexx]
#     defecty = defect_indices[indexy]
#     defectz = defect_indices[indexz]

#     squarex = get_square(square_size_list, square_num_list, origin_list=square_origin_list , dim=3)
#     squarey = squarex.copy()
#     squarey[:, [0, 1]] = squarey[:, [1, 0]]
#     squarez = squarex.copy()
#     squarez[:, [0, 1]] = squarez[:, [1, 0]]
#     squarez[:, [1, 2]] = squarez[:, [2, 1]]

#     defectx = np.repeat(defectx, length, axis=0).reshape(np.shape(defectx)[0],length,3)
#     defecty = np.repeat(defecty, length, axis=0).reshape(np.shape(defecty)[0],length,3)
#     defectz = np.repeat(defectz, length, axis=0).reshape(np.shape(defectz)[0],length,3)

#     defectx =  defectx + np.broadcast_to(squarex, (np.shape(defectx)[0], length,3))
#     defecty =  defecty + np.broadcast_to(squarey, (np.shape(defecty)[0], length,3))
#     defectz =  defectz + np.broadcast_to(squarez, (np.shape(defectz)[0], length,3))

#     result[indexx] = defectx
#     result[indexy] = defecty
#     result[indexz] = defectz

#     result = result.astype(int)

#     return result


# def defect_rotation(defect_indices, n,
#                     num_shell=1, box_size_periodic=[np.inf, np.inf, np.inf],
#                     method='cross'):

#     box_size_periodic = array_from_single_or_list(box_size_periodic)

#     vic_grid = defect_vinicity_grid(defect_indices, num_shell=num_shell)
#     vic_grid_wrap = np.where(box_size_periodic == np.inf, vic_grid, vic_grid%box_size_periodic)

#     vic_n = n[*tuple(vic_grid_wrap.T)].transpose((1,0,2))

#     if method == 'plane':
#         Omega = get_plane(vic_n)
#     elif method == 'cross':
#         Omega = get_rotation_axis(vic_n)
#     else: "method should be rather cross or plane"

#     return Omega


# def is_defects_connected(defect1, defect2, box_size_periodic=[np.inf, np.inf, np.inf]):
#     #! defect_indices half integer
#     '''
#     To examine if two defects are connected.
#     For any defect, one of the index must be integer, representing the layer,
#     and the other two indices must be half-integer, representing the center of one pixel in this layer.
#     The index is usually provided by defect_detect().
#     Supposing defecy1 = (layer, center1, center2), where layer is integer while center1 and center2 are half-integers,
#     the set of all the possible neighboring defects is
#     (layer+-1,     center1,        center2)
#     (layer+-0.5,   center1+-0.5,   center2)
#     (layer+-0.5,   center1,        center2+-0.5)
#     here +- means plusminus, and the order is unneccessary as (+,+), (-,+), (+,-), (-,-) are all possible
#     so there are 2+4+4=10 possible neighboring defects.
#     This function will examine if defect2 is one of the possible neighboring defects

#     Note that, if one of the box_size is np.inf (which means there is no periodic boundary condition),
#     then there should NOT be negative value in the correspoinding dimension in point, because it's meaningless.

#     Parameters
#     ----------
#     defect1 : array, (3,)
#               The indices of the first defect on the index grid (not coordinate of the real space)

#     defect2 : array, (3,)
#               The indices of the other defect on the index grid (not coordinate of the real space)

#     box_size_periodic : array of three floats, or one float, optional
#                         The number of indices in each dimension, x, y, z.
#                         If box_size is x, it will be interprepted as [x,x,x].
#                         If one of the boundary is not periodic, the corresponding value in box_size is np.inf.
#                         For example, if the box is periodic in x and y dimension, and the possible maximum index is X and Y,
#                         box_size should be [X+1, Y+1, np.inf].
#                         Default is [np.inf, np.inf, np.inf], which means the function only return the point itself.

#     Returns
#     -------
#     result : str
#              "same" means these two defects are the same.
#              "neighbor" means these two defects are connnected.
#              "far" means these two defects are not connnected.

#     Dependencies
#     ------------
#     - NumPy: 1.22.0
#     - .general.array_from_single_or_list()
#     - .field.find_mirror_point_boundary()

#     Called by
#     ---------
#     - class: DisclinationLine
#     '''

#     from .field import find_mirror_point_boundary

#     box_size_periodic = array_from_single_or_list(box_size_periodic)

#     is_boundary_periodic = box_size_periodic!=np.inf
#     defect1 = np.array(defect1)
#     defect2 = np.array(defect2)
#     defect1[is_boundary_periodic] = defect1[is_boundary_periodic] % box_size_periodic[is_boundary_periodic]
#     defect2[is_boundary_periodic] = defect2[is_boundary_periodic] % box_size_periodic[is_boundary_periodic]
#     defect_diff = np.abs(defect1 - defect2)
#     if np.linalg.norm(defect_diff) == 0:
#         return "same"

#     defect1_neighbor_possible = defect_neighbor_possible_get(defect1, box_size_periodic=box_size_periodic)
#     defect2 = find_mirror_point_boundary(defect2, box_size_periodic=box_size_periodic)
#     setA = set(map(tuple, defect1_neighbor_possible))
#     setB = set(map(tuple, defect2))

#     common_points = setA & setB

#     if len(common_points) > 0:
#         return "neighbor"
#     else:
#         return "far"


# def is_loop_new(lines, loop_indices,
#                 threshold=4, box_size_periodic=[np.inf, np.inf, np.inf]):

#     from scipy.spatial.distance import cdist
#     from .field import unwrap_trajectory

#     if len(lines) == 0:
#         return "new", -1

#     box_size_periodic = array_from_single_or_list(box_size_periodic)
#     loop_indices = np.where(box_size_periodic == np.inf, loop_indices, loop_indices % box_size_periodic)


#     for i,line in enumerate(lines): # line: one of the old loops. loop: the new loop to be checked.
#         line_indices = line._defect_indices[:-1]
#         line_indices = np.where(box_size_periodic == np.inf, line_indices, line_indices % box_size_periodic)
#         dist = cdist(loop_indices, line_indices)
#         if np.min(dist) <= threshold:
#             loop_start_index, line_start_index = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
#             loop_indices_unwrap = np.concatenate( [ loop_indices[loop_start_index:], loop_indices[:loop_start_index] ] )
#             line_indices_unwrap = np.concatenate( [ line_indices[line_start_index:], line_indices[:line_start_index] ] )
#             loop_indices_unwrap = unwrap_trajectory(loop_indices_unwrap, box_size_periodic=box_size_periodic)
#             line_indices_unwrap = unwrap_trajectory(line_indices_unwrap, box_size_periodic=box_size_periodic)
#             dist_unwrap = cdist(loop_indices_unwrap, line_indices_unwrap)
#             dist_unwrap = np.min(dist_unwrap, axis=1) # for each defect in loop, find the closest distance between this defect and the line
#             if np.max(dist_unwrap) > threshold:
#                 return "mix", i
#             else:
#                 return "old", i

#     return "new", -1


# # -----------------------------------------------------
# # Specific functions which are being used in my project
# # -----------------------------------------------------


# @time_record
# def example_visualize_defects(lines, is_wrap=True, min_length=50, window_length=61,
#                               opacity=1, radius=0.5, color_input=None,
#                               specular=1, specular_col=(1,1,1), specular_pow=11,
#                               outline_extent=None):

#     '''
#     Visualize a set of disclination lines using Mayavi 3D rendering.

#     This function filters, smooths, and visualizes disclination lines in 3D nematics,
#     with customizable visual properties including color, opacity, radius, and lighting effects.
#     Optionally, small loops could be excluded, and an outline box can be displayed around.


#     Parameters
#     ----------
#     lines : list
#             List of `DisclinationLine` objects to visualize.

#     is_wrap : bool, optional
#               Whether the line should be wrapped with periodic boudanry conditions.
#               Default is `True`.

#     min_length : int, optional
#                  Minimum number of defects required for a line to be visualized.
#                  Lines shorter than this threshold are discarded.
#                  Default is 50.

#     window_length : int, optional
#                     Window size for smoothing the defect trajectory before rendering.
#                     Default is 61.

#     opacity : float, optional
#               Opacity of the tube representing each line.
#               Ranges from 0 (transparent) to 1 (opaque).
#               Default is 1.

#     radius : float, optional
#              Radius of the tube used to render each line.
#              Default is 0.5.

#     color_input : tuple of three ints or None, optional
#                   A single RGB tuple to color all lines, or `None` to use the default colormap.
#                   The default colormap based on 'blue-red',
#                   with special designs trying to distinguish each line visually.
#                   Default is `None`.

#     specular : float, optional
#                Specular lighting intensity for visual effects.
#                Default is 1.

#     specular_col : tuple of 3 floats, optional
#                    RGB values for the color of specular highlights.
#                    Default is white (1, 1, 1).

#     specular_pow : float, optional
#                    Controls the sharpness of specular highlights.
#                    Higher values result in smaller, sharper highlights. Default is 11.

#     outline_extent : list of 6 floats, optional
#                      Extent of the outline box in the format [xmin, xmax, ymin, ymax, zmin, zmax].
#                      If set to `None`, no outline is drawn.
#                      Default is `None`.

#     Returns
#     -------
#     None
#         This function produces a Mayavi 3D visualization and does not return any value.

#     Dependencies
#     ------------
#     - NumPy: 1.26.4
#     - Mayavi: 4.8.2
#     '''

#     from mayavi import mlab

#     lines = [line for line in lines if line._defect_num>min_length]
#     lines = sorted(lines,
#                    key=lambda line: line._defect_num,
#                    reverse=True)

#     if color_input is None:
#         color_map = blue_red_in_white_bg()
#         color_map_length = np.shape(color_map)[0] - 1
#         lines_color = color_map[ (sample_far(len(lines))*color_map_length).astype(int)  ]
#     else:
#         lines_color = [color_input for line in lines_color]

#     for i, line in enumerate(lines):
#         if window_length != 0:
#             line.update_smoothen(window_length=window_length)
#         line.figure_init(tube_color=tuple(lines_color[i]), is_new=1-bool(i), is_wrap=is_wrap,
#                          tube_opacity=opacity, tube_radius=radius)
#         line.figure_update(tube_spec=specular, tube_spec_col=specular_col, tube_spec_pow=specular_pow)

#     if outline_extent is not None:
#         figure = mlab.gcf()
#         mlab.outline(figure=figure, color=(0,0,0), extent=outline_extent, line_width=4)
#         mlab.view(distance=450)

# @time_record
# def example_visualize_defects_loops_init(lines, is_wrap=True, min_length=30, window_length=61,
#                                          opacity=1, radius=1,
#                                          outline_extent=[0,382,0,382,0,382]):

#     from mayavi import mlab

#     lines = [line for line in lines if line._defect_num>min_length]
#     lines = sorted(lines,
#                    key=lambda line: line._defect_num,
#                    reverse=True)

#     color_map = blue_red_in_white_bg()
#     color_map_length = np.shape(color_map)[0] - 1

#     for i, line in enumerate(lines):
#         line.update_smoothen(window_length=window_length)
#         line.update_norm()
#         tube_color = tuple(color_map[int(np.abs(line._norm[0])*color_map_length)])
#         line.figure_init(tube_color=tube_color,
#                          is_new=1-bool(i), is_wrap=is_wrap,
#                          tube_opacity=opacity, tube_radius=radius)

#     figure = mlab.gcf()
#     mlab.outline(figure=figure, color=(0,0,0), extent=outline_extent, line_width=4)
#     mlab.view(azimuth=90, elevation=90, distance=950, roll=90)


# @time_record
# def example_visualize_defects_loop_lack(n, is_wrap=True,
#                               min_length=40, is_boundary_periodic=(1,1,1),
#                               cross_window_length=31, loop_window_length=31):

#     defect_indices = defect_detect(n, is_boundary_periodic=is_boundary_periodic)
#     lines = defect_classify_into_lines(defect_indices, np.shape(n)[:3])
#     lines = [line for line in lines if line._defect_num>min_length]
#     loops = [line for line in lines if line._end2end_category=='loop']
#     crosses = [line for line in lines if line._end2end_category=='cross']
#     crosses = sorted(crosses,
#                      key=lambda line: line._defect_num,
#                      reverse=True)
#     color_map = blue_red_in_white_bg()
#     color_map_length = np.shape(color_map)[0] - 1
#     crosses_color = color_map[ (sample_far(len(crosses))*color_map_length).astype(int)  ]

#     for i, cross in enumerate(crosses):
#         cross.update_smoothen(window_length=cross_window_length)
#         cross.figure_init(tube_color=tuple(crosses_color[i]), is_new=False, is_wrap=is_wrap)

#     for i, loop in enumerate(loops):
#         loop.update_smoothen(window_length=loop_window_length)
#         loop.figure_init(tube_color=(0,0,0), is_new=False, is_wrap=is_wrap)


# def plot_n_on_Pplane(n_box, height,
#                      color_axis=0, height_visual=0,
#                      space=3, line_width=2, line_density=1.5,
#                      if_cb=True, colormap='blue-red'):

#     #! warning: L is in the first axis

#     from .defect2D import get_streamlines
#     from mayavi import mlab


#     if color_axis == 0:
#         print('color_axis is not input')
#         print('use the default value: (1,0)')
#         color_axis = (1,0)


#     # select the 2D axes to color the directors
#     color_axis1 = color_axis / np.linalg.norm(color_axis)
#     color_axis2 = np.cross( np.array([0,0,1]), np.concatenate( [color_axis1,[0]] ) )
#     color_axis2 = color_axis2[:-1]


#     # the grid indices
#     x = np.arange(np.shape(n_box)[0])
#     y = np.arange(np.shape(n_box)[1])
#     z = np.arange(np.shape(n_box)[2])


#     # select the indices of directors to be plot
#     indexy = np.arange(0, np.shape(n_box)[1], space)
#     indexz = np.arange(0, np.shape(n_box)[2], space)
#     iny, inz = np.meshgrid(indexy, indexz, indexing='ij')
#     ind = (iny, inz)


#     # project the directors on the 2D N-M plane
#     n_plot = n_box[height]
#     n_plane = np.array( [n_plot[:,:,1][ind], n_plot[:,:,2][ind] ] )
#     n_plane = n_plane / np.linalg.norm( n_plane, axis=-1, keepdims=True)


#     # extract the streamlines of directors on the 2D N-M plane
#     stl = get_streamlines(
#                 y[indexy], z[indexz],
#                 n_plane[0].transpose(), n_plane[1].transpose(),
#                 density=line_density)
#     stl = np.array(stl)


#     # Prepare the lines to be plotted by mayavi
#     # This selects the pairs of points which are connected in the plot
#     # In other words, the neighboring points within the same streamline are connected to form a unit segment
#     connect_begin = np.where(np.abs( stl[1:,0] - stl[:-1,1]  ).sum(axis=-1) < 1e-5 )[0]
#     connections = np.zeros((len(connect_begin),2))
#     connections[:,0] = connect_begin
#     connections[:,1] = connect_begin + 1

#     lines_index = np.arange(np.shape(stl)[0])
#     disconnect = lines_index[~np.isin(lines_index, connect_begin)]


#     # the coordinates of points to be plotted
#     if height_visual == 0:
#         src_x = stl[:, 0, 0] * 0 + height
#     else:
#         src_x = stl[:, 0, 0] * 0 + height_visual
#     src_y = stl[:, 0, 0]
#     src_z = stl[:, 0, 1]


#     # To derive the colors for the streamline, express each unit segment in the color-axes
#     unit = stl[1:, 0] - stl[:-1, 0]
#     unit = unit / np.linalg.norm(unit, axis=-1, keepdims=True)
#     coe1 = np.einsum('ij, j -> i', unit, color_axis1)
#     coe2 = np.einsum('ij, j -> i', unit, color_axis2)
#     coe1 = np.concatenate([coe1, [coe1[-1]]])
#     coe2 = np.concatenate([coe2, [coe2[-1]]])
#     colors = np.arctan2(coe1,coe2)
#     nan_index = np.array(np.where(np.isnan(colors)==1))
#     colors[nan_index] = colors[nan_index-1]
#     colors[disconnect] = colors[disconnect-1]

#     # initialize the figure
#     src = mlab.pipeline.scalar_scatter(src_x, src_y, src_z, colors)
#     src.mlab_source.dataset.lines = connections
#     src.update()

#     lines = mlab.pipeline.stripper(src)
#     plot_lines = mlab.pipeline.surface(lines, line_width=line_width, colormap='blue-red')

#     # apply the input colormap
#     if type(colormap) == np.ndarray:
#         lut = plot_lines.module_manager.scalar_lut_manager.lut.table.to_array()
#         lut[:, :3] = colormap
#         plot_lines.module_manager.scalar_lut_manager.lut.table = lut

#     if if_cb == True:
#         cb = mlab.colorbar(object=plot_lines, orientation='vertical', nb_labels=5, label_fmt='%.2f')
#         cb.data_range = (0,1)
#         cb.label_text_property.color = (0,0,0)


# def show_loop_plane_2Ddirector(n_box, height_list,
#                                height_visual_list=0, plane_list=(1,0,1),
#                                smooth_window_ratio=3, smooth_order=3, smooth_N_out_ratio=5,
#                                tube_radius=0.5, tube_opacity=0.5, tube_color=(0.5,0.5,0.5),
#                                line_width=2, line_density=1.5,
#                                tube_specular=1, tube_specular_col=(1,1,1), tube_specular_pow=75,
#                                fig_size=(1920, 1360), bgcolor=(1,1,1), camera_set=0,
#                                if_cb=True, n_colormap='blue-red'):

#     #! warning: L is in the first axis

#     from mayavi import mlab

#     # define the interpolate function by parabola
#     if height_visual_list == 0:
#         height_visual_list = height_list
#         def parabola(x):
#             return x
#     else:
#         x, y, z = height_list
#         coe_matrix = np.array([
#                         [x**2, y**2, z**2],
#                         [x, y, z],
#                         [1,1,1]
#                         ])
#         del x, y, z
#         coe_parabola = np.dot(height_visual_list, np.linalg.inv(coe_matrix))
#         def parabola(x):
#             return coe_parabola[0]*x**2 + coe_parabola[1]*x + coe_parabola[2]


#     # For each N-M plane,
#     # project the directors on this 2D plane,
#     # and then plot them as streamlines
#     mlab.figure(size=fig_size, bgcolor=bgcolor)
#     for i, if_plane in enumerate(plane_list):
#         if if_plane:
#             plot_n_on_Pplane(n_box, height_list[i],
#                              height_visual=height_visual_list[i],
#                              line_width=line_width, line_density=line_density,
#                              if_cb=if_cb, colormap=n_colormap)


#     # identify the disclination loop from the input director field, and then visualize it
#     loop_indices = defect_detect(n_box)
#     if len(loop_indices) > 0:
#         loops = defect_classify_into_lines(loop_indices)
#         # !if len(loops) > 1:
#         loop = loops[0]
#         loop._defect_coord[:, 0] = parabola(loop._defect_coord[:, 0])
#         loop.update_smoothen(window_ratio=smooth_window_ratio,
#                             order=smooth_order,
#                             N_out_ratio=smooth_N_out_ratio)
#         loop.figure_init(tube_radius=tube_radius, tube_opacity=tube_opacity, tube_color=tube_color,
#                         is_new=False)
#         loop.figure_update(tube_spec=tube_specular, tube_spec_col=tube_specular_col, tube_spec_pow=tube_specular_pow)


#     # For each N-M plane,
#     # project the directors on this 2D plane,
#     # and then plot them as streamlines
#     for i, if_plane in enumerate(plane_list):
#         if if_plane:
#             plot_n_on_Pplane(n_box, height_list[i],
#                              height_visual=height_visual_list[i],
#                              line_width=line_width, line_density=line_density,
#                              if_cb=if_cb, colormap=n_colormap)


#     # change the camera
#     if camera_set != 0:
#         mlab.view(*camera_set[:3], roll=camera_set[3])
