import numpy as np
from typing import Optional, Tuple

from .smoothened_line import SmoothenedLine
from ..general import sort_line_indices  # , get_plane, get_tangent
from ..logging_decorator import logging_and_warning_decorator
from ..datatypes import (
    Vect3D,
    DefectIndex,
    DimensionPeriodicInput,
    as_dimension_info,
    boundary_periodic_size_to_flag,
)
from ..field import apply_linear_transform
from .visual_mayavi.plot_tube import PlotTube


class DisclinationLine:
    """
    Represent a single disclination line extracted from a 3D liquid crystal director field.

    This class handles the geometry and visualization of topological line defects.

    It supports:
    - Coordinate transformation and spatial offset for real-space analysis
    - Automated classification of line types (loop / cross / segment)
    - Optional smoothing using Savitzky-Golay filtering and spline interpolation
    - 3D Mayavi visualization, including support for periodic boundaries
    - Tological analysis (ongoing)

    Attributes
    ----------
    _defect_indices : DefectIndex, np.ndarray of shape (N_defects, 3)
        Grid-based defect point indices. Each point has one integer and two half-integer components.
        The geometrical meaning of these components is explained in the definition of `DefectIndex`
        in `datatype.py`.

    _defect_coords : np.ndarray of shape (N_defects, 3)
        Real-space coordinates of the defect points, after applying transformation and offset.

    _box_size_periodic : DimensionPeriodic, np.ndarray of shape (3,)
        Box size (in index units) for each dimension, indicating periodicity.
        - np.inf → non-periodic
        - int → periodic, with value as the boundary size

    _end2end_category : str
        Indicates the type of line:
        - 'loop': Closed loop (first == last)
        - 'cross': Cross-boundary loop (same after modulo) as the line crossing whole box
        - 'seg' : Open segment

    _defect_coords_smooth : np.ndarray, optional
        Smoothed version of the defect coordinates, if smoothing was applied.

    _figures : list of Mayavi figure handles
        3D visualization objects rendered using Mayavi.
    """

    def __init__(
        self,
        defect_indices: DefectIndex,
        box_size_periodic_index: DimensionPeriodicInput,
        is_sorted: bool = True,
        offset: Vect3D = np.array([0, 0, 0]),
        transform: np.ndarray = np.eye(3),
        name: Optional[str] = None,
    ):
        """
        Initialize a DisclinationLine object from a list of defect indices.

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

        is_sorted : bool, optional
            Whether the input defect indices are pre-sorted by nearest-neighbor order.
            If False, the constructor will reorder them using a greedy sorting algorithm.
            Default is True.

        offset : Vect3D, array_like of 3 floats, optional
            Global offset added to all coordinates after transformation.
            Useful for shifting lines in real space.
            Default is (0, 0, 0) (no shift).

        transform : np.ndarray of shape (3, 3), optional
            Linear transformation matrix applied to the defect indices
            to convert from grid space to physical space (e.g., for anisotropic grids).
            Default is np.eye(3) (identity transform).

        name : str, optional
            The name of this line.
        """
        if is_sorted == False:
            defect_indices = sort_line_indices(defect_indices)

        box_size_periodic_index = as_dimension_info(box_size_periodic_index)

        if np.linalg.norm(defect_indices[0] - defect_indices[-1]) == 0:
            self._end2end_category = "loop"
            self._defect_indices = defect_indices[:-1]
        else:
            defect1 = defect_indices[0].copy()
            defect2 = defect_indices[-1].copy()
            defect1 = np.where(
                box_size_periodic_index == np.inf,
                defect1,
                defect1 % box_size_periodic_index,
            )
            defect2 = np.where(
                box_size_periodic_index == np.inf,
                defect2,
                defect2 % box_size_periodic_index,
            )
            if np.linalg.norm(defect1 - defect2) == 0:
                self._end2end_category = "cross"
                self._defect_indices = defect_indices[:-1]
            else:
                self._end2end_category = "seg"
                self._defect_indices = defect_indices

        self._defect_num = np.shape(self._defect_indices)[0]
        self._box_size_periodic_index = box_size_periodic_index

        self.update_to_coord(grid_transform=transform, grid_offset=offset)

        if name == None:
            self._name = "line"
        else:
            self._name = name

    def update_to_coord(
        self,
        grid_transform: Optional[np.ndarray] = None,
        grid_offset: Optional[np.ndarray] = None,
    ):

        self._grid_transform = grid_transform
        self._grid_offset = grid_offset
        self._defect_coords = apply_linear_transform(
            self._defect_indices,
            transform=self._grid_transform,
            offset=self._grid_offset,
        )
        self._box_size_periodic_coord = apply_linear_transform(
            self._box_size_periodic_index,
            transform=self._grid_transform,
            offset=self._grid_offset,
        )

    def update_smoothen(
        self,
        window_ratio: Optional[int] = None,
        window_length: int = 21,
        order: int = 3,
        N_out_ratio: float = 3.0,
    ) -> np.ndarray:
        """
        Smoothen the defect line using Savitzky-Golay filtering and cubic spline interpolation.

        This method updates the smoothened version of the internal defect line coordinates,
        using either `window_ratio` or `window_length` to control the filter resolution.
        The smoothing mode is automatically selected based on whether the line is a loop.

        Parameters
        ----------
        window_ratio : int, optional
            Ratio used to compute the Savitzky-Golay filter window length.
            If provided, `window_length` is ignored.

        window_length : int, optional
            Directly specify the Savitzky-Golay filter window length (must be odd).
            Ignored if `window_ratio` is given. Default is 21.

        order : int, optional
            Order of the Savitzky-Golay filter polynomial. Default is 3.

        N_out_ratio : float, optional
            Ratio of output points to input points in the smoothened line.
            For example, 3.0 means 3× as many points as input. Default is 3.0.

        Returns
        -------
        smoothened_coords : np.ndarray of shape (N_out, M)
            The coordinates of the smoothened line.
            Also stored internally as `self._defect_coords_smooth`.
        """
        if self._end2end_category == "loop" or "cross":
            smoothen_mode = "wrap"
            tail_length = 0
            coords = self._defect_coords
        elif self._end2end_category == "cross":
            tail_length = 5
            indices_origin = self._defect_indices
            distance = indices_origin[-tail_length-1:-1] - indices_origin[:tail_length]
            tail = indices_origin[:tail_length,]
            
            for i in range(3):
                if np.isfinite(self._box_size_periodic_index[i]):
                    num_cross = np.round(distance[:,i] / self._box_size_periodic_index[i])
                    tail[:,i] += tail[:,i] + num_cross * self._box_size_periodic_coord[i]
            
            coords = self._defect_coords
            coords = np.concatenate([coords, tail])
        else:
            coords = self._defect_coords
            smoothen_mode = "interp"
            tail_length = 0
            
        output = SmoothenedLine(
            coords,
            window_ratio=window_ratio,
            window_length=window_length,
            order=order,
            N_out_ratio=N_out_ratio,
            mode=smoothen_mode,
        )

        self._defect_coords_smooth_obj = output
        self._defect_coords_smooth = output._output[:-tail_length-1]

        return output.output

    @logging_and_warning_decorator()
    def visualize(
        self,
        is_wrap: bool = False,
        is_smooth: bool = True,
        radius: float = 0.5,
        opacity: float = 1,
        color: Tuple[float, float, float] = (1, 1, 1),
        sides: int = 6,
        specular: float = 1,
        specular_color: Vect3D = (1.0, 1.0, 1.0),
        specular_power: float = 11,
        scalars: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        logger=None,
    ) -> None:
        """
        Visualize the defect line.

        Parameters
        ----------
        is_wrap : bool, optional
            Whether to apply periodic boundary wrapping to the defect line.
            Default is False.

        is_smooth : bool, optional
            Whether to use the smoothed version of the defect line.
            Default is True.

        radius : float, optional
            Radius of the 3D tube for visualization.
            Default is 1.

        opacity : float, optional
            Opacity of the tube. Range [0, 1].
            Default is 1.

        color : tuple of 3 floats, optional
            RGB color of the tube, values in [0, 1].
            Default is (1., 1., 1.), which is white.

        scalars : np.ndarray, optional
            Optional scalar values for each vertex.
            (enables gradient coloring). If provided, overrides 'color'.

        name : str, optional
            The name of this plotted line.
            If not provides, use the line's name is directly applied.
        """

        logger.debug(f"Start to visualize {self._name}")

        if is_smooth:
            if hasattr(self, "_defect_coords_smooth"):
                line_coords = self._defect_coords_smooth
            else:
                logger.warning(">>> The line has not been smoothened")
                logger.warning(">>> Use original data instead")
                line_coords = self._defect_coords
        else:
            line_coords = self._defect_coords

        if name == None:
            name = self._name

        line_coords_all = [line_coords]
        scalars_all = [scalars] if scalars != None else []

        if not is_wrap:
            line_plot = PlotTube(
                line_coords_all,
                color=color,
                radius=radius,
                opacity=opacity,
                sides=sides,
                specular=specular,
                specular_color=specular_color,
                specular_power=specular_power,
                scalars_all=scalars_all,
                name=name,
                logger=logger,
            )
        else:
            boundary_flag = boundary_periodic_size_to_flag(
                self._box_size_periodic_index
            )
            line_coords_origin = apply_linear_transform(
                line_coords,
                transform=np.linalg.inv(self._grid_transform),
                offset=-self._grid_offset,
            )

            line_coords_origin = np.where(
                boundary_flag,
                line_coords_origin % self._box_size_periodic_index,
                line_coords_origin,
            )
            diff = line_coords_origin[1:] - line_coords_origin[:-1]
            diff = np.linalg.norm(diff, axis=-1)
            end_list = np.where(diff > 1)[0] + 1
            end_list = np.concatenate([[0], end_list, [len(line_coords_origin)]])

            line_coords = apply_linear_transform(
                line_coords_origin,
                transform=self._grid_transform,
                offset=self._grid_offset,
            )

            coords_all = []
            scalars_all = []

            for i in range(len(end_list) - 1):
                coords_all.append(line_coords[end_list[i] : end_list[i + 1]])
                if scalars is not None:
                    scalars_all.append(scalars[end_list[i] : end_list[i + 1]])

            line_plot = PlotTube(
                coords_all,
                color=color,
                radius=radius,
                opacity=opacity,
                sides=sides,
                specular=specular,
                specular_color=specular_color,
                specular_power=specular_power,
                scalars_all=scalars_all,
                name=name,
                logger=logger,
            )

        self._line_plot = line_plot

        return line_plot

    # def update_norm(self):
    #     self._norm = get_plane(self._defect_coords)
    #     return self._norm

    # def update_center(self):
    #     self._center = np.average(self._defect_indices, axis=0)
    #     return self._center

    # def update_rotation(self, n, num_shell=1, method='plane'):
    #     self._Omega = defect_rotation(self._defect_indices, n,
    #                                   num_shell=num_shell, method=method, box_size_periodic=self._box_size_periodic)
    #     return self._Omega

    # def update_gamma(self, n=0, num_shell=1):

    #     if hasattr(self, '_Omega'):
    #         Omega = self._Omega
    #     else:
    #         Omega = self.update_rotation(n, num_shell=num_shell)

    #     if hasattr(self, '_norm'):
    #         norm = self._norm
    #     else:
    #         norm = self.update_norm()

    #     norm = np.broadcast_to(norm, (self._defect_num,3))
    #     self._gamma = np.arccos(np.abs(np.einsum('ia, ia -> i', norm, Omega))) / np.pi * 180

    #     return self._gamma

    # def update_geometry(self, is_smooth=True):

    #     if is_smooth:
    #         if hasattr(self, '_defect_coords_smooth'):
    #             if self._defect_coords_smooth_obj._N_out_ratio == 1:
    #                 points = self._defect_coords_smooth
    #             else:
    #                 print('There are more points in the smooth line')
    #                 print('Start to re-smooth it with N_out_ratio=1')
    #                 print(f'window_length={self._defect_coords_smooth_obj._window_length}')
    #                 print(f'order={self._defect_coords_smooth_obj._order}')
    #                 print(f'mode={self._defect_coords_smooth_obj._mode}')

    #                 points = SmoothenedLine(self._defect_coords,
    #                                         window_length=self._defect_coords_smooth_obj._window_length,
    #                                         order=self._defect_coords_smooth_obj._order,
    #                                         N_out_ratio=1,
    #                                         mode=self._defect_coords_smooth_obj._mode,
    #                                         is_keep_origin=False)._output
    #                 print('Done!')

    #         else:
    #             print('The line has not been smoothened')
    #             print('Use original data instead')
    #             points = self._defect_coords
    #     else:
    #         points = self._defect_coords

    #     is_periodic = self._end2end_category == 'loop'

    #     tangents = get_tangent(points, is_periodic=is_periodic, is_norm=False)
    #     tangents_size = np.linalg.norm(tangents, axis=1, keepdims=True)
    #     tangents = tangents / tangents_size

    #     dT_ds = get_tangent(tangents, is_periodic=is_periodic, is_norm=False)
    #     dT_ds_size = np.linalg.norm(dT_ds, axis=1, keepdims=False)
    #     curvatures = dT_ds_size / tangents_size[:,0]

    #     length = np.sum(tangents_size, axis=0)[0]

    #     self._tangent = tangents
    #     self._curvature = curvatures
    #     self._length = length

    # def update_beta(self, n=0):

    #     if hasattr(self, '_Omega'):
    #         Omega = self._Omega
    #     else:
    #         Omega = self.update_rotation(n)

    #     if not hasattr(self, '_tangent'):
    #         self.update_geometry()
    #     tangent = self._tangent

    #     self._beta = np.arccos(np.einsum('ia, ia -> i', tangent, Omega)) / np.pi * 180

    #     return self._beta
