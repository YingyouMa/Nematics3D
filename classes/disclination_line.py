import numpy as np
from typing import Optional, Tuple, Dict, Callable, Any

from .smoothened_line import SmoothenedLine
from ..general import sort_line_indices  # , get_plane, get_tangent
from ..logging_decorator import logging_and_warning_decorator
from ..datatypes import Vect3D, DefectIndex, DimensionPeriodicInput, as_dimension_info
from ..field import apply_linear_transform


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
    
    _defect_coord : np.ndarray of shape (N_defects, 3)
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

    _defect_coord_smooth : np.ndarray, optional
        Smoothed version of the defect coordinates, if smoothing was applied.

    _figures : list of Mayavi figure handles
        3D visualization objects rendered using Mayavi.
    """

    def __init__(
        self,
        defect_indices: DefectIndex,
        box_size_periodic: DimensionPeriodicInput,
        is_sorted: bool = True,
        offset: Optional[Vect3D] = None,
        transform: Optional[np.ndarray] = None,
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
            Default is None (no shift).

        transform : np.ndarray of shape (3, 3), optional
            Linear transformation matrix applied to the defect indices
            to convert from grid space to physical space (e.g., for anisotropic grids).
            Default is None (identity transform).
        """
        if is_sorted == False:
            defect_indices = sort_line_indices(defect_indices)

        box_size_periodic = as_dimension_info(box_size_periodic)

        if np.linalg.norm(defect_indices[0] - defect_indices[-1]) == 0:
            self._end2end_category = "loop"
            self._defect_indices = defect_indices[:-1]
        else:
            defect1 = defect_indices[0].copy()
            defect2 = defect_indices[-1].copy()
            defect1 = np.where(
                box_size_periodic == np.inf, defect1, defect1 % box_size_periodic
            )
            defect2 = np.where(
                box_size_periodic == np.inf, defect2, defect2 % box_size_periodic
            )
            if np.linalg.norm(defect1 - defect2) == 0:
                self._end2end_category = "cross"
                self._defect_indices = defect_indices[:-1]
            else:
                self._end2end_category = "seg"
                self._defect_indices = defect_indices

        self._defect_num = np.shape(self._defect_indices)[0]
        self._box_size_periodic = box_size_periodic

        self.update_to_coord(grid_transform=transform, grid_offset=offset)


    def update_to_coord(
        self,
        grid_transform: Optional[np.ndarray] = None,
        grid_offset: Optional[np.ndarray] = None,
    ):

        self._grid_transform = grid_transform
        self._grid_offset = grid_offset
        self._defect_coord = apply_linear_transform(
            self._defect_indices,
            transform=self._grid_transform,
            offset=self._grid_offset,
        )
        self._box_size_periodic_coord = apply_linear_transform(
            self._box_size_periodic,
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
            Also stored internally as `self._defect_coord_smooth`.
        """
        if self._end2end_category == "loop":
            smoothen_mode = "wrap"
        else:
            smoothen_mode = "interp"

        output = SmoothenedLine(
            self._defect_coord,
            window_ratio=window_ratio,
            window_length=window_length,
            order=order,
            N_out_ratio=N_out_ratio,
            mode=smoothen_mode,
            is_keep_origin=False,
        )

        self._defect_coord_smooth_obj = output
        self._defect_coord_smooth = output._output

        return output.output

    def figure_init(
        self,
        is_wrap: bool = False,
        is_smooth: bool = True,
        tube_radius: float = 0.5,
        tube_opacity: float = 0.5,
        tube_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        tube_sides: int = 6,
        is_new: bool = True,
        bgcolor: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        fig_size: Tuple[int, int] = (1920, 1360),
    ) -> None:
        """
        Initialize a Mayavi 3D figure to visualize the defect line.

        Parameters
        ----------
        is_wrap : bool, optional
            Whether to apply periodic boundary wrapping to the defect line.
            Default is False.

        is_smooth : bool, optional
            Whether to use the smoothed version of the defect line.
            Default is True.

        tube_radius : float, optional
            Radius of the 3D tube for visualization.
            Default is 0.5.

        tube_opacity : float, optional
            Opacity of the tube. Range [0, 1].
            Default is 0.5.

        tube_color : tuple of 3 floats, optional
            RGB color of the tube, values in [0, 1].
            Default is (0.5, 0.5, 0.5), which is medium gray.

        tube_sides : int, optional
            Number of polygonal sides used to draw the tube.
            Default is 6.

        is_new : bool, optional
            Whether to create a new Mayavi figure window.
            Default is True.

        bgcolor : tuple of 3 floats, optional
            RGB color of the figure background.
            Default is (1.0, 1.0, 1.0), which is white.

        fig_size : tuple of 2 ints, optional
            Size of the figure window in pixels.
            Default is (1920, 1360).
        """
        from mayavi import mlab

        if is_smooth:
            if hasattr(self, "_defect_coord_smooth"):
                line_coord = self._defect_coord_smooth
            else:
                print("the line has not been smoothened")
                print("use original data instead")
                line_coord = self._defect_coord
        else:
            line_coord = self._defect_coord

        if is_new:
            mlab.figure(bgcolor=bgcolor, size=fig_size)

        if not is_wrap:
            figure = mlab.plot3d(
                *(line_coord.T),
                tube_radius=tube_radius,
                opacity=tube_opacity,
                color=tube_color,
                tube_sides=tube_sides
            )
            figures = [figure]
        else:

            line_coord = np.where(
                self._box_size_periodic_coord == np.inf,
                line_coord,
                line_coord % self._box_size_periodic_coord,
            )
            diff = line_coord[1:] - line_coord[:-1]
            diff = np.linalg.norm(diff, axis=-1)
            end_list = np.where(diff > 1)[0] + 1
            end_list = np.concatenate([[0], end_list, [len(line_coord)]])

            figures = []
            for i in range(len(end_list) - 1):
                points = line_coord[end_list[i] : end_list[i + 1]]
                figure = mlab.plot3d(
                    *(points.T),
                    tube_radius=tube_radius,
                    opacity=tube_opacity,
                    color=tube_color,
                    tube_sides=tube_sides
                )
                figures.append(figure)

        if not is_new:
            figure.parent.parent.parent.parent.parent.scene.background = bgcolor

        self._figures = figures

    def generate_dict_figure_simplify(
        self, idx: int
    ) -> Dict[str, Tuple[Callable[[], Any], str]]:
        """
        Generate a dictionary for accessing and updating tube visualization parameters.

        Parameters
        ----------
        idx : int
            Index of the figure object in the self._figures list.

        Returns
        -------
        dict_figure_simplify : dict
            Dictionary mapping visual attributes (e.g., 'tube_radius') to
            (object_accessor_function, attribute_name) pairs.
        """
        dict_figure_simplify = {
            "bgcolor": (
                lambda: self._figures[idx].parent.parent.parent.parent.parent.scene,
                "background",
            ),
            "tube_radius": (lambda: self._figures[idx].parent.parent.filter, "radius"),
            "tube_opacity": (lambda: self._figures[idx].actor.property, "opacity"),
            "tube_sides": (
                lambda: self._figures[idx].parent.parent.filter,
                "number_of_sides",
            ),
            "tube_color": (lambda: self._figures[idx].actor.property, "color"),
            "tube_spec": (lambda: self._figures[idx].actor.property, "specular"),
            "tube_spec_col": (
                lambda: self._figures[idx].actor.property,
                "specular_color",
            ),
            "tube_spec_pow": (
                lambda: self._figures[idx].actor.property,
                "specular_power",
            ),
        }
        return dict_figure_simplify

    def figure_check_parameter(self, *args):
        """
        Print the current values of specified visual attributes in the figure.

        Parameters
        ----------
        *args : str
            Names of visual parameters to inspect (e.g., 'tube_opacity', 'bgcolor').

        verbose : bool, optional
            Whether to print detailed information. Default is True.
        """
        for arg in args:
            dict_figure_simplify = self.generate_generate_dict_figure_simplify(0)
            temp = dict_figure_simplify.get(arg)
            obj = temp[0]()
            attr = temp[1]
            print(arg + " : " + str(getattr(obj, attr)))

    def figure_update(self, **kwargs):
        """
        Update specified visual attributes of the rendered tube(s).

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments of the form {attribute_name: new_value}, where
            attribute_name can be one of:
            - 'bgcolor'
            - 'tube_radius'
            - 'tube_opacity'
            - 'tube_color'
            - 'tube_sides'
            - 'tube_spec'
            - 'tube_spec_col'
            - 'tube_spec_pow'
        """
        for attr in kwargs.keys():
            for idx in range(len(self._figures)):
                dict_figure_simplify = self.generate_dict_figure_simplify(idx)
                obj = dict_figure_simplify.get(attr)[0]()
                attr_final = dict_figure_simplify.get(attr)[1]
                setattr(obj, attr_final, kwargs.get(attr))

    # def update_norm(self):
    #     self._norm = get_plane(self._defect_coord)
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
    #         if hasattr(self, '_defect_coord_smooth'):
    #             if self._defect_coord_smooth_obj._N_out_ratio == 1:
    #                 points = self._defect_coord_smooth
    #             else:
    #                 print('There are more points in the smooth line')
    #                 print('Start to re-smooth it with N_out_ratio=1')
    #                 print(f'window_length={self._defect_coord_smooth_obj._window_length}')
    #                 print(f'order={self._defect_coord_smooth_obj._order}')
    #                 print(f'mode={self._defect_coord_smooth_obj._mode}')

    #                 points = SmoothenedLine(self._defect_coord,
    #                                         window_length=self._defect_coord_smooth_obj._window_length,
    #                                         order=self._defect_coord_smooth_obj._order,
    #                                         N_out_ratio=1,
    #                                         mode=self._defect_coord_smooth_obj._mode,
    #                                         is_keep_origin=False)._output
    #                 print('Done!')

    #         else:
    #             print('The line has not been smoothened')
    #             print('Use original data instead')
    #             points = self._defect_coord
    #     else:
    #         points = self._defect_coord

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
