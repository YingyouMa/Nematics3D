import numpy as np
import time
from typing import Tuple, Optional, List, Union, Callable, Literal
from dataclasses import replace

from ..logging_decorator import logging_and_warning_decorator, Logger
from ..datatypes import (
    Vect,
    as_Vect,
    Tensor,
    as_Tensor,
    QField5,
    QField9,
    as_QField5,
    SField,
    nField,
    Number,
    DimensionFlagInput,
    as_dimension_info,
    check_Sn,
    check_bool_flags,
)
from ..field import (
    Q_diagonalize,
    getQ,
    generate_coordinate_grid,
    apply_linear_transform,
    n_color_immerse,
)
from ..disclination import defect_detect, defect_classify_into_lines
from .Interpolator import Interpolator
from .visual_mayavi.plot_n_plane import PlotnPlane
from .visual_mayavi.plot_scene import PlotScene
from .visual_mayavi.plot_extent import PlotExtent
from .opts import (
    OptsExtent,
    OptsPlaneGrid,
    OptsnPlane,
    OptsScene,
    OptsSmoothen,
    OptsTube,
    merge_opts,
)
from ..general import get_box_corners


class QFieldObject:

    DEFAULT_SMOOTH_WINDOW_LENGTH = 61
    DEFAULT_MINIMUM_LINE_LENGTH = 75

    @logging_and_warning_decorator()
    def __init__(
        self,
        Q: Union[QField5, QField9] = None,
        S: SField = None,
        n: nField = None,
        box_periodic_flag: DimensionFlagInput = False,
        grid_offset: Vect(3) = np.array([0, 0, 0]),
        grid_transform: Tensor((3, 3)) = np.eye(3),
        is_diag: bool = True,
        logger: Logger = None,
    ) -> None:

        self._grid_offset = as_Vect(grid_offset, name="grid_offset")
        self._grid_transform = as_Tensor(grid_transform, (3, 3), name="grid_transform")

        start = time.time()
        logger.debug("Start to initialize Q field")
        if n is not None:
            n = check_Sn(n, "n")
            self._n = n
            logger.debug("Initialize Q field with S and n")
            if S is not None:
                S = check_Sn(S, "S")
                self._S = S
            else:
                logger.warning("No S input. Set to 1 everywhere.")
                self._S = np.zeros(np.shape(n)[:-1]) + 1.0
            if Q is not None:
                logger.warning("Both Q and n are provided. Q will be IGNORED.")
            self._Q = as_QField5(getQ(n, S=S))
            logger.debug(f"Q field initialized in {time.time() - start:.2f} seconds.")
        else:
            if Q is not None:
                logger.debug("Initialize Q field with Q directly")
                self._Q = as_QField5(Q)
                logger.debug(
                    f"Q field initialized in {time.time() - start:.2f} seconds."
                )
                if is_diag:
                    self._S, self._n = Q_diagonalize(self._Q, logger=logger)
            else:
                raise NameError("No data is input")

        self._box_periodic_flag = as_dimension_info(box_periodic_flag)
        self._box_size_periodic = np.zeros(3)
        for i, flag in enumerate(self._box_periodic_flag):
            if flag:
                self._box_size_periodic[i] = np.shape(self._Q)[i]
            else:
                self._box_size_periodic[i] = np.inf

        logger.debug("Start to transform lattice grid into real space")
        grid_shape = np.shape(self._Q)[:3]
        self._grid_origin, _, _ = generate_coordinate_grid(grid_shape, grid_shape)
        self.update_grid(grid_transform=grid_transform, grid_offset=grid_offset)

        self.figures = []

        Lx, Ly, Lz = np.shape(self._Q)[:3] - np.array([1, 1, 1])
        corners_index = get_box_corners(Lx, Ly, Lz)
        corners = apply_linear_transform(
            corners_index, transform=self._grid_transform, offset=self._grid_offset
        )

        self._corners_index = corners_index
        self._corners = corners

    @logging_and_warning_decorator()
    def update_diag(self, logger=None):
        self._S, self._n = Q_diagonalize(self._Q, logger=logger)

    @logging_and_warning_decorator()
    def update_defects(self, threshold=0, logger=None):

        self._defect_indices = defect_detect(
            self._n,
            threshold=threshold,
            is_boundary_periodic=self._box_periodic_flag,
            logger=logger,
        )

        self._defect_grid = apply_linear_transform(
            self._defect_indices,
            transform=self._grid_transform,
            offset=self._grid_offset,
        )

    def update_grid(
        self,
        grid_offset: Vect(3) = np.array([0, 0, 0]),
        grid_transform: Tensor((3, 3)) = np.eye(3),
        logger=None,
    ):
        """
        Generate the coordinates grid in the real space from the lattice indices through linear transform.
        See the document of apply_linear_transform()
        """
        self._grid_offset = as_Vect(grid_offset, name="grid_offset")
        self._grid_transform = as_Tensor(grid_transform, (3, 3), name="grid_transform")
        self._grid = apply_linear_transform(
            self._grid_origin, transform=self._grid_transform, offset=self._grid_offset
        )

        if hasattr(self, "_defect_indices"):
            self._defect_grid = apply_linear_transform(
                self._defect_indices,
                transform=self._grid_transform,
                offset=self._grid_offset,
            )

    @logging_and_warning_decorator()
    def update_lines_classify(self, logger=None):
        self._lines = defect_classify_into_lines(
            self._defect_indices,
            box_size_periodic=self._box_size_periodic,
            grid_offset=self._grid_offset,
            grid_transform=self._grid_transform,
            logger=logger,
        )
        self._lines = sorted(
            self._lines, key=lambda line: line._defect_num, reverse=True
        )
        for i, line in enumerate(self._lines):
            line._name = f"line{i}"

        return self._lines

    @logging_and_warning_decorator()
    def update_lines_smoothen(
        self,
        opts=OptsSmoothen(),
        logger=None,
        **kwargs,
    ):

        opts = merge_opts(opts, kwargs, prefix="smoothen_")

        if opts.window_length is not None:
            logger.warning(
                f">>> Window_length is manual input as {opts.window_length}. window_ratio would be ignored."
            )

        for line in self._lines:
            if line._defect_num >= opts.min_line_length:
                logger.debug(f"Start to smoothen {line._name}")
                line.update_smoothen(opts=opts)

    @logging_and_warning_decorator()
    def update_interpolator(self, logger=None):

        from scipy.interpolate import RegularGridInterpolator

        shape = np.shape(self._Q)[:3]
        u = np.arange(shape[0])
        v = np.arange(shape[1])
        w = np.arange(shape[2])

        interpolator = RegularGridInterpolator(
            (u, v, w), self._Q, method="linear", bounds_error=True
        )
        interpolator = Interpolator(
            interpolator,
            np.array([v[-1], u[-1], w[-1]]),
            grid_transform=self._grid_transform,
            grid_offset=self._grid_offset,
        )

        self._interpolator = interpolator

        return self._interpolator

    def inperpolate(self, points: np.ndarray, is_index=False):
        if not hasattr(self, "_interpolator"):
            self.update_interpolator()
        return self._interpolator.interpolate(points, is_index=is_index)

    @logging_and_warning_decorator()
    def visualize_disclination_lines(
        self,
        is_new: bool = True,
        is_wrap: bool = True,
        is_smooth: bool = True,
        is_extent: bool = True,
        min_line_length: Optional[int] = None,
        lines_scalars_name: Optional[str] = None,
        opts_scene=OptsScene(),
        opts_tube=OptsTube(color=None),
        opts_extent=OptsExtent(),
        logger=None,
        **kwargs,
    ):

        opts_extent.corners = self._corners

        opts_scene = merge_opts(opts_scene, kwargs, prefix="scene_")
        opts_tube = merge_opts(opts_tube, kwargs, prefix="line_")
        opts_extent = merge_opts(opts_extent, kwargs, prefix="extent_")

        check_bool_flags(locals())

        if min_line_length is None:
            msg = "No minimum line length has been provided for the plotted lines. "
            msg += f"Use the default value {self.DEFAULT_MINIMUM_LINE_LENGTH}"
            logger.info(msg)
            min_line_length = self.DEFAULT_MINIMUM_LINE_LENGTH

        if is_smooth and hasattr(self._lines[0], "_defect_coords_smooth_obj"):
            _min_len_length_smooth = self._lines[
                0
            ]._defect_coords_smooth_obj._opts_min_line_length
            if _min_len_length_smooth > min_line_length:
                msg = f">>> The minimum line length to be plotted ({min_line_length}) is shorter than the required minimum length for smoothing ({_min_len_length_smooth}) \n"
                msg += f">>> Use the larger value {_min_len_length_smooth} instead."
                min_line_length = _min_len_length_smooth
                logger.warning(msg)

        logger.debug(f"min_line_length = {min_line_length}")

        lines_plot = [
            line for line in self._lines if line._defect_num > min_line_length
        ]

        if lines_scalars_name is not None:
            logger.info("Scalars of lines are input")
            lines_scalars = [getattr(line, lines_scalars_name) for line in lines_plot]
            lines_colors = [None for line in lines_plot]
            if opts_tube.color is not None:
                logger.warning(
                    ">>> scalars of lines are input. Their color_input will be ignored"
                )
        else:
            lines_scalars = [None for line in lines_plot]

        if opts_tube.color is None:
            from ..general import blue_red_in_white_bg, sample_far

            color_map = blue_red_in_white_bg()
            color_map_length = np.shape(color_map)[0] - 1
            lines_colors = color_map[
                (sample_far(len(lines_plot)) * color_map_length).astype(int)
            ]
        else:
            lines_colors = [opts_tube.color for line in lines_plot]

        figure = self.add_scene(is_new, opts=opts_scene)

        logger.debug("Start to draw disclination lines")
        for line, line_color, line_scalar in zip(
            lines_plot, lines_colors, lines_scalars
        ):
            opts_tube = replace(opts_tube, name=line._name, color=line_color)
            line_visual = line.visualize(
                is_wrap=is_wrap,
                is_smooth=is_smooth,
                scalars=line_scalar,
                opts=opts_tube,
                logger=logger,
            )

            figure.add_object(line_visual, category="lines")

        if is_extent:
            extent = PlotExtent(opts_extent)
            figure.add_object(extent, category="extent")

    @logging_and_warning_decorator()
    def visualize_n_in_Q(
        self,
        plane_normal: Optional[Vect(3)] = None,
        plane_spacing: Optional[Number] = None,
        plane_size: Optional[Number] = None,
        is_new: bool = True,
        is_extent: bool = True,
        opts_grid=OptsPlaneGrid(),
        opts_nPlane=OptsnPlane(),
        opts_extent=OptsExtent(),
        opts_scene=OptsScene(),
        logger=None,
        **kwargs,
    ):

        opts_extent.corners = self._corners
        opts_grid.corners_limit = self._corners

        opts_grid = merge_opts(opts_grid, kwargs, prefix="plane_")
        opts_nPlane = merge_opts(opts_nPlane, kwargs, prefix="n_")
        opts_extent = merge_opts(opts_extent, kwargs, prefix="extent_")
        opts_scene = merge_opts(opts_scene, kwargs, prefix="scene_")

        if not hasattr(self, "_interpolator"):
            self.update_interpolator()

        opts_grid.normal = plane_normal
        opts_grid.spacing1 = plane_spacing
        opts_grid.spacing2 = plane_spacing
        opts_grid.size = plane_size

        check_bool_flags(locals())

        figure = self.add_scene(is_new, opts=opts_scene)

        nPlane = PlotnPlane(
            QInterpolator=self._interpolator,
            opts_grid=opts_grid,
            opts_nPlane=opts_nPlane,
            logger=logger,
        )

        figure.add_object(nPlane, category="nPlane")

        if is_extent:
            extent = PlotExtent(opts_extent)
            figure.add_object(extent, category="extent")

    def add_scene(self, is_new=True, opts=OptsScene):
        figure = PlotScene(is_new=is_new, opts=opts)
        if is_new or (not is_new and len(self.figures) == 0):
            self.figures.append(figure)
        else:
            figure = self.figures[-1]

        return figure

    def reset_figures(self):
        self.figures = []

    def __call__(self) -> np.ndarray:
        return self._Q
