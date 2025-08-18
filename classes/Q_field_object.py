import numpy as np
import time
from typing import Tuple, Optional, List, Union, Callable, Literal

from ..logging_decorator import logging_and_warning_decorator, Logger
from ..datatypes import (
    Vect3D,
    as_Vect3D,
    QField5,
    QField9,
    as_QField5,
    SField,
    nField,
    ColorRGB,
    DimensionFlagInput,
    as_dimension_info,
    check_Sn,
)
from ..field import (
    Q_diagonalize,
    getQ,
    generate_coordinate_grid,
    apply_linear_transform,
    n_color_immerse
)
from ..disclination import defect_detect, defect_classify_into_lines
from .Interpolator import Interpolator
from .visual_mayavi.plot_n_plane import PlotnPlane
from .visual_mayavi.plot_scene import PlotScene


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
        grid_offset: Vect3D = np.array([0, 0, 0]),
        grid_transform: np.ndarray = np.eye(3),
        is_diag: bool = True,
        logger: Logger = None,
    ) -> None:
        
        grid_offset = as_Vect3D(grid_offset)
        grid_transform = np.asarray(grid_transform, float)
        shape = np.shape(grid_transform)
        if shape[0]!=3 or shape[1]!=3:
            raise ValueError(f"grid_transform must be in shape (3,3). Got {grid_transform} instead.")

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
        self._grid_transform = grid_transform
        self._grid_offset = grid_offset
        self.update_grid(grid_transform=grid_transform, grid_offset=grid_offset)

        self.figures = []

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
        grid_transform: Optional[np.ndarray] = None,
        grid_offset: Optional[np.ndarray] = None,
        logger=None,
    ):
        """
        Generate the coordinates grid in the real space from the lattice indices through linear transform.
        See the document of apply_linear_transform()
        """
        self._grid_transform = grid_transform
        self._grid_offset = grid_offset
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
            offset=self._grid_offset,
            transform=self._grid_transform,
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
        window_ratio: Optional[int] = None,
        window_length: Optional[int] = None,
        order: int = 3,
        N_out_ratio: float = 3.0,
        min_line_length: Optional[int] = None,
        logger=None,
    ):
        if window_length is None:
            msg = "No data of window_length is input for smoothening lines. \n"
            msg += f"Use the default value {self.DEFAULT_SMOOTH_WINDOW_LENGTH}"
            logger.info(msg)
            window_length = self.DEFAULT_SMOOTH_WINDOW_LENGTH
        else:
            logger.debug(f"window_length = {window_length}")

        if min_line_length is None:
            msg = (
                "No data of minimum line length is input for lines to be smoothened. \n"
            )
            msg += f"Use the default value {self.DEFAULT_MINIMUM_LINE_LENGTH}"
            logger.info(msg)
            min_line_length = self.DEFAULT_MINIMUM_LINE_LENGTH
        else:
            logger.debug(f"min_line_length = {min_line_length}")

        for line in self._lines:
            if line._defect_num >= min_line_length:
                logger.debug(f"Start to smoothen {line._name}")
                line.update_smoothen(
                    window_ratio=window_ratio,
                    window_length=window_length,
                    order=order,
                    N_out_ratio=N_out_ratio,
                )

    def update_corners(self):

        from ..general import get_box_corners

        Lx, Ly, Lz = np.shape(self._Q)[:3] - np.array([1,1,1])
        corners_index = get_box_corners(Lx, Ly, Lz)
        corners = apply_linear_transform(
            corners_index, transform=self._grid_transform, offset=self._grid_offset
        )

        self._corners_index = corners_index
        self._corners = corners

        return corners

    @logging_and_warning_decorator()
    def update_interpolator(self, logger=None):

        from scipy.interpolate import RegularGridInterpolator

        shape = np.shape(self._Q)[:3]
        u = np.arange(shape[0])
        v = np.arange(shape[1])
        w = np.arange(shape[2])

        interpolator = RegularGridInterpolator(
                                (u, v, w),
                                self._Q,
                                method='linear',
                                bounds_error=True
                            )
        interpolator = Interpolator(
            interpolator,
            np.array([v[-1], u[-1], w[-1]]),
            transform=self._grid_transform,
            offset=self._grid_offset
        )

        self._interpolator = interpolator

        return self._interpolator
        
    def inperpolate(self, points: np.ndarray, is_index=False):
        if not hasattr(self, '_interpolator'):
            self.update_interpolator()
        return self._interpolator.interpolate(points, is_index=is_index)


    @logging_and_warning_decorator()
    def visualize_disclination_lines(
        self,
        min_line_length: Optional[int] = None,
        is_new: bool = True,
        fig_size: Tuple[int, int] = (1920, 1360),
        bgcolor: Vect3D = (1.0, 1.0, 1.0),
        fgcolor: Vect3D = (0.0, 0.0, 0.0),
        is_wrap: bool = True,
        is_smooth: bool = True,
        lines_color_input_all: Optional[np.ndarray] = None,
        lines_scalars_name: Optional[str] = None,
        radius: float = 0.5,
        sides: int = 6,
        opacity: float = 1,
        specular: float = 1,
        specular_color: Vect3D = (1.0, 1.0, 1.0),
        specular_power: float = 11,
        names_all: Optional[List[str]] = None,
        is_extent: bool = True,
        extent_radius: float = 1,
        extent_opacity: float = 1,
        logger=None,
    ):

        if min_line_length is None:
            msg = "No data of minimum line length is input for lines to be plotted. "
            msg += f"Use the default value {self.DEFAULT_MINIMUM_LINE_LENGTH}"
            logger.info(msg)
            min_line_length = self.DEFAULT_MINIMUM_LINE_LENGTH
        else:
            logger.debug(f"min_line_length = {min_line_length}")

        lines_plot = [
            line for line in self._lines if line._defect_num > min_line_length
        ]

        if lines_scalars_name is not None:
            logger.info("Scalars of lines are input")
            lines_scalars = [getattr(line, lines_scalars_name) for line in lines_plot]
            lines_colors = [None for line in lines_plot]
            if lines_color_input_all is not None:
                logger.warning(
                    ">>> scalars of lines are input. Their color_input will be ignored"
                )
        else:
            lines_scalars = [None for line in lines_plot]
            if lines_color_input_all is not None:
                if np.shape(lines_color_input_all) == (3,):
                    lines_colors = [tuple(lines_color_input_all) for line in lines_plot]
                elif np.shape(lines_color_input_all) == (len(lines_plot), 3):
                    lines_colors = lines_color_input_all
                else:
                    raise ValueError(
                        f"The shape of lines_color_input_all should either be (3,) or (len(lines_plot), 3), which is ({len(lines_plot)}, 3)"
                    )
            else:
                logger.info(
                    "No color data is input. Use the default color map, trying to set those longest lines with distinct colors"
                )
                from ..general import blue_red_in_white_bg, sample_far

                color_map = blue_red_in_white_bg()
                color_map_length = np.shape(color_map)[0] - 1
                lines_colors = color_map[
                    (sample_far(len(lines_plot)) * color_map_length).astype(int)
                ]

        if names_all is not None:
            if len(names_all) != len(lines_plot):
                logger.warning(
                    f"The length of names_all {len(names_all)} does not match the total numbers of lines to be plotted: {len(lines_plot)}.\n Use the default names instead."
                )
        else:
            names_all = [line._name for line in lines_plot]


        figure = self.add_scene(is_new, fig_size, bgcolor, fgcolor)

        logger.debug("Start to draw disclination lines")
        for line, line_color, line_scalar, name in zip(
            lines_plot, lines_colors, lines_scalars, names_all
        ):
            line_visual = line.visualize(
                is_wrap=is_wrap,
                is_smooth=is_smooth,
                radius=radius,
                opacity=opacity,
                color=line_color,
                sides=sides,
                specular=specular,
                specular_color=specular_color,
                specular_power=specular_power,
                scalars=line_scalar,
                name=name,
                logger=logger,
            )

            figure.add_object(line_visual, category="lines")

        if is_extent:
            extent = self.add_extent(extent_radius, extent_opacity)
            figure.add_object(extent, category="extent")

    @logging_and_warning_decorator()
    def visualize_n_in_Q(
            self,
            normal: Vect3D,
            space: float,
            size: float,
            is_new: bool = True,
            fig_size: Tuple[int, int] = (1920, 1360),
            bgcolor: Vect3D = (1.0, 1.0, 1.0),
            fgcolor: Vect3D = (0.0, 0.0, 0.0),
            shape: Literal["circle", "rectangle"] = "rectangle",
            origin: Vect3D = (0,0,0),
            axis1: Optional[Vect3D] = None,
            colors: Union[Callable[nField,ColorRGB], ColorRGB] = n_color_immerse,
            opacity: Union[Callable[nField, np.ndarray], float] = 1,
            length: float = 3.5,
            radius: float = 0.5,
            is_n_defect: bool = True,
            defect_opacity: float = 1,
            is_extent: bool = True,
            extent_radius: float = 1,
            extent_opacity: float = 1,
            logger=None,
    ):
        
        figure = self.add_scene(is_new, fig_size, bgcolor, fgcolor)

        self.update_interpolator()
        self.update_corners()

        nPlane = PlotnPlane(
            normal,
            space,
            size,
            self._interpolator,
            shape=shape,
            origin=origin,
            axis1=axis1,
            corners_limit=self._corners,
            colors=colors,
            opacity=opacity,
            length=length,
            radius=radius,
            is_n_defect=is_n_defect,
            defect_opacity=defect_opacity,
            grid_offset=self._grid_offset,
            grid_transform=self._grid_transform,
            logger=logger
        )

        figure.add_object(nPlane, category="nPlane")

        if is_extent:
            extent = self.add_extent(extent_radius, extent_opacity)
            figure.add_object(extent, category="extent")

    def add_scene(self, is_new, fig_size, bgcolor, fgcolor):
        figure = PlotScene(
            is_new=is_new,
            size=fig_size,
            bgcolor=bgcolor,
            fgcolor=fgcolor,
        )
        if is_new or (not is_new and len(self.figures)==0):
            self.figures.append(figure)
        else:
            figure = self.figures[-1]

        return figure

    def add_extent(self, extent_radius, extent_opacity):
        from .visual_mayavi.plot_extent import PlotExtent

        if not hasattr(self, "_corners"):
            self.update_corners()
        extent = PlotExtent(
            self._corners, radius=extent_radius, opacity=extent_opacity
        )

        return extent


    def reset_figures(self):
        self.figures = []


    def __call__(self) -> np.ndarray:
        return self._Q
