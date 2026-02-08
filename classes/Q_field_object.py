import numpy as np
import time
from typing import Union
from dataclasses import replace, dataclass, field, fields
from pyvistaqt import BackgroundPlotter

from ..logging_decorator import logging_and_warning_decorator
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
    check_Sn,
    Number,
    as_Number,
    DimensionFlagInput,
    as_dimension_info,
    check_bool_flags,
    UNSET,
    Unset,
    as_str,
    as_bool,
)
from ..field import (
    Q_diagonalize,
    getQ,
    generate_coordinate_grid,
    apply_linear_transform,
)
from ..disclination import defect_detect, defect_classify_into_lines
from .Interpolator import Interpolator
from .visual.plot_extent import PlotExtent
from .visual.plot_tube import OptsTube
from .visual.plot_rod import OptsRod
from .visual.plot_sphere import OptsSphere
from .visual.plot_delaunay import OptsDelaunay
from .visual.plot_figure import PlotFigure, OptsFigure
from .Q_plane import QPlane
from .visual.figure_manager import FigureManager
from .plane_grid import OptsPlaneGrid
from .opts import merge_opts_all, cover_value
from ..general import get_box_corners
from .smoothed_line import OptsSmooth
from .registry_base import RegistryBase
from .disclination_line import DisclinationLine
from .class_base import ClassBase


@dataclass(slots=True)
class InputQ:
    Q: Union[QField5, QField9] | Unset = UNSET
    S: SField | Unset = UNSET
    n: nField | Unset = UNSET
    box_periodic_flag: DimensionFlagInput = False
    grid_offset: Vect(3) = (0, 0, 0)
    grid_transform: Tensor((3, 3)) = field(default_factory=lambda: np.eye(3))
    default_miminum_line_length_smooth: Number = 61
    default_smooth_window_length: Number = 41
    default_miminum_line_length_visual: Number = 75

    __descriptions__ = {
        "Q": "Q field (tensor order parameter)",
        "S": "S field (scalar order parameter)",
        "n": "director field",
        "box_periodic_flag": "flag indicating whether periodic boundary condition is applied along each dimension",
        "grid_offset": "grid translation offset to map lattice indices to real-space coordinates",
        "grid_transform": "grid transform matrix to map lattice indices to real-space coordinates (3x3)",
        "default_miminum_line_length_smooth": "the minimum length (#points) of disclination lines to be smoothed",
        "default_smooth_window_length": "the default window length  (#points) of disclination lines to be smoothed",
        "default_miminum_line_length_visual": "the minimum length (#points) of disclination lines to be visualized",
    }

    _validators = {
        "Q": lambda v, d: as_QField5(v, name=d),
        "n": lambda v, d: check_Sn(v, "n"),
        "S": lambda v, d: check_Sn(v, "S"),
        "box_periodic_flag": lambda v, d: as_dimension_info(v, name=d, is_bool=True),
        "grid_offset": lambda v, d: as_Vect(v, name=d),
        "grid_transform": lambda v, d: as_Tensor(v, (3, 3), name=d),
        "default_miminum_line_length_smooth": lambda v, d: as_Number(
            v, name=d, value_range=(1, np.inf)
        ),
        "default_smooth_window_length": lambda v, d: as_Number(
            v, name=d, value_range=(2, np.inf)
        ),
        "default_miminum_line_length_visual": lambda v, d: as_Number(
            v, name=d, value_range=(2, np.inf)
        ),
    }

    def __setattr__(self, key, value):
        if key in self._validators:
            if value is not UNSET:
                desc = f"{key!r}: {self.__class__.__descriptions__[key]}"
                value = self._validators[key](value, desc)
        object.__setattr__(self, key, value)


class QFieldObject(ClassBase):

    __descriptions__ = {
        # --- Identity ---
        "raw_name": "Name identifier of this Q tensor object.",
        # --- Raw inputs ---
        "_raw_Q": "Raw Q-tensor field on lattice. Typically QField5 or QField9 (shape: (Nx, Ny, Nz, ...)).",
        "_raw_S": "Raw scalar order parameter field S on lattice (shape: (Nx, Ny, Nz)).",
        "_raw_n": "Raw director field n on lattice (shape: (Nx, Ny, Nz, 3)).",
        "_raw_box_periodic_flag": "Per-dimension periodic boundary condition flags (bool array-like of length 3).",
        "_raw_grid_offset": "A 3D vector, as the grid translation offset mapping lattice indices -> real-space coordinates.",
        "_raw_grid_transform": "A 3x3 tensor, as the linear transform mapping lattice indices -> real-space coordinates",
        # --- Defaults / thresholds ---
        "default_miminum_line_length_smooth": "Default minimum line length (#points) required to apply smoothing.",
        "default_smooth_window_length": "Default smoothing window length (#points) used when not specified.",
        "default_miminum_line_length_visual": "Default minimum line length (#points) required for visualization.",
        "default_cross_line_padding_num_": "Default number of points padded for smoothing cross-type disclination line.",
        # --- Derived grids / geometry ---
        "_calc_grid_index": "Lattice coordinate grid in index space (before applying transform/offset).",
        "_calc_grid": "Coordinate grid in real space after applying grid_transform and grid_offset.",
        "_calc_corners_index": "Box corners in lattice-index space.",
        "_calc_corners": "Box corners in real space coordinates.",
        "_calc_box_size_periodic_index": (
            "Effective periodic box size in index units. "
            "For periodic dims equals grid size, otherwise inf."
        ),
        "_calc_box_size_periodic_coord": "Effective periodic box size in real-space coordinates.",
        # --- Defects / disclinations  ---
        "_calc_defect_indices": "Indices (lattice coordinates) of detected defect points.",
        "_calc_defect_grid": "Real-space coordinates of detected defect points.",
        # --- Interpolation ---
        "_calc_interpolator": "Interpolator object for Q field in real space / index space.",
        # --- Visualization ---
        "_entity_figures": "FigureManager object to manage PlotFigure objects created for visualization.",
        "_entity_objects": "RegistryBase object to manage physical objects related to this Q field.",
        # --- Public properties (semantic) ---
        "S": "Property: scalar order parameter field S. This equals _raw_S",
        "n": "Property: director field n. This equals _raw_n",
        "lines": "Property: classified disclination lines.",
        "figs": "Property: visualization figures. This equals _entity_figures",
        "objs": "Property: physcial objects. This equals _entity_objects",
    }

    # __slots__ = tuple(
    #     k
    #     for k, v in __descriptions__.items()
    #     if not v.startswith("Property:") and k not in ClassBase.__slots__
    # ) #!!!

    @logging_and_warning_decorator()
    def __init__(
        self,
        is_detect_defects: bool = True,
        is_classify_lines: bool = True,
        inputValue=InputQ(),
        name: str = "Q",
        logger=None,
        **kwargs,
    ) -> None:

        super().__init__(name=name, name_replace="Q")
        # self.name = name  #!!!!!!!

        inputValue = merge_opts_all({"": inputValue}, kwargs, type(self).__name__)[""]
        for f in fields(inputValue):
            k = f.name
            v = getattr(inputValue, k)
            if k.startswith("default"):
                object.__setattr__(self, k, v)
            else:
                object.__setattr__(self, f"_raw_{k}", v)

        object.__setattr__(
            self,
            "_entity_objects",
            RegistryBase(f"The objects manager of Q field {self.name!r}"),
        )

        logger.progress(f"Start to initialize Q tensor `{self.name}`.")
        if self._raw_n is not UNSET:
            logger.debug("Initialize Q field with S and n")
            if self._raw_S is UNSET:
                logger.warning("No S input. Set to 1 everywhere.")
                object.__setattr__(
                    self, "_raw_S", np.zeros(np.shape(self._raw_n)[:-1]) + 1.0
                )
            if self._raw_Q is not UNSET:
                logger.warning(
                    "Both Q and n are provided to initialize Q field. Q will be IGNORED."
                )
            if np.shape(self._raw_S) != np.shape(self._raw_n)[:3]:
                raise ValueError(
                    "Shape mismatch between director field `n` and scalar field `S`: "
                    f"expected n.shape[:3] == S.shape, "
                    f"but got n.shape = {self._raw_n.shape}, S.shape = {self._raw_S.shape}."
                )
            object.__setattr__(
                self, "_raw_Q", as_QField5(getQ(self._raw_n, S=self._raw_S))
            )
        else:
            if self._raw_Q is not UNSET:
                temp_S, temp_n = Q_diagonalize(self._raw_Q)
                object.__setattr__(self, "_raw_S", temp_S)
                object.__setattr__(self, "_raw_n", temp_n)
            else:
                raise NameError("No data is input to initialize Q field.")

        logger.detail("Recording the information of periodic boundary conditions.")
        object.__setattr__(self, "_calc_box_size_periodic_index", np.zeros(3))
        for i, flag in enumerate(self._raw_box_periodic_flag):
            if flag:
                self._calc_box_size_periodic_index[i] = np.shape(self._raw_Q)[i]
            else:
                self._calc_box_size_periodic_index[i] = np.inf
        T = np.asarray(self._raw_grid_transform, dtype=float)
        diag = np.diag(T).astype(float)
        object.__setattr__(
            self, "_calc_box_size_periodic_coord", np.full(3, np.inf, dtype=float)
        )
        finite_mask = np.isfinite(self._calc_box_size_periodic_index)
        self._calc_box_size_periodic_coord[finite_mask] = (
            diag[finite_mask] * self._calc_box_size_periodic_index[finite_mask]
        )
        msg = f"Effective periodic box size in lattice-index units is {self._calc_box_size_periodic_index}.\n"
        msg += f"Effective periodic box size in real-space coordinates is {self._calc_box_size_periodic_coord}."
        logger.detail(msg)

        logger.detail("Generating grid of Q")
        grid_shape = np.shape(self._raw_Q)[:3]
        object.__setattr__(
            self,
            "_calc_grid_index",
            generate_coordinate_grid(grid_shape, grid_shape)[0],
        )
        object.__setattr__(
            self,
            "_calc_grid",
            apply_linear_transform(
                self._calc_grid_index,
                transform=self._raw_grid_transform,
                offset=self._raw_grid_offset,
            ),
        )

        logger.debug("Generating the coorners of Q.")
        Lx, Ly, Lz = np.shape(self._raw_Q)[:3] - np.array([1, 1, 1])
        corners_index = get_box_corners(Lx, Ly, Lz)
        corners = apply_linear_transform(
            corners_index,
            transform=self._raw_grid_transform,
            offset=self._raw_grid_offset,
        )
        
        object.__setattr__(self, "_calc_corners_index", corners_index)
        object.__setattr__(self, "_calc_corners", corners)
        logger.debug(
            f"Box corners in lattice-index units is {self._calc_corners_index}."
            f"Box corners in reap-space coordinates is {self._calc_corners}."
            )

        if (not is_detect_defects) and is_classify_lines:
            is_classify_lines = False
            msg = (
                f"Invalid combination: is_detect_defects={is_detect_defects} "
                f"and is_classify_lines={is_classify_lines}.\n"
                "Line classification depends on defect detection. "
                "Automatically disabling line classification."
            )
            logger.warning(msg)

        if is_detect_defects:

            start = time.time()

            msg = "Start defect analysis as detecting defects"
            if is_classify_lines:
                msg += " and classifying them into distinct lines"
            msg += f" for Q tensor `{self.name}` \n"
            msg += "This operation might take a while.\n"
            msg += "You can disable this automatic operation by setting is_detect_defects=False and is_classify_lines=False when initializing the Q tensor."
            logger.progress(msg)

            self.act_defect_detect()

            if is_classify_lines:
                self.act_lines_classify()

            logger.progress(
                f"Defect analysis is finished, with {time.time()-start:.2f} s"
            )

        self.act_add_interpolator()

        object.__setattr__(self, "_entity_figures", FigureManager())

    @logging_and_warning_decorator(start_finish_level=5)
    def act_defect_detect(self, logger=None):
        object.__setattr__(
            self, 
            "_calc_defect_indices", 
            defect_detect(
                self._raw_n,
                is_boundary_periodic=self._raw_box_periodic_flag,
            )
        )
        logger.info(f"{len(self._calc_defect_indices)} defects are found.")

        logger.detail("Start to calculate the coordinates of defects in real space.")
        object.__setattr__(
            self, 
            "_calc_defect_grid",
            apply_linear_transform(
                self._calc_defect_indices,
                transform=self._raw_grid_transform,
                offset=self._raw_grid_offset,
            )
        )

    @logging_and_warning_decorator(start_finish_level=5)
    def act_lines_classify(self, logger=None):

        lines = defect_classify_into_lines(
            self._calc_defect_indices,
            box_size_periodic=self._calc_box_size_periodic_index,
            grid_offset=self._raw_grid_offset,
            grid_transform=self._raw_grid_transform,
        )
        logger.detail("Sorting lines by length")
        lines = sorted(lines, key=lambda line: line._calc_defect_num, reverse=True)
        for i, line in enumerate(lines):
            line.name = f"disclination line {i}"
            self._entity_objects.act_register(line)

        logger.info(f"{len(lines)} lines are found.")

        return lines

    @logging_and_warning_decorator()
    def act_lines_smooth(
        self,
        opts: OptsSmooth | None = None,
        logger=None,
        **kwargs,
    ):

        logger.detail("Start to smoothen disclination lines.")

        if opts is None:
            opts = OptsSmooth()

        opts = merge_opts_all({"": opts}, kwargs, "SmoothedLine")[""]
        opts.is_window_warning = False

        if opts.min_line_length is UNSET:
            opts.min_line_length = self.default_miminum_line_length_smooth
            msg = "No input value provided for minimum smoothed line length. \n"
            msg += f"Using the default value self.default_miminum_line_length_smooth={self.default_smooth_window_length}."
            logger.info(msg)

        opts.act_finalize()

        if opts.window_length is not None and opts.window_ratio is not None:
            msg = f"``window_length`` of smoothing disclination lines is manual input as {opts.window_length}.\n"
            msg += f"``window_ratio`` as {opts.window_ratio} would be ignored."
            logger.warning(msg)
            opts.window_ratio = None

        if opts.window_length is None and opts.window_ratio is None:
            opts.window_length = self.default_smooth_window_length
            msg = "No input value provided for smooth window length of disclination lines. \n"
            msg += f"Using the default value self.default_smooth_window_length={self.default_smooth_window_length}."
            logger.info(msg)

        msg = f"Start to smooth disclination lines in Q tensor {self.name!r} With paramaters: \n"
        msg += f"window length = {opts.window_length}\n"
        msg += f"window ratio = {opts.window_ratio}\n"
        msg += f"minimum smoothed line length = {opts.min_line_length}"
        logger.debug(msg)

        num_smooth = 0
        window_list = {}
        for line in self.lines:
            if line._calc_defect_num >= opts.min_line_length:
                line.act_smooth(opts=opts)
                num_smooth += 1
                window_list[line.name] = line.smooth.opts.window_length
            else:
                logger.debug(
                    f"Line `{line.name}` is not smoothed because it is too short, with only {line._calc_defect_num} defects. "
                )

        msg = f"There are {len(self.lines)} disclination lines in total, with {num_smooth} lines are smoothed.\n"
        msg += "The smoothing window length is: "
        if opts.window_length is not None:
            msg += str(opts.window_length)
        else:
            msg += "\n"
            for k, v in window_list.items():
                msg += f"{k}: {v} \n"
        logger.info(msg)

    @logging_and_warning_decorator()
    def act_add_interpolator(self, logger=None):

        from scipy.interpolate import RegularGridInterpolator

        shape = np.shape(self._raw_Q)[:3]
        u = np.arange(shape[0])
        v = np.arange(shape[1])
        w = np.arange(shape[2])

        interpolator = RegularGridInterpolator(
            (u, v, w), self._raw_Q, method="linear", bounds_error=True
        )
        interpolator = Interpolator(
            interpolator,
            np.array([u[-1], v[-1], w[-1]]),
            grid_transform=self._raw_grid_transform,
            grid_offset=self._raw_grid_offset,
        )

        object.__setattr__(self, "_calc_interpolator", interpolator)

        return self._calc_interpolator

    def act_interpolate(self, points: np.ndarray, is_index=False):
        if not hasattr(self, "_interpolator"):
            self.act_act_add_interpolator()
        return self._calc_interpolator.interpolate(points, is_index=is_index)

    @logging_and_warning_decorator()
    def _helper_set_figure(
        self,
        is_new: bool,
        figure: PlotFigure | str | int | BackgroundPlotter | None,
        opts_figure: OptsFigure,
        title: str,
        logger=None,
    ):

        is_new = as_bool(is_new, name="Whether to create a new figure", replace=True)

        if is_new:
            if figure is not None:
                logger.warning(
                    "is_new=True was specified while figure is not None."
                    "The figure argument will be ignored and a new figure will be created."
                )
            figure = PlotFigure(opts=opts_figure, name=title)
        else:
            try:
                if isinstance(figure, (str, int)):
                    figure = self.figs[figure]
                    figure.act_commit(opts_figure)
                elif figure is None:
                    if (
                        hasattr(self.figs, "_state_active_name")
                        and self.figs[self.figs._state_active_name]
                    ):
                        figure = self.figs[self.figs._state_active_name]
                        figure.act_commit(opts_figure)
                    else:
                        figure = PlotFigure(opts=opts_figure, name=title)
                elif isinstance(figure, PlotFigure):
                    figure.act_commit(opts_figure)
                elif isinstance(figure, BackgroundPlotter):
                    figure = PlotFigure(plotter=figure, opts=opts_figure, name=title)
                else:
                    raise ValueError(
                        "`figure` input must be either index in FigureManager (str or int) "
                        "or a valid PlotFigure object, or a valid pyvistaqt BackgroundPlotter object, "
                        "or None (creating a new figure) "
                        "Got type {type(figure)!r} instead."
                    )
            except:
                logger.exception("Could not find figure in FigureManager.")
                logger.recovery("Create a new figure instead.")
                figure = PlotFigure(opts=opts_figure, name=title)

        if figure.name.startswith(figure._DEFAULT_NAME):
            figure.name = title
        self.figs.act_register(figure, is_contain_ok=True)
        self.figs.act_set_active(figure.name)

        return figure

    @logging_and_warning_decorator()
    def act_visualize_disclination_lines(
        self,
        figure: PlotFigure | str | int | BackgroundPlotter | None = None,
        is_new: bool = False,
        is_wrap: bool = True,
        is_smooth: bool = True,
        is_extent: bool = True,
        min_line_length: int | None = None,
        opts_figure: OptsFigure | None = None,
        opts_tube: OptsTube | None = None,
        opts_extent: OptsTube | None = None,
        title: str = "disclination lines",
        logger=None,
        **kwargs,
    ):

        #!!! lines_scalars_name

        logger.detail("Dealing with the parameters")

        if opts_extent is None:
            opts_extent = OptsTube()
        if opts_figure is None:
            opts_figure = OptsFigure()
        if opts_tube is None:
            opts_tube = OptsTube(color="sample_far")

        merge = merge_opts_all(
            {"figure_": opts_figure, "line_": opts_tube, "extent_": opts_extent},
            kwargs,
            type(self).__name__,
        )

        opts_figure = merge["figure_"]
        opts_tube = merge["line_"]
        opts_extent = merge["extent_"]

        check_bool_flags(locals())

        figure = self._helper_set_figure(is_new, figure, opts_figure, title)

        if min_line_length is None:
            logger.info(
                "No minimum line length has been provided for the plotted lines. "
                f"Use the default value {self.default_miminum_line_length_visual}"
            )
            min_line_length = self.default_miminum_line_length_visual

        logger.debug(f"min_line_length = {min_line_length}")

        lines_plot = [
            line for line in self.lines if line._calc_defect_num > min_line_length
        ]

        # logger.detail("Searching the attributes of ")
        # if lines_scalars_name is not None:
        #     logger.info("Scalars of lines are input")
        #     try:
        #         lines_scalars = [getattr(line, lines_scalars_name) for line in lines_plot]
        #         lines_colors = 'scalars'
        #         if opts_tube.color is not 'scalars':
        #             logger.warning(
        #                 "scalars of lines are input. Their color_input will be ignored"
        #             )
        # else:
        #     lines_scalars = [None for line in lines_plot]
        lines_scalars = [UNSET for line in lines_plot]

        if opts_tube.color == "sample_far":
            logger.detail(
                "Apply a variety of colors to ensure disclination lines are easily identifiable."
            )
            from ..general import blue_red_in_white_bg, sample_far

            color_map = blue_red_in_white_bg()
            color_map_length = np.shape(color_map)[0] - 1
            lines_colors = color_map[
                (sample_far(len(lines_plot)) * color_map_length).astype(int)
            ]
        else:
            lines_colors = [opts_tube.color for line in lines_plot]

        # figure = self.act_add_scene(is_new, opts=opts_scene)

        logger.debug("Start to draw disclination lines")
        for line, line_color, line_scalar in zip(
            lines_plot, lines_colors, lines_scalars
        ):
            opts_tube = replace(opts_tube, color=line_color)
            line_visual = line.act_visualize(
                figure=figure,
                is_wrap=is_wrap,
                is_smooth=is_smooth,
                # scalars=line_scalar,
                opts=opts_tube,
            )

        if is_extent:
            extent = PlotExtent(
                self._calc_corners,
                figure=figure,
                opts=opts_extent,
                is_reset_camera=False,
            )

    @logging_and_warning_decorator()
    def act_visualize_n_plane(
        self,
        figure: PlotFigure | str | int | BackgroundPlotter | None = None,
        is_new: bool = False,
        is_extent: bool = True,
        is_defect: bool = False,
        opts_grid: OptsPlaneGrid | None = None,
        opts_n: OptsRod | None = None,
        opts_nb: OptsRod | None = None,
        opts_nd: OptsRod | None = None,
        opts_figure: OptsFigure | None = None,
        opts_extent: OptsTube | None = None,
        opts_defect: OptsSphere | None = None,
        title: str = "visualization of n plane",
        name_plane: str = "n-plane",
        logger=None,
        **kwargs,
    ):

        logger.detail("Dealing with the parameters")
        if opts_grid is None:
            opts_grid = OptsPlaneGrid()
        if opts_extent is None:
            opts_extent = OptsTube()
        if opts_figure is None:
            opts_figure = OptsFigure()
        if opts_n is None:
            opts_n = OptsRod()
        if opts_nb is None:
            opts_nb = OptsRod()
        if opts_nd is None:
            opts_nd = OptsRod()
        if opts_defect is None:
            opts_defect = OptsSphere()

        merge = merge_opts_all(
            {
                "figure_": opts_figure,
                "grid_": opts_grid,
                "extent_": opts_extent,
                "n_": opts_n,
                "nb_": opts_nb,
                "nd_": opts_nd,
                "defect_": opts_defect,
            },
            kwargs,
            type(self).__name__,
        )

        opts_figure = merge["figure_"]
        opts_grid = merge["grid_"]
        opts_extent = merge["extent_"]
        opts_n = merge["n_"]
        opts_nb = merge["nb_"]
        opts_nd = merge["nd_"]
        opts_defect = merge["defect_"]

        cover_value(opts_nb, is_allow_cover_target_set=False, **(opts_n.act_asdict()))
        cover_value(opts_nd, is_allow_cover_target_set=False, **(opts_n.act_asdict()))

        figure = self._helper_set_figure(is_new, figure, opts_figure, title)

        if not hasattr(self, "_calc_interpolator"):
            self.act_add_interpolator()

        logger.detail("Create the plane.")
        n_plane = QPlane(
            self._calc_interpolator,
            name=name_plane,
            opts=opts_grid,
            opts_defaults_override={
                "size": 1.8 * np.max(self.S.shape),
                "spacing": 1,
                "corners_limit": self._calc_corners_index,
                "grid_offset": self._raw_grid_offset,
                "grid_transform": self._raw_grid_transform,
            },
        )
        self.objs.act_register(n_plane)

        n_plane.act_visualize_n(
            figure=figure,
            is_defect=is_defect,
            opts_nb=opts_nb,
            opts_nd=opts_nd,
            opts_defect=opts_defect,
        )

        if is_extent:
            PlotExtent(
                self._calc_corners,
                figure=figure,
                opts=opts_extent,
                is_reset_camera=False,
            )

    @logging_and_warning_decorator()
    def act_visualize_S_plane(
        self,
        figure: PlotFigure | str | int | BackgroundPlotter | None = None,
        is_new: bool = False,
        is_extent: bool = True,
        opts_grid: OptsPlaneGrid | None = None,
        opts_S: OptsDelaunay | None = None,
        opts_figure: OptsFigure | None = None,
        opts_extent: OptsTube | None = None,
        title: str = "visualization of S plane",
        name_plane: str = "S-plane",
        logger=None,
        **kwargs,
    ):

        logger.detail("Dealing with the parameters")
        if opts_grid is None:
            opts_grid = OptsPlaneGrid()
        if opts_extent is None:
            opts_extent = OptsTube()
        if opts_figure is None:
            opts_figure = OptsFigure()
        if opts_S is None:
            opts_S = OptsDelaunay()

        merge = merge_opts_all(
            {
                "figure_": opts_figure,
                "grid_": opts_grid,
                "extent_": opts_extent,
                "S_": opts_S,
            },
            kwargs,
            type(self).__name__,
        )

        opts_figure = merge["figure_"]
        opts_grid = merge["grid_"]
        opts_extent = merge["extent_"]
        opts_S = merge["S_"]

        figure = self._helper_set_figure(is_new, figure, opts_figure, title)

        if not hasattr(self, "_calc_interpolator"):
            self.act_add_interpolator()

        logger.detail("Create the plane.")
        S_plane = QPlane(
            self._calc_interpolator,
            name=name_plane,
            opts=opts_grid,
            opts_defaults_override={
                "size": 1.8 * np.max(self.S.shape),
                "spacing": 1,
                "corners_limit": self._calc_corners_index,
                "grid_offset": self._raw_grid_offset,
                "grid_transform": self._raw_grid_transform,
            },
        )
        self.objs.act_register(S_plane)

        S_plane.act_visualize_S(
            figure=figure,
            opts_S=opts_S,
        )

        if is_extent:
            PlotExtent(
                self._calc_corners,
                figure=figure,
                opts=opts_extent,
                is_reset_camera=False,
            )

    @property
    def lines(self):
        result = [
            item for item in self._entity_objects if isinstance(item, DisclinationLine)
        ]
        return result

    @property
    def figs(self):
        return self._entity_figures

    @property
    def objs(self):
        return self._entity_objects

    @property
    def S(self):
        return self._raw_S

    @property
    def n(self):
        return self._raw_n

    def __call__(self) -> np.ndarray:
        return self._raw_Q

    @property
    def name(self):
        return self.raw_name

    @name.setter
    def name(self, value: str):
        name = as_str(value, name="The name of the Q field")
        self.raw_name = name
