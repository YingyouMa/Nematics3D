import numpy as np
import time
from typing import Optional, Union
from dataclasses import replace, dataclass, asdict, field

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
    as_str
)
from ..field import (
    Q_diagonalize,
    getQ,
    generate_coordinate_grid,
    apply_linear_transform,
)
from ..disclination import defect_detect, defect_classify_into_lines
from .Interpolator import Interpolator
from .visual_mayavi.plot_n_plane import OptsnPlane, PlotnPlane
from .visual_mayavi.plot_scene import PlotScene, OptsScene
from .visual_mayavi.plot_extent import PlotExtent, OptsExtent
from .visual.plot_tube import OptsTube
from .plane_grid import OptsPlaneGrid
from .opts import merge_opts_all
from ..general import get_box_corners
from .smoothed_line import OptsSmooth


@dataclass(slots=True)
class InputQ:
    Q: Union[QField5, QField9] = None
    S: SField = None
    n: nField = None
    box_periodic_flag: DimensionFlagInput = False
    grid_offset: Vect(3) = (0,0,0)
    grid_transform: Tensor((3, 3)) = field(default_factory=lambda: np.eye(3))
    default_miminum_line_length_smooth: Number = 61
    default_smooth_window_length: Number = 41
    default_miminum_line_length_visual: Number = 75
    name: str = "None"
    
    __descriptions__ = {
        "Q": "Q field (tensor order parameter)",
        "S": "S field (scalar order parameter)",
        "n": "director field",
        "box_periodic_flag": "flag indicating whether periodic boundary condition is applied along each dimension",
        "grid_offset": "grid translation offset to map lattice indices to real-space coordinates",
        "grid_transform": "grid transform matrix to map lattice indices to real-space coordinates (3x3)",
        "name": "name identifier of this Q field",
        "default_miminum_line_length_smooth": "the minimum length (#points) of disclination lines to be smoothed",
        "default_smooth_window_length": "the default window length  (#points) of disclination lines to be smoothed",
        "default_miminum_line_length_visual": "the minimum length (#points) of disclination lines to be visualized"
    }
    
    _validators = {
        "Q": lambda self, v:(
            None
            if v is None
            else as_QField5(v, name=self.__descriptions__["Q"])
            ),
        "n": lambda self, v:(
            None
            if v is None
            else check_Sn(v, "n")
            ),
        "S": lambda self, v:(
            None
            if v is None
            else check_Sn(v, "S")
            ),
        "box_periodic_flag": lambda self, v: (
            as_dimension_info(v, name=self.__descriptions__["box_periodic_flag"], is_bool=True)
            ),
        "grid_offset": lambda self, v: as_Vect(
            v, name=self.__descriptions__["grid_offset"]
        ),
        "grid_transform": lambda self, v: as_Tensor(
            v, (3, 3), name=self.__descriptions__["grid_transform"]
        ),
        "default_miminum_line_length_smooth": lambda self, v: (
            int(as_Number(v, name=self.__descriptions__["default_miminum_line_length_smooth"], value_range=(1, np.inf)))
            ),
        "default_smooth_window_length": lambda self, v: (
            as_Number(v, name=self.__descriptions__["default_smooth_window_length"], value_range=(2, np.inf))
            ),
        "default_miminum_line_length_visual": lambda self, v: (
            as_Number(v, name=self.__descriptions__["default_miminum_line_length_visual"], value_range=(1, np.inf))
            ),
        "name": lambda self, v: as_str(v, name="Name of Q field")
        }
    
    def __setattr__(self, key, value):
        if key in self._validators:
            value = self._validators[key](self, value)
        object.__setattr__(self, key, value)


class QFieldObject:
    
    __descriptions__ = {
        # --- Identity ---
        "name": ["Name identifier of this Q tensor object.", "SLOT"],

        # --- Raw inputs (stored as _raw_*) ---
        "_raw_Q": ["Raw Q-tensor field on lattice. Typically QField5 or QField9 (shape: (Nx, Ny, Nz, ...)).", "SLOT"],
        "_raw_S": ["Raw scalar order parameter field S on lattice (shape: (Nx, Ny, Nz)).", "SLOT"],
        "_raw_n": ["Raw director field n on lattice (shape: (Nx, Ny, Nz, 3)).", "SLOT"],
        "_raw_box_periodic_flag": ["Per-dimension periodic boundary condition flags (bool array-like of length 3).", "SLOT"],
        "_raw_grid_offset": ["Grid translation offset mapping lattice indices -> real-space coordinates (Vect(3)).", "SLOT"],
        "_raw_grid_transform": ["Grid linear transform mapping lattice indices -> real-space coordinates (3x3 Tensor).", "SLOT"],

        # --- Defaults / thresholds (stored as _raw_default_*) ---
        "_raw_default_miminum_line_length_smooth": ["Default minimum line length (#points) required to apply smoothing.", "SLOT"],
        "_raw_default_smooth_window_length": ["Default smoothing window length (#points) used when not specified.", "SLOT"],
        "_raw_default_miminum_line_length_visual": ["Default minimum line length (#points) required for visualization.", "SLOT"],
        "_raw_default_cross_line_padding_num_": ["Default number of points padded for smoothing cross-type disclination line.", "SLOT"],

        # --- Derived grids / geometry (stored as _calc_*) ---
        "_calc_grid_index": ["Lattice coordinate grid in index space (before applying transform/offset).", "SLOT"],
        "_calc_grid": ["Coordinate grid in real space after applying grid_transform and grid_offset.", "SLOT"],
        "_calc_corners_index": ["Box corners in lattice-index space.", "SLOT"],
        "_calc_corners": ["Box corners in real space coordinates.", "SLOT"],
        "_calc_box_size_periodic_index": [(
            "Effective periodic box size in index units. "
            "For periodic dims equals grid size, otherwise inf."
        ), "SLOT"],
        "_calc_box_size_periodic_coord": ["Effective periodic box size in real-space coordinates.", "SLOT"],

        # --- Defects / disclinations (stored as _calc_*) ---
        "_calc_defect_indices": ["Indices (lattice coordinates) of detected defect points.", "SLOT"],
        "_calc_defect_grid": ["Real-space coordinates of detected defect points.", "SLOT"],
        "_calc_lines": ["List of disclination line objects classified from defect indices.", "SLOT"],

        # --- Interpolation (stored as _calc_*) ---
        "_calc_interpolator": ["Interpolator object for Q field in real space / index space.", "SLOT"],

        # --- Visualization state ---
        "_calc_figures": ["List of PlotScene objects created for visualization.", "SLOT"],

        # --- Public properties (semantic) ---
        "S": ["Property accessor: scalar order parameter field S. This equals _raw_S", "PROPERTY"],
        "n": ["Property accessor: director field n. This equals _raw_n", "PROPERTY"],
        "lines": ["Property accessor: classified disclination lines. This equals _calc_lines", "PROPERTY"],
        "figs": ["Property accessor: visualization scenes/figures. This equals _calc_figures", "PROPERTY"],
    }

    __slots__ = tuple(
        k for k, (_, kind) in __descriptions__.items()
        if kind == "SLOT"
    )

    @logging_and_warning_decorator()
    def __init__(
        self,
        is_detect_defects: bool = True,
        is_classify_lines: bool = True,
        inputValue = InputQ(),
        logger=None,
        **kwargs,
    ) -> None:
        
        inputValue = merge_opts_all({"": inputValue}, kwargs, type(self).__name__)[""]
        for k, v in asdict(inputValue).items():
            if k == "name":
               setattr(self, "name", v)
            else:
               setattr(self, f"_raw_{k}", v)
        
        logger.progress(f"Start to initialize Q tensor `{self.name}`.")
        if self._raw_n is not None:
            logger.debug("Initialize Q field with S and n")
            if self._raw_S is None:
                logger.warning("No S input. Set to 1 everywhere.")
                self._raw_S = np.zeros(np.shape(self._raw_n)[:-1]) + 1.0
            if self._raw_Q is not None:
                logger.warning("Both Q and n are provided to initialize Q field. Q will be IGNORED.")
            if np.shape(self._raw_S) != np.shape(self._raw_n)[:3]:
                raise ValueError(
                    "Shape mismatch between director field `n` and scalar field `S`: "
                    f"expected n.shape[:3] == S.shape, "
                    f"but got n.shape = {self._raw_n.shape}, S.shape = {self._raw_S.shape}."
                )
            self._raw_Q = as_QField5(getQ(self._raw_n, S=self._raw_S))
        else:
            if self._raw_Q is not None:
                self._raw_S, self._raw_n = Q_diagonalize(self._raw_Q)
            else:
                raise NameError("No data is input to initialize Q field.")
            
        
        logger.detail("Recording the information of periodic boundary conditions.")
        self._calc_box_size_periodic_index = np.zeros(3)
        for i, flag in enumerate(self._raw_box_periodic_flag):
            if flag:
                self._calc_box_size_periodic_index[i] = np.shape(self._raw_Q)[i]
            else:
                self._calc_box_size_periodic_index[i] = np.inf
        T = np.asarray(self._raw_grid_transform, dtype=float)
        diag = np.diag(T).astype(float)
        self._calc_box_size_periodic_coord = np.full(3, np.inf, dtype=float)
        finite_mask = np.isfinite(self._calc_box_size_periodic_index)
        self._calc_box_size_periodic_coord[finite_mask] = (
            diag[finite_mask] * self._calc_box_size_periodic_index[finite_mask]
        )
        msg = f"Effective periodic box size in lattice-index units is {self._calc_box_size_periodic_index}.\n"
        msg += f"Effective periodic box size in real-space coordinates is {self._calc_box_size_periodic_coord}."
        logger.detail(msg)

        logger.detail("Generating grid of Q")
        grid_shape = np.shape(self._raw_Q)[:3]
        self._calc_grid_index, _, _ = generate_coordinate_grid(grid_shape, grid_shape)
        self._calc_grid = apply_linear_transform(
            self._calc_grid_index, transform=self._raw_grid_transform, offset=self._raw_grid_offset
        )

        self._calc_figures = []
        
        logger.debug("Generating the coorners of Q.")
        Lx, Ly, Lz = np.shape(self._raw_Q)[:3] - np.array([1, 1, 1])
        corners_index = get_box_corners(Lx, Ly, Lz)
        corners = apply_linear_transform(
            corners_index, transform=self._raw_grid_transform, offset=self._raw_grid_offset
        )
        self._calc_corners_index = corners_index
        self._calc_corners = corners
        msg = f"Box corners in lattice-index units is {self._calc_corners_index}.\n"
        msg = f"Box corners in reap-space coordinates is {self._calc_corners}."
        
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
                
            logger.progress(f"Defect analysis is finished, with {time.time()-start:.2f} s")
        
          
    @logging_and_warning_decorator(start_finish_level=5)
    def act_defect_detect(self, logger=None):
        self._calc_defect_indices = defect_detect(
            self._raw_n,
            is_boundary_periodic=self._raw_box_periodic_flag,
        )
        logger.info(f"{len(self._calc_defect_indices)} defects are found.")
        
        logger.detail("Start to calculate the coordinates of defects in real space.")
        self._calc_defect_grid = apply_linear_transform(
            self._calc_defect_indices,
            transform=self._raw_grid_transform,
            offset=self._raw_grid_offset,
        )

    @logging_and_warning_decorator(start_finish_level=5)
    def act_lines_classify(self, logger=None):
        
        self._calc_lines = defect_classify_into_lines(
            self._calc_defect_indices,
            box_size_periodic=self._calc_box_size_periodic_index,
            grid_offset=self._raw_grid_offset,
            grid_transform=self._raw_grid_transform,
            logger=logger,
        )
        logger.detail("Sorting lines by length")
        self._calc_lines = sorted(
            self._calc_lines, key=lambda line: line._calc_defect_num, reverse=True
        ) 
        for i, line in enumerate(self._calc_lines):
            line.name = f"line{i}"
            
        logger.info(f"{len(self._calc_lines)} lines are found.")

        return self._calc_lines

    @logging_and_warning_decorator()
    def act_lines_smooth(
        self,
        opts=OptsSmooth(),
        logger=None,
        **kwargs,
    ):
        
        logger.detail("Start to smoothen disclination lines.")
        
        opts = merge_opts_all({"": opts}, kwargs, "SmoothedLine")[""]
        opts.is_window_warning = False

        if opts.window_length is not None and opts.window_ratio is not None:
            msg = f"``window_length`` of smoothing disclination lines is manual input as {opts.window_length}.\n"
            msg += f"``window_ratio`` as {opts.window_ratio} would be ignored."
            logger.warning(msg)
            opts.window_ratio = None
            
        if opts.window_length is None and opts.window_ratio is None:
            opts.window_length = self._raw_default_smooth_window_length
            msg = "No input value provided for smooth window length of disclination lines. \n"
            msg += "Using the default value self._raw_default_smooth_window_length={self._raw_default_smooth_window_length}."
            logger.info(msg)
        
        if 'min_line_length' not in kwargs.keys():
            opts.min_line_length = self._raw_default_miminum_line_length_smooth
            msg = "No input value provided for minimum smoothed line length. \n"
            msg += "Using the default value self._raw_default_miminum_line_length_smooth={self._raw_default_smooth_window_length}."
            logger.info(msg)
            
        msg = "Start to smooth disclination lines in Q tensor `self.name` With paramaters: \n"
        msg += f"window length = {opts.window_length}\n"
        msg += f"window ratio = {opts.window_ratio}\n"
        msg += f"minimum smoothed line length = {opts.min_line_length}"
        logger.debug(msg)

        num_smooth = 0
        window_list = {}
        for line in self._calc_lines:
            if line._calc_defect_num >= opts.min_line_length:
                line.act_smooth(opts=opts)
                num_smooth += 1
                window_list[line.name] = line._calc_defect_coords_smooth_obj.opts_window_length
            else:
                logger.debug(f"Line `{line.name}` is not smoothed because it is too short, with only {line._calc_defect_num} defects. ")
                
        msg = f"There are {len(self._calc_lines)} disclination lines in total, with {num_smooth} lines are smoothed.\n"
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
            np.array([v[-1], u[-1], w[-1]]),
            grid_transform=self._raw_grid_transform,
            grid_offset=self._raw_grid_offset,
        )

        self._calc_interpolator = interpolator

        return self._calc_interpolator

    def act_interpolate(self, points: np.ndarray, is_index=False):
        if not hasattr(self, "_interpolator"):
            self.act_act_add_interpolator()
        return self._calc_interpolator.interpolate(points, is_index=is_index)

    @logging_and_warning_decorator()
    def act_visualize_disclination_lines(
        self,
        is_new: bool = True,
        is_wrap: bool = True,
        is_smooth: bool = True,
        is_extent: bool = True,
        min_line_length: Optional[int] = None,
        lines_scalars_name: Optional[str] = None,
        opts_scene=OptsScene(),
        opts_tube=OptsTube(), #!!!!!!!!!!!!!!!!!!
        opts_extent=OptsExtent(),
        logger=None,
        **kwargs,
    ):

        opts_extent.corners = self._calc_corners
        
        merge = merge_opts_all(
            {
                "scene_": opts_scene, 
                "line_": opts_tube,
                "extent_": opts_extent
            },
            kwargs, type(self).__name__)

        opts_scene = merge["scene_"]
        opts_tube = merge["line_"]
        opts_extent = merge["extent_"]

        check_bool_flags(locals())

        if min_line_length is None:
            msg = "No minimum line length has been provided for the plotted lines. "
            msg += f"Use the default value {self._raw_default_miminum_line_length_smooth}"
            logger.info(msg)
            min_line_length = self._raw_default_miminum_line_length_smooth

        if is_smooth and hasattr(self._calc_lines[0], "_calc_defect_coords_smooth_obj"):
            _min_len_length_smooth = self._calc_lines[0]._calc_defect_coords_smooth_obj.opts_min_line_length
            if _min_len_length_smooth > min_line_length:
                msg = f"The minimum line length to be plotted ({min_line_length}) is shorter than the required minimum length for smoothing ({_min_len_length_smooth}) \n"
                msg += f"Use the larger value {_min_len_length_smooth} instead."
                min_line_length = _min_len_length_smooth
                logger.warning(msg)

        logger.debug(f"min_line_length = {min_line_length}")

        lines_plot = [
            line for line in self._calc_lines if line._calc_defect_num > min_line_length
        ]

        if lines_scalars_name is not None:
            logger.info("Scalars of lines are input")
            lines_scalars = [getattr(line, lines_scalars_name) for line in lines_plot]
            lines_colors = [(1,1,1) for line in lines_plot] #!!!!!!!!!!!!!!!!!!!!!!!
            if opts_tube.color_rule is not None:
                logger.warning(
                    "scalars of lines are input. Their color_input will be ignored"
                )
        else:
            lines_scalars = [None for line in lines_plot]

        if opts_tube.color_rule == 'sample_far':   #!!!!!!!!!!!!!!!!!!!!!
            from ..general import blue_red_in_white_bg, sample_far

            color_map = blue_red_in_white_bg()
            color_map_length = np.shape(color_map)[0] - 1
            lines_colors = color_map[
                (sample_far(len(lines_plot)) * color_map_length).astype(int)
            ]
        else:
            lines_colors = [opts_tube.color_rule for line in lines_plot]

        figure = self.act_add_scene(is_new, opts=opts_scene)

        logger.debug("Start to draw disclination lines")
        for line, line_color, line_scalar in zip(
            lines_plot, lines_colors, lines_scalars
        ):
            opts_tube = replace(opts_tube, name=line.name, color=line_color)
            line_visual = line.act_visualize(
                is_wrap=is_wrap,
                is_smooth=is_smooth,
                scalars=line_scalar,
                opts=opts_tube,
                logger=logger,
            )

            figure.add_object(line_visual, category="lines")
            print(self.figs[-1].objects, 'BBBBBBBBBBBBB')

        if is_extent:
            extent = PlotExtent(opts_extent)
            figure.add_object(extent, category="extent")
            figure.scene.distance = opts_scene.distance

    @logging_and_warning_decorator()
    def act_visualize_n_in_Q(
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

        opts_extent.corners = self._calc_corners
        opts_grid.corners_limit = self._calc_corners
        
        merge = merge_opts_all(
            {
             "plane_": opts_grid,
             "n_": opts_nPlane,
             "extent_": opts_extent,
             "scene_": opts_scene
             },
            kwargs, "QFieldObject.act_visualize_n_in_Q"
            )

        opts_grid = merge["plane_"]
        opts_nPlane = merge["n_"]
        opts_extent = merge["extent_"]
        opts_scene = merge["scene_"]

        if not hasattr(self, "_calc_interpolator"):
            self.act_add_interpolator()

        opts_grid.normal = plane_normal
        opts_grid.spacing = plane_spacing
        opts_grid.size = plane_size

        check_bool_flags(locals())

        figure = self.act_add_scene(is_new, opts=opts_scene)

        nPlane = PlotnPlane(
            QInterpolator=self._calc_interpolator,
            opts_grid=opts_grid,
            opts_nPlane=opts_nPlane,
            logger=logger,
        )
        
        print(self.figs[-1].objects, 'CCCCCCCCCCCC')
        print(figure.objects, 'DDDDDDDDDDD')
        figure.add_object(nPlane, category="nPlanes")
        print(self.figs[-1].objects, 'BBBBBBBBBBBBB')

        if is_extent:
            extent = PlotExtent(opts_extent)
            figure.add_object(extent, category="extent")
            figure.scene.distance = opts_scene.distance

        if opts_scene.distance != None:
            figure.scene.distance = opts_scene.distance

    def act_add_scene(self, is_new=True, opts=OptsScene):
        figure = PlotScene(is_new=is_new, opts=opts)
        if is_new or (not is_new and len(self._calc_figures) == 0):
            self._calc_figures.append(figure)
        return figure

    def act_reset_figures(self):
        self._calc_figures = []
        
    @property
    def lines(self):
        return self._calc_lines
    
    @property
    def figs(self):
        return self._calc_figures
    
    @property
    def S(self):
        return self._raw_S

    @property
    def n(self):
        return self._raw_n

    def __call__(self) -> np.ndarray:
        return self._raw_Q
    
    
    
'''

        # --- Core actions ---
        "act_defect_detect": ["Detect defect points from director field n under specified boundary conditions.", "METHOD"],
        "act_lines_classify": ["Group defect points into disclination lines under periodic connectivity.", "METHOD"],
        "act_lines_smooth": ["Smooth disclination line trajectories using OptsSmooth options.", "METHOD"],
        "act_add_interpolator": ["Build and attach an interpolator for Q field queries at arbitrary points.", "METHOD"],
        "act_interpolate": ["Interpolate Q at given points (index or real space).", "METHOD"],

        # --- Visualization actions ---
        "act_add_scene": ["Create a PlotScene and optionally register it into internal figure list.", "METHOD"],
        "act_visualize_disclination_lines": ["Visualize disclination lines (tube rendering) in a Mayavi scene.", "METHOD"],
        "act_visualize_n_in_Q": ["Visualize director field sampled on planes via Q interpolator in a Mayavi scene.", "METHOD"],
        "act_reset_figures": ["Clear all cached figures/scenes.", "METHOD"],

        # --- Misc ---
        "__call__": ["Return the raw Q field array.", "MAGIC"]
'''
