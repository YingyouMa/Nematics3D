import numpy as np
import time
from typing import Union
from dataclasses import replace, dataclass, field, fields
from pyvistaqt import BackgroundPlotter
import pyvista as pv

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
from .QInterpolator import QInterpolator
from .visual.plot_tube import OptsTube
from .visual.plot_rod import OptsRod
from .visual.plot_sphere import OptsSphere
from .visual.plot_surface import OptsSurface
from .visual.plot_figure import PlotFigure, OptsFigure
from .Q_plane import QPlane, QPlanePolar
from .visual.figure_manager import FigureManager
from .plane_grid import OptsPlaneGrid
from .plane_grid_polar import OptsPlaneGridPolar
from .bounds import as_bounds
from .opts import merge_opts_all, cover_value
from ..general import get_box_corners
from .smoothed_line import OptsSmooth
from .registry_base import RegistryBase
from .disclination_line import DisclinationLine, DisclinationLineSmooth
from .class_base import ClassBase


@dataclass(slots=True)
class InputQ:
    """
    Validated input bundle for initializing a `QFieldObject`.

    At least one field description must be provided:

    - provide `Q`, or
    - provide `n`, optionally together with `S`.

    If `n` is provided while `S` is omitted, `S=1` is used everywhere.
    If both `Q` and `n` are provided, `n`/`S` take priority and `Q` is ignored.

    Parameters
    ----------
    Q
        Q-tensor field on the lattice. Compatible input representations are
        accepted and normalized to the internal `QField5` representation.
    S
        Scalar order parameter field with shape matching `n.shape[:3]`.
        Used together with `n` to reconstruct `Q`.
    n
        Director field with shape `(..., 3)`. Used to reconstruct `Q` when a
        raw Q-tensor field is not supplied or should be overridden.
    box_periodic_flag
        Periodic-boundary-condition flags for the three lattice directions.
    grid_offset
        Translation offset that maps lattice indices to real-space coordinates.
    grid_transform
        3x3 linear transform that maps lattice indices to real-space
        coordinates.
    default_miminum_line_length_smooth
        Default minimum disclination-line length required for smoothing.
    default_smooth_window_length
        Default smoothing window length used for line smoothing.
    default_miminum_line_length_visual
        Default minimum disclination-line length required for visualization.
    """

    Q: Union[QField5, QField9] | Unset = UNSET
    S: SField | Unset = UNSET
    n: nField | Unset = UNSET
    box_periodic_flag: DimensionFlagInput = False
    grid_offset: Vect(3) = (0, 0, 0)
    grid_transform: Tensor((3, 3)) = field(default_factory=lambda: np.eye(3))
    default_miminum_line_length_smooth: Number = 61
    default_smooth_window_length: Number = 41
    default_miminum_line_length_visual: Number = 75

    __attrs__ = {
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

    # ==================== OVERRIDE ====================
    # InputQ overrides dataclass assignment so every field stays validated both
    # during initialization and during later interactive edits.
    # ==================================================
    def __setattr__(self, key, value):
        if key in self._validators:
            if value is not UNSET:
                desc = f"{key!r}: {self.__class__.__attrs__[key]}"
                value = self._validators[key](value, desc)
        object.__setattr__(self, key, value)


class QFieldObject(ClassBase):
    """
    QFieldObject stores a Q-tensor field together with derived geometry,
    detected defects, disclination lines, and common visualization helpers.

    Important readable attributes:

    - `name`: identity of this Q field object.
    - `S`: scalar-order field derived from or paired with the Q data.
    - `n`: director field derived from or paired with the Q data.
    - `lines`: classified disclination lines registered under this Q field.
    - `figures` / `figs`: FigureManager storing figures created from this Q field.
    - `objects` / `objs`: RegistryBase storing physical objects derived from this Q field.
    - `interpolator`: QInterpolator used for off-grid sampling.
    - `_calc_grid`: full real-space lattice coordinates of the Q field.
    - `_calc_corners`: Bounds object describing the Q-field box.
    - `_calc_defect_indices` / `_calc_defect_grid`: detected defect positions in index and world coordinates.

    Common inspection helpers:

    - `show_getattrs()`: show the main readable Q-field attributes.
    - `show_attr_desc(name)`: describe a specific readable attribute.
    - `show_relations()`: show bound figures, object registry, and interpolator.
    - `show_relation_tree()`: show how this Q field connects to derived objects.

    Common user actions:

    - `act_defect_detect()`: detect defect points from the director field.
    - `act_lines_classify()`: classify detected defects into disclination lines.
    - `act_lines_smooth(...)`: smooth eligible classified lines.
    - `act_add_interpolator()`: create and bind a QInterpolator if absent.
    - `act_interpolate(points, ...)`: interpolate the Q field at arbitrary points.
    - `act_visualize_disclination_lines(...)`: draw disclination lines on a figure.
    - `act_visualize_n_plane(...)`: create a Cartesian director analysis plane.
    - `act_visualize_S_plane(...)`: create a Cartesian scalar-order analysis plane.
    - `act_visualize_n_near_defect(...)`: create a polar director analysis plane around a smoothed line.

    Representation:

    - `str(obj)` returns the short ClassBase-style identity.
    - `repr(obj)` returns the compact ClassBase summary.
    """

    __attrs__ = {
        # --- Identity ---
        "raw_name": "Name identifier of this Q tensor object.",
        # --- Raw inputs ---
        "_raw_Q": "Raw Q-tensor field on lattice. Typically QField5 or QField9 (shape: (Nx, Ny, Nz, ...)).",
        "_raw_S": "Raw scalar order parameter field S on lattice (shape: (Nx, Ny, Nz)).",
        "_raw_n": "Raw director field n on lattice (shape: (Nx, Ny, Nz, 3)).",
        "_raw_box_periodic_flag": "Per-dimension periodic boundary condition flags (bool array-like of length 3).",
        "_raw_grid_offset": "A 3D vector, as the grid translation offset mapping lattice indices -> real-space coordinates.",
        "_raw_grid_transform": "A 3x3 tensor, as the linear transform mapping lattice indices -> real-space coordinates",
        # --- consts / thresholds ---
        "default_miminum_line_length_smooth": "Default minimum line length (#points) required to apply smoothing.",
        "default_smooth_window_length": "Default smoothing window length (#points) used when not specified.",
        "default_miminum_line_length_visual": "Default minimum line length (#points) required for visualization.",
        "default_cross_line_padding_num_": "Default number of points padded for smoothing cross-type disclination line.",
        # --- Derived grids / geometry ---
        "_calc_grid_index": "Lattice coordinate grid in index space (before applying transform/offset).",
        "_calc_grid": "Coordinate grid in real space after applying grid_transform and grid_offset.",
        "_calc_corners_index": "Box corners in lattice-index space.",
        "_calc_corners": "Bounds object describing the Q-field box in real-space coordinates.",
        "_calc_box_size_periodic_index": (
            "Effective periodic box size in index units. "
            "For periodic dims equals grid size, otherwise inf."
        ),
        "_calc_box_size_periodic_coord": "Effective periodic box size in real-space coordinates.",
        # --- Defects / disclinations  ---
        "_calc_defect_indices": "Indices (lattice coordinates) of detected defect points.",
        "_calc_defect_grid": "Real-space coordinates of detected defect points.",
    }
    __relations__ = {
        **(ClassBase.__relations__),
        "figures": "FigureManager object that manages PlotFigure objects created for this Q field.",
        "objects": "RegistryBase object that manages physical objects related to this Q field.",
        "interpolator": "The QInterpolator object associated with this Q field.",
    }
    __properties__ = {
        **(ClassBase.__properties__),
        "S": "Read-only: Scalar order parameter field. Alias of `_raw_S`.",
        "n": "Read-only: Director field. Alias of `_raw_n`.",
        "lines": "Read-only: Classified disclination lines.",
        "figs": "Read-only: Visualization figures. Alias of `figures`.",
        "objs": "Read-only: Physical objects. Alias of `objects`.",
    }
    __slots__ = tuple(k for k in __attrs__.keys() if k not in ClassBase.__slots__)

    # -------------------------------
    # Initialization
    # -------------------------------

    # ==================== OVERRIDE ====================
    # QFieldObject overrides ClassBase.__init__ because it must normalize input
    # Q/S/n data, derive geometry caches, optionally detect defects and lines,
    # and create the default interpolator and figure/object managers.
    # ==================================================
    @logging_and_warning_decorator()
    def __init__(
        self,
        is_detect_defects: bool = True,
        is_classify_lines: bool = True,
        inputValue: InputQ | None = None,
        name: str = "Q",
        logger=None,
        **kwargs,
    ) -> None:

        super().__init__(name=name, name_replace="Q")
        if inputValue is None:
            inputValue = InputQ()

        inputValue = merge_opts_all({"": inputValue}, kwargs, type(self).__name__)[""]
        for f in fields(inputValue):
            k = f.name
            v = getattr(inputValue, k)
            if k.startswith("default"):
                object.__setattr__(self, k, v)
            else:
                object.__setattr__(self, f"_raw_{k}", v)

        objects = RegistryBase(
            "objects manager",
            info=f"physical objects attached to Q field {self.name!r}",
        )
        self.act_bind_relation_base("objects", objects, is_weak=False)
        objects.act_bind_relation_base("owner", self, is_weak=True)

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
        corners_coord = apply_linear_transform(
            corners_index,
            transform=self._raw_grid_transform,
            offset=self._raw_grid_offset,
        )
        bounds = as_bounds(corners_coord, name=f"Bounds of Q field {self.name!r}")

        object.__setattr__(self, "_calc_corners_index", corners_index)
        object.__setattr__(self, "_calc_corners", bounds)
        self.objs.act_register(bounds, is_contain_ok=True)
        logger.debug(
            f"Box corners in lattice-index units is {self._calc_corners_index}."
            f"Box bounds in real-space coordinates is {self._calc_corners}."
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

        figures = FigureManager()
        self.act_bind_relation_base("figures", figures, is_weak=False)
        figures.act_bind_relation_base("owner", self, is_weak=True)

    # -------------------------------
    # Defect and line analysis
    # -------------------------------
    @logging_and_warning_decorator(start_finish_level=5)
    def act_defect_detect(self, logger=None):
        """
        Detect defect points from the current director field.

        This updates both `_calc_defect_indices` in lattice-index coordinates
        and `_calc_defect_grid` in real-space coordinates using the current
        grid transform and offset.
        """
        object.__setattr__(
            self,
            "_calc_defect_indices",
            defect_detect(
                self._raw_n,
                is_boundary_periodic=self._raw_box_periodic_flag,
            ),
        )
        logger.info(f"{len(self._calc_defect_indices)} defects are found.")

        object.__setattr__(
            self,
            "_calc_defect_grid",
            apply_linear_transform(
                self._calc_defect_indices,
                transform=self._raw_grid_transform,
                offset=self._raw_grid_offset,
            ),
        )

    @logging_and_warning_decorator(start_finish_level=5)
    def act_lines_classify(self, logger=None):
        """
        Classify detected defect points into disclination lines.

        The classified lines are sorted by defect count, renamed in display
        order, registered into `self.objects`, and returned as a list.
        """
        lines = defect_classify_into_lines(
            self._calc_defect_indices,
            box_size_periodic=self._calc_box_size_periodic_index,
            grid_offset=self._raw_grid_offset,
            grid_transform=self._raw_grid_transform,
        )
        lines = sorted(lines, key=lambda line: line._calc_defect_num, reverse=True)
        for i, line in enumerate(lines):
            line.name = f"disclination line {i}"
            self.objects.act_register(line)

        logger.info(f"{len(lines)} lines are found.")

        return lines

    @logging_and_warning_decorator()
    def act_lines_smooth(
        self,
        opts: OptsSmooth | None = None,
        logger=None,
        **kwargs,
    ):
        """
        Smooth eligible disclination lines using shared smoothing options.

        Lines shorter than the configured minimum length are skipped. Missing
        smoothing defaults are filled from the Q-field object before
        delegating the actual smoothing to each line.

        Parameters
        ----------
        opts
            Base `OptsSmooth` configuration applied to all candidate lines.
        **kwargs
            Keyword overrides merged into `opts` before smoothing. Supported
            keys are the fields of `OptsSmooth`, including commonly used
            options such as `window_length`, `window_ratio`,
            `min_line_length`, and `order`.

        Notes
        -----
        If `min_line_length` is not provided, the method uses
        `self.default_miminum_line_length_smooth`.

        If both `window_length` and `window_ratio` are omitted, the method
        uses `self.default_smooth_window_length` as the default window length.

        If both `window_length` and `window_ratio` are provided,
        `window_length` takes priority and `window_ratio` is ignored.

        Examples
        --------
        Smooth all eligible lines with the object defaults::

            q.act_lines_smooth()

        Smooth using an explicit window length::

            q.act_lines_smooth(window_length=31)

        Smooth only sufficiently long lines::

            q.act_lines_smooth(min_line_length=100, window_ratio=8)

        See Also
        --------
        OptsSmooth
            Full smoothing-option container used by each line.
        """
        if opts is None:
            opts = OptsSmooth()

        opts = merge_opts_all({"": opts}, kwargs, "SmoothedLine")[""]

        if opts.min_line_length is UNSET:
            opts.min_line_length = self.default_miminum_line_length_smooth
            msg = "No input value provided for minimum smoothed line length. \n"
            msg += f"Using the default value self.default_miminum_line_length_smooth={self.default_miminum_line_length_smooth}."
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
                line.act_smooth(opts=opts, is_window_warning=False)
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
        """Create and bind a `QInterpolator` if one is not already present."""
        interpolator_old = self.interpolator
        if isinstance(interpolator_old, QInterpolator):
            return interpolator_old

        interpolator = QInterpolator(self, name=f"{self.name} interpolator")
        self.act_bind_relation_base("interpolator", interpolator, is_weak=False)

        return self.interpolator

    def act_interpolate(self, points: np.ndarray, is_index=False):
        """
        Interpolate the Q field at arbitrary sample points.

        Parameters
        ----------
        points
            Sample positions where the Q field should be evaluated.
        is_index
            If False, `points` are interpreted in real-space coordinates.
            If True, `points` are interpreted in lattice-index coordinates
            before interpolation.
        """
        if self.interpolator is None:
            self.act_add_interpolator()
        return self.interpolator.interpolate(points, is_index=is_index)

    # -------------------------------
    # Visualization helpers
    # -------------------------------

    @logging_and_warning_decorator()
    def _helper_set_figure(
        self,
        is_new: bool,
        figure: PlotFigure | BackgroundPlotter | pv.Plotter | str | int | None,
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
                    active_name = getattr(self.figs, "_state_active_name", None)
                    if active_name is not None:
                        figure_active = self.figs[active_name]
                        if figure_active.is_alive:
                            figure = figure_active
                            figure.act_commit(opts_figure)
                        else:
                            figure = PlotFigure(opts=opts_figure, name=title)
                    elif len(self.figs) == 1 and self.figs[0].is_alive:
                        figure = self.figs[0]
                        figure.act_commit(opts_figure)
                    else:
                        figure = PlotFigure(opts=opts_figure, name=title)
                elif isinstance(figure, PlotFigure):
                    figure.act_commit(opts_figure)
                elif isinstance(figure, (BackgroundPlotter, pv.Plotter)):
                    figure = PlotFigure(plotter=figure, opts=opts_figure, name=title)
                else:
                    raise ValueError(
                        "`figure` input must be either index in FigureManager (str or int) "
                        "or a valid PlotFigure object, or a valid pyvistaqt BackgroundPlotter object, "
                        "or None (creating a new figure) "
                        f"Got type {type(figure)!r} instead."
                    )
            except Exception:
                logger.exception("Could not find figure in FigureManager.")
                logger.recovery("Create a new figure instead.")
                figure = PlotFigure(opts=opts_figure, name=title)

        if figure.name.startswith(figure._DEFAULT_NAME):
            figure.act_set_name(title)
        self.figs.act_register(figure, is_contain_ok=True)
        self.figs.act_set_active(figure.name)

        return figure

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_resolve_visual_bounds(
        self, bounds=None, *, label: str = "plot", logger=None
    ):
        if bounds is None:
            return self._calc_corners

        try:
            bounds_obj = as_bounds(bounds, name=f"{label} bounds")
        except Exception:
            logger.exception("Check input.")
            logger.recovery("Use the default Q bounds instead.")
            return self._calc_corners

        return bounds_obj

    @logging_and_warning_decorator()
    def act_visualize_disclination_lines(
        self,
        figure: PlotFigure | BackgroundPlotter | pv.Plotter | str | int | None = None,
        is_new: bool = False,
        is_wrap: bool = True,
        is_smooth: bool = True,
        is_extent: bool = True,
        min_line_length: int | None = None,
        opts_figure: OptsFigure | None = None,
        opts_line: OptsTube | None = None,
        opts_extent: OptsTube | None = None,
        bounds=None,
        title: str = "disclination lines",
        logger=None,
        **kwargs,
    ):
        """
        Visualize classified disclination lines on a figure.

        Parameters
        ----------
        figure
            Target figure, plotter, registered figure name/index, or `None`.
        is_new
            If True, always create a new figure instead of reusing an existing
            one.
        is_wrap
            Whether periodic wrapping should be applied before plotting each
            line.
        is_smooth
            Whether smoothed line geometry should be used when available.
        is_extent
            Whether to also draw the bounding extent.
        min_line_length
            Minimum defect count required for a line to be plotted. If not
            provided, `self.default_miminum_line_length_visual` is used.
        opts_figure
            Base `OptsFigure` configuration for the target figure.
        opts_line
            Base `OptsTube` configuration for the plotted lines.
        opts_extent
            Base `OptsTube` configuration for the optional bounding extent.
        bounds
            Bounds used for visualization and optional clipping. If omitted,
            the default Q-field bounds are used.
        title
            Title used when a new figure is created.
        **kwargs
            Keyword overrides merged into `opts_figure`, `opts_line`, and
            `opts_extent` using the prefixes `figure_`, `line_`, and
            `extent_`.

        Examples
        --------
        Plot lines with the default visualization settings::

            q.act_visualize_disclination_lines()

        Plot only longer lines on a new figure::

            q.act_visualize_disclination_lines(is_new=True, min_line_length=100)

        Override line and extent options through keyword prefixes::

            q.act_visualize_disclination_lines(
                line_radius=0.8,
                extent_color=(0, 0, 0),
            )
        """
        if opts_extent is None:
            opts_extent = OptsTube()
        if opts_figure is None:
            opts_figure = OptsFigure()
        if opts_line is None:
            opts_line = OptsTube(color="sample_far")

        merge = merge_opts_all(
            {"figure_": opts_figure, "line_": opts_line, "extent_": opts_extent},
            kwargs,
            type(self).__name__,
        )

        opts_figure = merge["figure_"]
        opts_line = merge["line_"]
        opts_extent = merge["extent_"]

        check_bool_flags(locals())

        figure = self._helper_set_figure(is_new, figure, opts_figure, title)
        bounds_input = bounds
        bounds = self._helper_resolve_visual_bounds(bounds, label=title)
        line_bounds = None if not is_wrap and bounds_input is None else bounds

        if min_line_length is None:
            logger.info(
                "No minimum line length has been provided for the plotted lines. "
                f"Use the default value {self.default_miminum_line_length_visual}"
            )
            min_line_length = self.default_miminum_line_length_visual

        logger.debug(f"min_line_length = {min_line_length}")

        lines_plot = [
            line for line in self.lines if line._calc_defect_num >= min_line_length
        ]

        if opts_line.color == "sample_far":
            from ..general import blue_red_in_white_bg, sample_far

            color_map = blue_red_in_white_bg()
            color_map_length = np.shape(color_map)[0] - 1
            lines_colors = color_map[
                (sample_far(len(lines_plot)) * color_map_length).astype(int)
            ]
        else:
            lines_colors = [opts_line.color for line in lines_plot]

        logger.debug("Start to draw disclination lines")
        for line, line_color in zip(lines_plot, lines_colors):
            opts_line = replace(opts_line, color=line_color)
            line.act_visualize(
                figure=figure,
                is_wrap=is_wrap,
                is_smooth=is_smooth,
                bounds=line_bounds,
                opts=opts_line,
            )

        if is_extent:
            bounds.act_visualize(
                figure=figure,
                opts=opts_extent,
                is_reset_camera=False,
            )

    @logging_and_warning_decorator()
    def act_visualize_n_plane(
        self,
        figure: PlotFigure | BackgroundPlotter | pv.Plotter | str | int | None = None,
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
        bounds=None,
        title: str = "visualization of n plane",
        name_plane: str = "n-plane",
        logger=None,
        **kwargs,
    ):
        """
        Visualize the director field on a Cartesian analysis plane.

        This creates a `QPlane` from the current Q-field interpolator, then
        renders directors in bulk and near-defect regions on the target plane.

        Parameters
        ----------
        figure
            Target figure, plotter, registered figure name/index, or `None`.
        is_new
            If True, always create a new figure instead of reusing an existing
            one.
        is_extent
            Whether to also draw the bounding extent.
        is_defect
            Whether detected defect points on the plane should be visible.
        opts_grid
            Base `OptsPlaneGrid` configuration for constructing the analysis
            plane.
        opts_n
            Shared base `OptsRod` configuration copied into both bulk and
            near-defect director visuals unless those visuals override it.
        opts_nb
            `OptsRod` overrides for directors in bulk regions.
        opts_nd
            `OptsRod` overrides for directors near detected defects.
        opts_figure
            Base `OptsFigure` configuration for the target figure.
        opts_extent
            Base `OptsTube` configuration for the optional bounding extent.
        opts_defect
            `OptsSphere` configuration for defect-point markers.
        bounds
            Bounds used for the plane construction and optional extent drawing.
            If omitted, the default Q-field bounds are used.
        title
            Title used when a new figure is created.
        name_plane
            Name assigned to the generated `QPlane` object.
        **kwargs
            Keyword overrides merged into the option objects using the prefixes
            `figure_`, `grid_`, `extent_`, `n_`, `nb_`, `nd_`, and `defect_`.

        Examples
        --------
        Visualize the director plane with default settings::

            q.act_visualize_n_plane()

        Show defect markers on a new figure::

            q.act_visualize_n_plane(is_new=True, is_defect=True)

        Override plane-grid and director options through keyword prefixes::

            q.act_visualize_n_plane(
                grid_spacing=2,
                n_radius=0.4,
                defect_radius=1.5,
            )
        """
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
        bounds = self._helper_resolve_visual_bounds(bounds, label=title)

        if self.interpolator is None:
            self.act_add_interpolator()

        n_plane = QPlane(
            self.interpolator,
            name=name_plane,
            opts=opts_grid,
            bounds=bounds,
            opts_defaults_override={
                "size": 1.8 * np.max(self.S.shape),
                "spacing": 1,
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
            bounds.act_visualize(
                figure=figure,
                opts=opts_extent,
                is_reset_camera=False,
            )

    @logging_and_warning_decorator()
    def act_visualize_S_plane(
        self,
        figure: PlotFigure | BackgroundPlotter | pv.Plotter | str | int | None = None,
        is_new: bool = False,
        is_extent: bool = True,
        opts_grid: OptsPlaneGrid | None = None,
        opts_S: OptsSurface | None = None,
        opts_figure: OptsFigure | None = None,
        opts_extent: OptsTube | None = None,
        bounds=None,
        title: str = "visualization of S plane",
        name_plane: str = "S-plane",
        logger=None,
        **kwargs,
    ):
        """
        Visualize the scalar order parameter on a Cartesian analysis plane.

        This creates a `QPlane` from the current Q-field interpolator, then
        renders the plane as an `S` surface on the target figure.

        Parameters
        ----------
        figure
            Target figure, plotter, registered figure name/index, or `None`.
        is_new
            If True, always create a new figure instead of reusing an existing
            one.
        is_extent
            Whether to also draw the bounding extent.
        opts_grid
            Base `OptsPlaneGrid` configuration for constructing the analysis
            plane.
        opts_S
            Base `OptsSurface` configuration for the rendered scalar-order
            surface.
        opts_figure
            Base `OptsFigure` configuration for the target figure.
        opts_extent
            Base `OptsTube` configuration for the optional bounding extent.
        bounds
            Bounds used for the plane construction and optional extent drawing.
            If omitted, the default Q-field bounds are used.
        title
            Title used when a new figure is created.
        name_plane
            Name assigned to the generated `QPlane` object.
        **kwargs
            Keyword overrides merged into the option objects using the prefixes
            `figure_`, `grid_`, `extent_`, and `S_`.

        Examples
        --------
        Visualize the scalar-order plane with default settings::

            q.act_visualize_S_plane()

        Create the plane on a new figure without the extent box::

            q.act_visualize_S_plane(is_new=True, is_extent=False)

        Override plane-grid and surface options through keyword prefixes::

            q.act_visualize_S_plane(
                grid_spacing=2,
                S_opacity=0.8,
            )
        """
        if opts_grid is None:
            opts_grid = OptsPlaneGrid()
        if opts_extent is None:
            opts_extent = OptsTube()
        if opts_figure is None:
            opts_figure = OptsFigure()
        if opts_S is None:
            opts_S = OptsSurface()

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
        bounds = self._helper_resolve_visual_bounds(bounds, label=title)

        if self.interpolator is None:
            self.act_add_interpolator()

        S_plane = QPlane(
            self.interpolator,
            name=name_plane,
            opts=opts_grid,
            bounds=bounds,
            opts_defaults_override={
                "size": 1.8 * np.max(self.S.shape),
                "spacing": 1,
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
            bounds.act_visualize(
                figure=figure,
                opts=opts_extent,
                is_reset_camera=False,
            )

    @logging_and_warning_decorator()
    def act_visualize_n_near_defect(
        self,
        x_param: float,
        smooth: DisclinationLineSmooth,
        figure: PlotFigure | BackgroundPlotter | pv.Plotter | str | int | None = None,
        is_new: bool = False,
        is_extent: bool = False,
        opts_grid: OptsPlaneGridPolar | None = None,
        opts_n: OptsRod | None = None,
        opts_nb: OptsRod | None = None,
        opts_nd: OptsRod | None = None,
        opts_figure: OptsFigure | None = None,
        opts_extent: OptsTube | None = None,
        opts_defect: OptsSphere | None = None,
        bounds=None,
        title: str = "visualization of n near defect",
        plane_name: str | None = None,
        logger=None,
        **kwargs,
    ):
        """
        Visualize the director field on a polar cross-section around a defect.

        This creates a `DefectSectionGrid` from the given smoothed
        disclination line, wraps it as a `QPlanePolar`, and renders the
        director field near the selected cross-section.

        Parameters
        ----------
        x_param
            Parametric position along the smoothed disclination line used to
            choose the cross-section.
        smooth
            Smoothed disclination line supplying the local cross-section frame.
        figure
            Target figure, plotter, registered figure name/index, or `None`.
        is_new
            If True, always create a new figure instead of reusing an existing
            one.
        is_extent
            Whether to also draw the bounding extent.
        opts_grid
            Base `OptsPlaneGridPolar` configuration for constructing the polar
            cross-section grid.
        opts_n
            Shared base `OptsRod` configuration copied into both bulk and
            near-defect director visuals unless those visuals override it.
        opts_nb
            `OptsRod` overrides for directors in bulk regions.
        opts_nd
            `OptsRod` overrides for directors near detected defects.
        opts_figure
            Base `OptsFigure` configuration for the target figure.
        opts_extent
            Base `OptsTube` configuration for the optional bounding extent.
        opts_defect
            Reserved `OptsSphere` configuration for defect-point markers.
        bounds
            Bounds used for cross-section construction and optional extent
            drawing. If omitted, the default Q-field bounds are used.
        title
            Title used when a new figure is created.
        plane_name
            Name assigned to the generated polar plane object.
        **kwargs
            Keyword overrides merged into the option objects using the prefixes
            `figure_`, `grid_`, `extent_`, `n_`, `nb_`, `nd_`, and `defect_`.

        Examples
        --------
        Visualize the director field near the middle of a smoothed line::

            q.act_visualize_n_near_defect(0.5, smooth_line)

        Create a new figure and override polar-grid settings::

            q.act_visualize_n_near_defect(
                0.25,
                smooth_line,
                is_new=True,
                grid_N_r=40,
                grid_N_theta=80,
            )
        """

        if opts_grid is None:
            opts_grid = OptsPlaneGridPolar()
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
        bounds = self._helper_resolve_visual_bounds(bounds, label=title)

        if self.interpolator is None:
            self.act_add_interpolator()

        section = smooth.act_cross_section(
            x_param,
            opts_grid=opts_grid,
            name=plane_name,
            bounds=bounds,
        )
        plane_grid = section.wrapped
        n_plane_name = section.name + " of " + smooth.name
        n_plane = QPlanePolar(
            self.interpolator,
            name=n_plane_name,
            grid=plane_grid,
        )
        self.objs.act_register(n_plane)

        n_plane.act_visualize_n(
            figure=figure,
            opts_nb=opts_nb,
            opts_nd=opts_nd,
        )

        if is_extent:
            bounds.act_visualize(
                figure=figure,
                opts=opts_extent,
                is_reset_camera=False,
            )

    # -------------------------------
    # Readable properties and array-style access
    # -------------------------------

    @property
    def lines(self):
        result = [item for item in self.objects if isinstance(item, DisclinationLine)]
        return result

    @property
    def figs(self):
        return self.figures

    @property
    def objs(self):
        return self.objects

    @property
    def S(self):
        return self._raw_S

    @property
    def n(self):
        return self._raw_n

    def __call__(self) -> np.ndarray:
        return self._raw_Q
