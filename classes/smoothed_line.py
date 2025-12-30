import numpy as np
from typing import Optional, Literal
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev
from dataclasses import dataclass, field
import os
import json

from ..logging_decorator import logging_and_warning_decorator
from .opts import merge_opts_all
from ..datatypes import Number, as_Number, as_str, ColorRGB, as_ColorRGB, Vect, as_Vect, as_bool
from .visual.plot_figure import PlotFigure
from .visual.plot_tube import OptsTube, PlotTube


@dataclass(slots=True)
class OptsSmooth:
    window_ratio: Number | None = None
    window_length: Number | int = None
    order: int = 3
    N_out_ratio: Number = 3.0
    mode: Literal["interp", "wrap"] = "interp"
    min_line_length: int = 50
    name: str = "smoothed_line"
    is_window_warning: bool = True
    
    _internal_owner: object | None = field(default=None, repr=False, init=False)

    __descriptions__ = {
        "window_ratio": "window ratio for smoothing: line_length / window_length",
        "window_length": "explicit window length for smoothing",
        "order": "smoothing polynomial order",
        "N_out_ratio": "ratio between output and input #points in smoothing",
        "mode": "smoothing mode (interp or wrap)",
        "min_line_length": "minimum line length to be smoothed",
        "name": "name identifier of the line",
        "is_window_warning" : "whether present the warning when window_length and window_ratio are both input"
    }

    _validators = {
        "window_ratio": lambda self, v: (
            None
            if v is None
            else as_Number(v, name=self.__descriptions__["window_ratio"])
        ),
        "window_length": lambda self, v: (
            None
            if v is None
            else as_Number(v, name=self.__descriptions__["window_length"])
        ),
        "order": lambda self, v: as_Number(v, name=self.__descriptions__["order"], is_int=True, replace=3),
        "N_out_ratio": lambda self, v: as_Number(
            v, name=self.__descriptions__["N_out_ratio"], replace=3
        ),
        "mode": lambda self, v: (
            v
            if v in ("interp", "wrap")
            else (_ for _ in ()).throw(
                ValueError(
                    f"{self.__descriptions__['mode']} must be 'interp' or 'wrap', got {v!r}"
                )
            )
        ),
        "min_line_length": lambda self, v: as_Number(
            v, name=self.__descriptions__['min_line_length'], is_int = True
            ),
        "name": lambda self, v: as_str(v, name=self.__descriptions__["name"]),
        "is_window_warning": lambda self, v: v if isinstance(v, bool) else (_ for _ in ()).throw(
            TypeError(f"{self.__descriptions__['is_window_warning']} must be a bool, got {v}")
        )
    }

    def __setattr__(self, key, value):
        if key in self._validators:
            value = self._validators[key](self, value)
        object.__setattr__(self, key, value)
        
        if key != "_internal_owner" and hasattr(self, "_internal_owner") and self._internal_owner is not None:
            self._internal_owner.act_commit(**{key: value})
        
class SmoothingConfigError(ValueError):
    """
    Recoverable configuration error for smoothing.

    Raised only for explicitly recognized, user-fixable issues inside
    the smoothing helper (e.g., missing window length). This exception
    is intended to be caught locally and converted to RECOVERY + fallback.
    """
    pass


class SmoothedLine:
    """
    Smooth and resample a polyline using Savitzky–Golay filtering
    and parametric B-spline interpolation.

    Workflow
    --------
    1. Apply **Savitzky–Golay filter** to locally smooth the input coordinates.
    2. Perform **parametric B-spline interpolation** (`scipy.interpolate.splprep`
       with ``s=0``) on the smoothed points.
    3. Evaluate the spline at a uniformly spaced parameter grid (`splev`)
       to produce a resampled output line with higher or lower resolution.

    Parameters
    ----------
    line_coord_input : np.ndarray
        Input line coordinates of shape (N, D), where N is the number of points
        and D is the dimension (2D or 3D typically).

    opts : OptsSmooth, optional
        Options controlling the smoothing and resampling procedure.
        See :attr:`OptsSmooth.__descriptions__` for definitions.

    logger : logging.Logger, optional
        Logger instance for warnings and information messages.
        If None, falls back to global logging configuration.

    **kwargs
        Extra keyword arguments to override fields in `opts`.
        Keys must match attributes of :class:`OptsSmooth`.

    Attributes
    ----------
    See :`SmoothedLine.__descriptions__` and `SmoothOpts.__descriptions__` for 
    a full list and explanation of attributes.

    Methods
    -------
    _helper_apply_smooth(opts, logger=None)
        Internal method that performs smoothing and resampling.
        Not intended for direct user calls. Use :meth:`act_commit` or re-initialize instead.

    act_commit()
        Commit changes to options and reapply smoothing.

    act_log_parameters()
        Log or return a formatted summary of parameters and results.

    act_visualize()
        Visualize smoothed lines by points    

    Python Special Methods
    ----------------------
    - ``len(line)`` → number of output points
    - ``iter(line)`` → iterate over smoothed points
    - ``line[i]`` → get the i-th point
    - ``np.array(line)`` → convert to NumPy array of points
    - ``str(line)`` → formatted summary of parameters. (e.g., ``print(line)``)
    - ``repr(line)`` → short identifier for debugging. (e.g., just type ``line`` in an interactive shell)
    - ``with line: ...`` → context manager for safe temporary option changes

    Notes
    -----
    - If both ``window_length`` and ``window_ratio`` are provided, ``window_ratio``
      will be IGNORED. Warnings depend on the `is_window_warning` flag in :class:`OptsSmooth`. 
    """

    __descriptions__ = {
        "name": "Name identifier of this line object",

        # --- internal states ---
        "_raw_coord": "Raw input line coordinates (shape: N x D)",
        "_calc_N_init": "Number of input points (before smoothing)",
        "_calc_N_out": "Number of output points (after smoothing)",
        "_entities": "Whose first element is smoothed output coordinates (shape: M x D)",
        "_state_is_smoothed": "Boolean flag indicating whether smoothing was applied",
        "_state_status": (
            "Status indicator of the smoothing pipeline. "
            "Set to 'success' if smoothing completes normally. "
            "If smoothing is skipped or disabled due to internally detected "
            "conditions (e.g. line too short, invalid window size, "
            "or numerical failures), this field stores a human-readable "
            "string describing the specific reason."
        ),

        # ==== options mirrored onto the instance ====
        "opts": "The OptsSmooth instance that controlls smoothing options."
    }

    __slots__ = tuple(__descriptions__.keys())

    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        line_coord_input: np.ndarray,
        opts: OptsSmooth = OptsSmooth(),
        logger=None,
        **kwargs,
    ):
        
        line_coord_input = np.asarray(line_coord_input)
        if line_coord_input.ndim != 2:
            raise ValueError("line_coord_input for smoothing must be a 2D array of shape (N, D)")

        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
        object.__setattr__(opts, '_internal_owner', self)
        object.__setattr__(self, "opts", opts)

        object.__setattr__(self, "_raw_coord", line_coord_input)
        object.__setattr__(self, "_calc_N_init", len(self._raw_coord))

        object.__setattr__(self, "_state_is_smoothed", False)
        object.__setattr__(self, "_state_status", "Failure, reason unknown.")

        self._helper_apply_smooth()

    
    def _helper_fallback_no_smooth(self, reason: str) -> None:
        object.__setattr__(self, "_state_is_smoothed", False)
        object.__setattr__(self, "_entities", self._raw_coord)
        object.__setattr__(self, "_calc_N_out", self._calc_N_init)
        object.__setattr__(self, "_state_status", f"The line `{self.opts.name}` is not smoothed, reason: {reason}.")
        
    @logging_and_warning_decorator()
    def _helper_apply_smooth(self, logger=None):
                
        msg = f'Start to smooth line {self.opts.name!r} with {self._calc_N_init} points.\n'
        msg += f"window length = {self.opts.window_length}\n"
        msg += f"window ratio = {self.opts.window_ratio}\n"
        msg += f"minimum smoothed line length = {self.opts.min_line_length}"
        logger.debug(msg)
                
        if self._calc_N_init < self.opts.min_line_length:
            reason = f"the minimum length of line smoothing is set to be {self.opts.min_line_length} points, while the current line has {self._calc_N_init} points"
            logger.warning(f"{self.opts.name!r} is not smoothed, because {reason}.")
            self._helper_fallback_no_smooth(reason)
            return
        
        try:
            logger.detail("Start to determine the smoothing window length.")
            if self.opts.window_length is None:
                if self.opts.window_ratio is None:
                    raise SmoothingConfigError("No input value provided for smooth window length.")
                object.__setattr__(self.opts, 'window_length', int(self._calc_N_init / self.opts.window_ratio / 2) * 2 + 1)
                object.__setattr__(self.opts, 'window_ratio', self._calc_N_init / self.opts.window_length)
            else:
                if self.opts.window_ratio is not None and self.opts.is_window_warning == True:
                    logger.warning(
                        f"Window_length is manual input as {self.opts.window_length}. window_ratio would be ignored."
                    )     
                object.__setattr__(self.opts, 'window_ratio', self._calc_N_init / self.opts.window_length)

            if self.opts.window_length >= self._calc_N_init:
                raise SmoothingConfigError(
                    f"Filter window length {self.opts.window_length} should not be larger than line length {self._calc_N_init}"
                )
            
            if self.opts.window_length <= self.opts.order:
                raise SmoothingConfigError(
                    f"Filter window length {self.opts.window_length} should not be smaller than filter order {self.opts.order}"
                )
                
            logger.debug(f"Smoothing window length is finally chosen as {self.opts.window_length}")

            object.__setattr__(self, "_calc_N_out", int(self._calc_N_init * self.opts.N_out_ratio))
            logger.detail(f"Number of output points after smoothing is {self._calc_N_out}.")

            logger.detail('Applying Savitzky-Golay filter to smooth the curve')
            line_points = savgol_filter(
                self._raw_coord,
                self.opts.window_length,
                self.opts.order,
                axis=0,
                mode=self.opts.mode,
            )

            logger.detail('Defining spline parameter u')
            uspline = np.arange(self._calc_N_init) / self._calc_N_init

            logger.detail('Fitting and evaluate spline')
            tck = splprep(line_points.T, u=uspline, s=0)[0]
            entity = np.array(splev(np.linspace(0, 1, self._calc_N_out), tck)).T
            object.__setattr__(self, "_entities", entity)
            
            object.__setattr__(self, "_state_is_smoothed", True)
            object.__setattr__(self, "_state_status", "Success")
        
        except SmoothingConfigError:
            logger.exception("Smoothing aborted (manual check)")
            logger.recovery("Fallback applied: smoothing disabled; using raw coordinates.")
            self._helper_fallback_no_smooth(reason)

        except Exception:
            logger.exception("Smoothing aborted (system error)")
            logger.recovery("Fallback applied: smoothing disabled; using raw coordinates.")
            self._helper_fallback_no_smooth(reason)
    
    @logging_and_warning_decorator(start_finish_level=5)
    def act_commit(self, logger=None, **changes):

        if not changes:
            return
        
        for k, v in changes.items():
            try:
                if k not in self.opts.__descriptions__:
                    raise ValueError(f"Unknown attribute: {k} in class: SmoothedLine.opts")
                object.__setattr__(self.opts, k, v)
            except:
                logger.exception(f"Failed to reset value of {k!r}")
                logger.recovery("Ignore this modification")
                
        self._helper_apply_smooth()

    @logging_and_warning_decorator(start_finish_level=5)
    def act_visualize(self, 
                      Figure: PlotFigure = PlotFigure(), 
                      move: Vect(3) = (0,0,0),
                      logger=None,
                      **kwargs,
                      ):

        move = as_Vect(move, name="The replacement to move smooth line", replace=(0,0,0))

        pts = np.array(self)
        pts = pts[:, :3] + move
        PlotTube(pts, Figure, **kwargs)
        
        
    def __array__(self, dtype=None):
        arr = self._entities
        return np.asarray(arr, dtype=dtype) if dtype is not None else arr
        
        