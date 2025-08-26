import numpy as np
from typing import Optional, Literal
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev
from dataclasses import dataclass, asdict

from ..logging_decorator import logging_and_warning_decorator
from .opts import merge_opts
from ..datatypes import Number, as_Number, as_str


@dataclass(slots=True)
class OptsSmoothen:
    window_ratio: Optional[Number] = None
    window_length: Optional[Number] = 41
    order: Number = 3
    N_out_ratio: Number = 3.0
    mode: Literal["interp", "wrap"] = "interp"
    min_line_length: int = 50
    name: str = "None"
    is_window_warning: bool = True

    __descriptions__ = {
        "window_ratio": "window ratio for smoothening: line_length / window_length",
        "window_length": "explicit window length for smoothening",
        "order": "smoothing polynomial order",
        "N_out_ratio": "ratio between output and input #points in smoothening",
        "mode": "smoothing mode (interp or wrap)",
        "min_line_length": "minimum line length to be smoothened",
        "name": "name identifier of smoothen options",
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
        "order": lambda self, v: as_Number(v, name=self.__descriptions__["order"]),
        "N_out_ratio": lambda self, v: as_Number(
            v, name=self.__descriptions__["N_out_ratio"]
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
        "min_line_length": lambda self, v: (
            v
            if isinstance(v, int)
            else (_ for _ in ()).throw(
                TypeError(
                    f"{self.__descriptions__['min_line_length']} must be int, got {type(v)}"
                )
            )
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



class SmoothenedLine:
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

    opts : OptsSmoothen, optional
        Options controlling the smoothing and resampling procedure.
        See :attr:`OptsSmoothen.__descriptions__` for definitions.

    logger : logging.Logger, optional
        Logger instance for warnings and information messages.
        If None, falls back to global logging configuration.

    **kwargs
        Extra keyword arguments to override fields in `opts`.
        Keys must match attributes of :class:`OptsSmoothen`.

    Attributes
    ----------
    See :attr:`SmoothenedLine.__descriptions__` for a full list and explanation
    of attributes (including both internal state such as ``_raw_coord`` and
    mirrored options such as ``opts_window_length``).

    Methods
    -------
    _helper_apply_smoothen(opts, logger=None)
        Internal method that performs smoothing and resampling.
        Not intended for direct user calls. Use :meth:`act_commit` or re-initialize instead.

    act_commit(logger=None, **changes)
        Commit changes to options and reapply smoothing.

    log_parameters(is_return=False, logger=None)
        Log or return a formatted summary of parameters and results.

    output : np.ndarray
        Property returning the smoothened output line.

    input : np.ndarray
        Property returning the raw input coordinates.

    Notes
    -----
    - If both ``window_length`` and ``window_ratio`` are provided, ``window_ratio``
      will be IGNORED. Warnings depend on the `is_window_warning` flag in :class:`OptsSmoothen`.
    """

    __descriptions__ = {
        "name": "Name identifier of this line object",
        "_raw_coord": "Raw input line coordinates (shape: N x D)",
        "_calc_N_init": "Number of input points (before smoothing)",
        "_calc_N_out": "Number of output points (after smoothing)",
        "_entities": "Whose first element is smoothed output coordinates (shape: M x D)",
        "_state_is_smoothened": "Boolean flag indicating whether smoothing was applied",

        # ==== options mirrored onto the instance ====
        "opts_window_ratio": "Ratio used to compute window_length if not explicitly provided",
        "opts_window_length": "Explicit smoothing window length (overrides window_ratio if set)",
        "opts_order": "Polynomial order of Savitzky–Golay filter",
        "opts_N_out_ratio": "Ratio between output and input number of points",
        "opts_mode": "Smoothing mode (either 'interp' or 'wrap')",
        "opts_min_line_length": "Minimum line length required to apply smoothing",
        "opts_is_window_warning": "Whether present the warning when window_length and window_ratio are both input",
        "_opts_all": "The dataclass project to store all options values"
    }

    __slots__ = tuple(__descriptions__.keys())

    @logging_and_warning_decorator()
    def __init__(
        self,
        line_coord_input: np.ndarray,
        opts: OptsSmoothen = OptsSmoothen(),
        logger=None,
        **kwargs,
    ):

        opts = merge_opts(opts, kwargs, prefix="")
        self._opts_all = opts

        self._raw_coord = line_coord_input
        self._calc_N_init = len(self._raw_coord)

        self._helper_apply_smoothen(self._opts_all, logger=logger)

    @logging_and_warning_decorator()
    def _helper_apply_smoothen(self, opts, logger=None):

        for k, v in asdict(opts).items():
            if k == "name":
                setattr(self, "name", v)
            else:
                setattr(self, f"opts_{k}", v)

        if len(self._raw_coord) < self.opts_min_line_length:
            self._state_is_smoothened = False
            logger.warning(
                f"{self.name} is not smoothened, because its length {self._raw_coord} is shorter than the minum length {self.opts_min_line_length}."
            )
            self._entities = [self._raw_coord]
        else:

            self._state_is_smoothened = True

            if self.opts_window_length is None:
                if self.opts_window_ratio is None:
                    raise ValueError("No input for smoothing window length!")
                self.opts_window_length = (
                    int(self._calc_N_init / self.opts_window_ratio / 2) * 2 + 1
                )
                self.opts_window_ratio = self._calc_N_init / self.opts_window_length
            else:
                if self.opts_window_ratio is not None and self.opts_is_window_warning == True:
                    logger.warning(
                        f"Window_length is manual input as {self.opts_window_length}. window_ratio would be ignored."
                    )     
                self.opts_window_length = self.opts_window_length
                self.opts_window_ratio = self._calc_N_init / self.opts_window_length

            self._calc_N_out = int(self._calc_N_init * self.opts_N_out_ratio)

            # Step 1: Apply Savitzky-Golay filter to smoothen the curve
            line_length = self._calc_N_init
            if self.opts_window_length >= line_length:
                raise ValueError(
                    f"Filter window size {len(self.opts_window_length)} must be smaller than line length {line_length}"
                )
            line_points = savgol_filter(
                self._raw_coord,
                self.opts_window_length,
                self.opts_order,
                axis=0,
                mode=self.opts_mode,
            )

            # Step 2: Define spline parameter u
            uspline = np.arange(self._calc_N_init) / self._calc_N_init

            # Step 3: Fit and evaluate spline
            tck = splprep(line_points.T, u=uspline, s=0)[0]
            self._entities = [
                np.array(splev(np.linspace(0, 1, self._calc_N_out), tck)).T
            ]

    @logging_and_warning_decorator()
    def act_commit(self, logger=None, **changes):

        if not changes:
            return
        
        for k, v in changes.items():
            setattr(self._opts_all, k, v)

        self._helper_apply_smoothen(self._opts_all, logger=logger)



    @logging_and_warning_decorator()
    def log_parameters(self, is_return: bool = False, logger=None) -> None:
        """
        Log internal filter and output parameters for inspection.

        This is the standard logging interface used in this library, which
        can be redirected to console or to a file depending on the logger
        configuration and the behavior of ``logging_and_warning_decorator``.

        All attributes listed in ``__descriptions__`` are included,
        formatted in a single log entry with a clear separator.
        """
        lines = []
        lines.append("-------------- SmoothenLine Parameters --------------")

        if self._state_is_smoothened:
            lines.append(f"[{self.name}] smoothing parameters and results:")
            for attr in self.__slots__:
                desc = self.__descriptions__.get(attr, "(no description)")
                value = getattr(self, attr, None)

                # 针对 window_length 和 window_ratio 特殊处理
                if attr in ("opts_window_length", "opts_window_ratio"):
                    lines.append(f"  {attr}: {value!r}  # {desc} (derived final value)")
                elif attr == "is_window_warning":
                    pass
                else:
                    lines.append(f"  {attr}: {value!r}  # {desc}")
        else:
            lines.append(
                f"[{self.name}] is not smoothened, because its length "
                f"{len(self._raw_coord)} < minimum required {self.opts_min_line_length}."
            )

        lines.append("-----------------------------------------------------")

        msg = "\n".join(lines)

        if is_return:
            return msg
        else:
            logger.info(msg)

    def __str__(self) -> str:
        header = f"<{self.__class__.__name__} object>"
        return header + "\n" + self.log_parameters(is_return=True)
 
    @property
    def output(self) -> np.ndarray:
        """Get the smoothened output line."""
        return self._entities[0]

    @property
    def input(self) -> np.ndarray:
        """Get the input data of line"""
        return self._raw_coord
