import numpy as np
from typing import Optional, Literal
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev
from dataclasses import dataclass

from ..logging_decorator import logging_and_warning_decorator
from .opts import OptsSmoothen, merge_opts


class SmoothenedLine:
    """
    Apply smoothing and spline interpolation to a polyline (sequence of 3D/ND coordinates).

    This class takes discrete line coordinates and applies a two-stage smoothing procedure:
    1. Savitzky–Golay filtering to reduce noise in the line coordinates.
    2. B-spline interpolation to resample the line with more evenly distributed points.

    Parameters
    ----------
    line_coord_input : np.ndarray

        Input line coordinates, shape (N, D), where N is number of points and D is spatial dimension.
    opts : OptsSmoothen, optional

        Dataclass storing smoothening parameters (window length, polynomial order, output ratio, etc.).
    logger : optional

        Logger instance used by the decorated methods to print info/warnings.
    **kwargs : dict
        Extra keyword arguments to override fields in `OptsSmoothen`.

    Attributes (selected)
    ---------------------
    - See ``__descriptions__`` for a complete list of attributes and their meanings.
    - Attributes are fixed via ``__slots__`` and cannot be arbitrarily extended.

    Notes
    -----
    - If the input line is shorter than `min_line_length`, smoothening is skipped.
    - `window_length` and `window_ratio` are mutually dependent:
      if `window_length` is not provided, it is derived from `window_ratio`.
    """

    __descriptions__ = {
        "name": "Name identifier of this line object",
        "_data_coord": "Raw input line coordinates (shape: N x D)",
        "_calc_N_init": "Number of input points (before smoothing)",
        "_calc_N_out": "Number of output points (after smoothing)",
        "_entities": "Whose first element is smoothed output coordinates (shape: M x D)",
        "_state_is_smoothened": "Boolean flag indicating whether smoothing was applied",
        "_opts_window_ratio": "Ratio used to compute window_length if not explicitly provided",
        "_opts_window_length": "Explicit smoothing window length (overrides window_ratio if set)",
        "_opts_order": "Polynomial order of Savitzky–Golay filter",
        "_opts_N_out_ratio": "Ratio between output and input number of points",
        "_opts_mode": "Smoothing mode (either 'interp' or 'wrap')",
        "_opts_min_line_length": "Minimum line length required to apply smoothing",
    }

    __slots__ = tuple(__descriptions__.keys())

    def __init__(
        self,
        line_coord_input: np.ndarray,
        opts: OptsSmoothen = OptsSmoothen(),
        logger=None,
        **kwargs,
    ):

        opts = merge_opts(opts, kwargs, prefix="")

        self._data_coord = line_coord_input
        self._calc_N_init = len(self._data_coord)

        for k, v in vars(opts).items():
            if k == "name":
                setattr(self, "name", v)
            else:
                setattr(self, f"_opts_{k}", v)

        self.apply_smoothen(logger=logger)

    @logging_and_warning_decorator()
    def apply_smoothen(self, logger=None):

        if len(self._data_coord) < self._opts_min_line_length:
            self._state_is_smoothened = False
            logger.warning(
                f"{self.name} is not smoothened, because its length {self._data_coord} is shorter than the minum length {self._opts_min_line_length}."
            )
            self._entities = [self._data_coord]
        else:

            self._state_is_smoothened = True

            if self._opts_window_length is None:
                self._opts_window_length = (
                    int(self._calc_N_init / self._opts_window_ratio / 2) * 2 + 1
                )
                self._opts_window_ratio = self._calc_N_init / self._opts_window_length
            else:
                # if self._opts_window_ratio is not None:
                #     logger.debug(
                #         f"Window_length is manual input as {self._opts_window_length}. window_ratio would be ignored."
                #     )
                self._opts_window_length = self._opts_window_length
                self._opts_window_ratio = self._calc_N_init / self._opts_window_length

            self._calc_N_out = int(self._calc_N_init * self._opts_N_out_ratio)

            # Step 1: Apply Savitzky-Golay filter to smoothen the curve
            line_length = self._calc_N_init
            if self._opts_window_length >= line_length:
                raise ValueError(
                    f"Filter window size {self._opts_window_length} must be smaller than line length {line_length}"
                )
            line_points = savgol_filter(
                self._data_coord,
                self._opts_window_length,
                self._opts_order,
                axis=0,
                mode=self._opts_mode,
            )

            # Step 2: Define spline parameter u
            uspline = np.arange(self._calc_N_init) / self._calc_N_init

            # Step 3: Fit and evaluate spline
            tck = splprep(line_points.T, u=uspline, s=0)[0]
            self._entities = [
                np.array(splev(np.linspace(0, 1, self._calc_N_out), tck)).T
            ]

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
                if attr in ("_opts_window_length", "_opts_window_ratio"):
                    lines.append(f"  {attr}: {value!r}  # {desc} (derived final value)")
                else:
                    lines.append(f"  {attr}: {value!r}  # {desc}")
        else:
            lines.append(
                f"[{self.name}] is not smoothened, because its length "
                f"{len(self._data_coord)} < minimum required {self._opts_min_line_length}."
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
        return self._data_coord
