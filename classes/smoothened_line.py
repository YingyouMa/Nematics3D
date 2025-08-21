import numpy as np
from typing import Optional, Literal
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev
from dataclasses import dataclass

from ..logging_decorator import logging_and_warning_decorator
from ..datatypes import Number, as_Number
from .opts import OptsSmoothen

class SmoothenedLine:
    """
    Smooth a polyline using Savitzky-Golay filtering and cubic spline interpolation.

    This class is typically used to smoothen disclination lines or other geometric lines
    in 2D/3D space. It first applies Savitzky-Golay filtering to remove high-frequency
    noise, then uses cubic spline interpolation to upsample the line.

    Parameters
    ----------
    line_coord_input : array-like of shape (N, M)
        Coordinates of the original line to be smoothened.
        N is the number of points, and M is the dimension (usually 2 or 3).

    window_ratio : int, optional
        Used to compute the Savitzky-Golay filter window length:
        window_length = (N / window_ratio) rounded to nearest odd integer.
        Ignored if `window_length` is explicitly provided. Default is 3.

    window_length : int, optional
        The length of the Savitzky-Golay filter window.
        Must be odd. Overrides `window_ratio` if provided.
        Default is None.

    order : int, optional
        Order of the Savitzky-Golay polynomial filter. Default is 3.

    N_out_ratio : float, optional
        Ratio of output points to input points.
        For example, if 3.0 and input has 100 points, output will have 300 points.
        Default is 3.0.

    mode : {"interp", "wrap"}, optional
        Extension mode for the Savitzky-Golay filter.
        - "interp": no extension (default)
        - "wrap": wrap around edges (useful for closed loops)

    Attributes
    ----------
    output : np.ndarray of shape (N_out, M)
        The smoothened and interpolated output line.

    _input : np.ndarray or None
        The original input coordinates, only stored if `is_keep_origin=True`.

    _order : int
        Savitzky-Golay filter order.

    _window_length : int
        Savitzky-Golay filter window size (must be odd).

    _N_init : int
        Number of input points.

    _N_out : int
        Number of output points.

    _window_ratio : float
        Ratio of N_init / window_length.

    _mode : str
        Filter boundary mode.
    """

    def __init__(
        self,
        line_coord_input: np.ndarray,
        opts: OptsSmoothen = OptsSmoothen(),
        logger=None,
    ):
        
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
            self._is_smoothened = False
            logger.warning(f"{self.name} is not smoothened, because its length {self._data_coord} is shorter than the minum length {self._opts_min_line_length}.")
            self._output = self._data_coord
        else:

            if self._opts_window_length is None:
                self._opts_window_length = int(self._calc_N_init / self._opts_window_ratio / 2) * 2 + 1
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
            self._output = np.array(splev(np.linspace(0, 1, self._calc_N_out), tck)).T

    @logging_and_warning_decorator()
    def print_parameters(self, logger=None) -> None:
        """
        Print internal filter and output parameters for inspection.
        These information could be automatically saved into a log file.
        See the documentation of logging_and_warning_decorator()
        """
        if self._is_smoothened:
            logger.info(f"name: {self.name}")
            logger.info(f"filter order: {self._opts_order}")
            logger.info(f"filter mode: {self._opts_mode}")
            logger.info(f"ratio between output and input: {self._opts_N_out_ratio}")
            logger.info(f"length of output: {self._calc_N_out}")
            logger.info(f"length of input: {self._calc_N_init}")
            logger.info(f"window length: {self._opts_window_length}")
            logger.info(f"input/window ratio: {self._opts_window_ratio}")
        else:
            logger.info(f"{self.name} is not smoothened, because its length {self._data_coord} is shorter than the minum length {self._opts_min_line_length}.")


    @property
    def output(self) -> np.ndarray:
        """Get the smoothened output line."""
        return self._output
    
    @property
    def input(self) -> np.ndarray:
        """Get the input data of line"""
        return self._data_coord
