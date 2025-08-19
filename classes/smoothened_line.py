import numpy as np
from typing import Optional, Literal
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev
from dataclasses import dataclass

from ..logging_decorator import logging_and_warning_decorator
from ..datatypes import Number, as_Number

@dataclass
class SmoothenConfig:
    window_ratio: Number = 3                    
    window_length: Optional[Number] = None       
    order: Number = 3                                
    N_out_ratio: Number = 3.0                      
    mode: Literal["interp", "wrap"] = "interp"  

    def __post_init__(self):

        self.window_ratio = as_Number(self.window_ratio, name="window_ratio of smoothening")
        if self.window_length is not None:
            self.window_length = as_Number(self.window_ratio, name="window_ratio of smoothening")
        self.order = as_Number(self.order, name="smoothing order")
        self.N_out_ratio = as_Number(self.N_out_ratio, name="ratio between # points of output and input in smoothening")

        if self.mode not in ("interp", "wrap"):
            raise ValueError("smoothing mode must be interp or wrap")

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
        config: SmoothenConfig = SmoothenConfig(),
        logger=None,
    ):
        
        self._line_coord_input = line_coord_input
        self._config = config

        self.apply_smoothen(logger=logger)

    @logging_and_warning_decorator()
    def apply_smoothen(self, logger=None):

        if self._config.window_length is None:
            self._config.window_length = int(self._config.N_init / self._config.window_ratio / 2) * 2 + 1
            self._config.window_ratio = self._config.N_init / self._config.window_length
        else:
            if self._config.window_ratio is not None:
                logger.warning(
                    ">>> Window_length is manual input. window_ratio would be ignored."
                )
            self._config.window_length = self._config.window_length
            self._config.window_ratio = self._config.N_init / self._config.window_length

        self._config.N_out = int(self._config.N_init * self._config.N_out_ratio)

        # Step 1: Apply Savitzky-Golay filter to smoothen the curve
        line_length = self._config.N_init
        if self._config.window_length >= line_length:
            raise ValueError(
                f"Filter window size {self._config.window_length} must be smaller than line length {line_length}"
            )
        line_points = savgol_filter(
            self._line_coord_input,
            self._config.window_length,
            self._config.order,
            axis=0,
            mode=self._config.mode,
        )

        # Step 2: Define spline parameter u
        uspline = np.arange(self._config.N_init) / self._config.N_init

        # Step 3: Fit and evaluate spline
        tck = splprep(line_points.T, u=uspline, s=0)[0]
        self._output = np.array(splev(np.linspace(0, 1, self._config.N_out), tck)).T

    @logging_and_warning_decorator()
    def print_parameters(self, logger=None) -> None:
        """
        Print internal filter and output parameters for inspection.
        These information could be automatically saved into a log file.
        See the documentation of logging_and_warning_decorator()
        """
        logger.info(f"filter order: {self._config.order}")
        logger.info(f"filter mode: {self._config.mode}")
        logger.info(f"ratio between output and input: {self._config.N_out_ratio}")
        logger.info(f"length of output: {self._config.N_out}")
        logger.info(f"length of input: {self._config.N_init}")
        logger.info(f"window length: {self._config.window_length}")
        logger.info(f"input/window ratio: {self._config.window_ratio}")

    @property
    def output(self) -> np.ndarray:
        """Get the smoothened output line."""
        return self._output
