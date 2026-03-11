import numpy as np
from typing import Literal
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Any

from ..logging_decorator import logging_and_warning_decorator
from .host_base import OptsBase, HostBase
from .opts import cover_value
from ..datatypes import (
    Number,
    as_Number,
    as_str,
    as_bool,
    UNSET,
    Unset,
    as_points
)

# fmt: off
@dataclass(slots=True, repr=False)
class OptsSmooth(OptsBase):
    
    window_ratio:               Number | None | Unset               = UNSET
    window_length:              int | None | Unset                  = UNSET
    order:                      int | Unset                         = UNSET
    N_out_ratio:                Number | Unset                      = UNSET
    mode:                       Literal["interp", "wrap"] | Unset   = UNSET
    min_line_length:            int | Unset                         = UNSET
    is_window_warning:          bool | Unset                        = UNSET

    __descriptions__ = {
        **(OptsBase.__descriptions__),
        "window_ratio":         "window ratio for smoothing: line_length / window_length",
        "window_length":        "explicit window length for smoothing",
        "order":                "smoothing polynomial order",
        "N_out_ratio":          "ratio between output and input #points in smoothing",
        "mode":                 "smoothing mode (interp or wrap)",
        "min_line_length":      "minimum line length to be smoothed",
        "is_window_warning" :   "whether present the warning when window_length and window_ratio are both input"
    }

    _validators = {
        **(OptsBase._validators),
        "window_ratio":         lambda v, d: None if v is None else as_Number(v, name=d),
        "window_length":        lambda v, d: None if v is None else as_Number(v, name=d, is_int=True),
        "order":                lambda v, d: as_Number(v, name=d, is_int=True, value_range=(3, np.inf)),
        "N_out_ratio":          lambda v, d: as_Number(v, name=d, value_range=(1e-12, np.inf)),
        "mode":                 lambda v, d: as_str(v, name=d, pool=("interp", "wrap")),
        "min_line_length":      lambda v, d: as_Number(v, name=d, is_int=True, value_range=(2, np.inf)),
        "is_window_warning":    lambda v, d: as_bool(v, name=d)
    }
    
    _DEFAULTS_FROZEN = MappingProxyType({
        **(OptsBase._DEFAULTS_FROZEN),
        "tag":                  "smooth options",
        "window_ratio":         None,
        "window_length":        None,
        "order":                3,
        "N_out_ratio":          2,
        "mode":                 "interp",
        "min_line_length":      50,
        "is_window_warning":    True
    })
# fmt: on


class SmoothingConfigError(ValueError):
    """
    Recoverable configuration error for smoothing.

    Raised only for explicitly recognized, user-fixable issues inside
    the smoothing helper (e.g., missing window length). This exception
    is intended to be caught locally and converted to RECOVERY + fallback.
    """

    pass


class SmoothedLine(HostBase):

    # fmt: off
    __descriptions__ = {
        **dict(HostBase.__descriptions__),
        "raw_name":                 "The name identifier of the original line",    
        "raw_coords":               "Raw input line coordinates (shape: N x D)",
        "_calc_N_init":             "Number of input points (before smoothing)",   #!!! property
        "_calc_N_out":              "Number of output points (after smoothing)",
        "_calc_result":             "The smoothed output coordinates (shape: M x D)",
        "_entity_tck":              "B-spline representation (tck) used for evaluating curve derivatives",
        "_state_is_smoothed":       "Boolean flag indicating whether smoothing was applied",
        
        "_state_status": (
            "Status indicator of the smoothing pipeline. "
            "Set to 'success' if smoothing completes normally. "
            "If smoothing is skipped or disabled due to internally detected "
            "conditions (e.g. line too short, invalid window size, "
            "or numerical failures), this field stores a human-readable "
            "string describing the specific reason."),
        }
    # fmt: on

    __slots__ = tuple(
        k
        for k, v in __descriptions__.items()
        if not v.startswith("Property:") and k not in HostBase.__slots__
    )
    
    _impl_validators = {
        **HostBase._impl_validators,
        "coords": lambda v, d: as_points(v, name=d, dim=None),
    }
    
    _impl_attrs_reapply_opts_after_raw = {"coords"}

    def __init__(
        self,
        line_coord_input: np.ndarray,
        name: str | None = None,
        opts: OptsSmooth | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        **kwargs,
    ):

        line_coord_input = self._impl_validators["coords"](
            line_coord_input,
            self.__descriptions__["raw_coords"]
        )

        object.__setattr__(self, "raw_coords", line_coord_input)
        object.__setattr__(self, "_calc_N_init", len(self.raw_coords))

        object.__setattr__(self, "_state_is_smoothed", False)
        object.__setattr__(self, "_state_status", "Failure, reason unknown.")

        super().__init__(
            OptsSmooth,
            opts,
            opts_defaults_override,
            name=name,
            name_replace="line",
            **kwargs,
        )

        self.opts.act_finalize()
        self._helper_commit_apply_opts(is_reapply_opts=True)


    def _helper_fallback_no_smooth(self, reason: str) -> None:
        object.__setattr__(self, "_state_is_smoothed", False)
        object.__setattr__(self, "_calc_result", self.raw_coords)
        object.__setattr__(self, "_calc_N_out", self._calc_N_init)
        object.__setattr__(self, "_entity_tck", None)
        object.__setattr__(
            self,
            "_state_status",
            f"The line `{self.name}` is not smoothed, reason: {reason}.",
        )


    @logging_and_warning_decorator()
    def _helper_commit_apply_opts_main(self, is_reapply_opts=False, logger=None, **kwargs):
        
        object.__setattr__(self, "_calc_N_init", len(self.raw_coords))
        
        if not is_reapply_opts and not kwargs:
            return

        if kwargs:
            if "window_ratio" in kwargs and "window_length" not in kwargs:
                object.__setattr__(self.opts, "window_length", None)
            if "window_ratio" not in kwargs and "window_length" in kwargs:
                object.__setattr__(self.opts, "window_ratio", None)

        with self.opts._helper_internal_update():
            cover_value(
                self.opts,
                is_allow_cover_target_set=True,
                is_allow_unset_source=False,
                **kwargs,
            )

        msg = f"Start to smooth line {self.name!r} with {self._calc_N_init} points.\n"
        msg += f"window length = {self.opts.window_length}\n"
        msg += f"window ratio = {self.opts.window_ratio}\n"
        msg += f"minimum smoothed line length = {self.opts.min_line_length}"
        logger.debug(msg)

        try:
            logger.detail("Start to determine the smoothing window length.")
            if self.opts.window_length is None:
                if self.opts.window_ratio is None:
                    reason = "No input value provided for smooth window length."
                    raise SmoothingConfigError(reason)
                object.__setattr__(
                    self.opts,
                    "window_length",
                    int(self._calc_N_init / self.opts.window_ratio / 2) * 2 + 1,
                )
                object.__setattr__(
                    self.opts,
                    "window_ratio",
                    self._calc_N_init / self.opts.window_length,
                )
            else:
                if (
                    self.opts.window_ratio is not None
                    and self.opts.is_window_warning == True
                ):
                    logger.warning(
                        f"Window_length is manual input as {self.opts.window_length}. "
                        f"window_ratio ({self.opts.window_ratio}) would be ignored and reset."
                    )
                window_length = int(self.opts.window_length)
                if window_length % 2 == 0:
                    window_length += 1
                object.__setattr__(self.opts, "window_length", window_length)
                object.__setattr__(
                    self.opts,
                    "window_ratio",
                    self._calc_N_init / self.opts.window_length,
                )
                
            if self._calc_N_init < self.opts.min_line_length:
                reason = f"the minimum length of line smoothing is set to be {self.opts.min_line_length} points, while the current line has {self._calc_N_init} points"
                self._helper_fallback_no_smooth(reason)
                raise SmoothingConfigError(reason)

            if self.opts.window_length >= self._calc_N_init:
                reason = f"Filter window length {self.opts.window_length} should not be larger than line length {self._calc_N_init}"
                raise SmoothingConfigError(reason)

            if self.opts.window_length <= self.opts.order:
                reason = f"Filter window length {self.opts.window_length} should not be smaller than filter order {self.opts.order}"
                raise SmoothingConfigError(reason)

            logger.debug(
                f"Smoothing window length is finally chosen as {self.opts.window_length}"
            )

            object.__setattr__(
                self, "_calc_N_out", int(self._calc_N_init * self.opts.N_out_ratio)
            )
            logger.detail(
                f"Number of output points after smoothing is {self._calc_N_out}."
            )

            logger.detail("Applying Savitzky-Golay filter to smooth the curve")
            line_points = savgol_filter(
                self.raw_coords,
                self.opts.window_length,
                self.opts.order,
                axis=0,
                mode=self.opts.mode,
            )

            logger.detail("Defining spline parameter u")
            uspline = np.arange(self._calc_N_init) / self._calc_N_init

            logger.detail("Fitting and evaluate spline")
            u_out = np.linspace(0, 1, self._calc_N_out)
            tck = splprep(line_points.T, u=uspline, s=0)[0]
            result = np.array(splev(u_out, tck)).T
            object.__setattr__(self, "_entity_tck", tck)
            object.__setattr__(self, "_calc_result", result)

            object.__setattr__(self, "_state_is_smoothed", True)
            object.__setattr__(self, "_state_status", "Success")
 

        except SmoothingConfigError as e:
            logger.exception("Smoothing aborted (manual check)")
            logger.recovery(
                "Fallback applied: smoothing disabled; using raw coordinates."
            )
            self._helper_fallback_no_smooth(str(e))

        except Exception:
            logger.exception("Smoothing aborted (system error)")
            logger.recovery(
                "Fallback applied: smoothing disabled; using raw coordinates."
            )
            self._helper_fallback_no_smooth("system error")
        



    def act_calc_tangent(self, x_param, is_return_coord=False):
        
        tck = getattr(self, "_entity_tck", None)
        if tck is None:
            raise RuntimeError(
                "Spline cache `_entity_tck` is missing."
                "Probably the line is not properly initialized or successfully smoothed."
            )
        
        x_param = as_Number(x_param, value_range=(0,100), name="Continuous spline parameter along the curve")
        x_param /= 100
        dr_dx = np.asarray(splev(x_param, self._entity_tck, der=1), dtype=float)
        
        length = float(np.linalg.norm(dr_dx))
        if (not np.isfinite(length)) or length < 1e-9:
            raise ValueError(
                f"Degenerate spline derivative at {x_param}: ||dr/dx||={length}."
            )
    
        t_hat = dr_dx / length
        
        if is_return_coord:
            coord = np.asarray(splev(x_param, self._entity_tck, der=0), dtype=float)
            
        return (t_hat, coord) if is_return_coord else t_hat
            
            
        
        
        

    def __array__(self, dtype=None):
        arr = self._calc_result
        return np.asarray(arr, dtype=dtype) if dtype is not None else arr

    def __getitem__(self, idx):
        return self._calc_result[idx]

    def __iter__(self):
        return iter(self._calc_result)

    def __len__(self) -> int:
        return self._calc_N_out

    @property
    def result(self):
        return self._calc_result
