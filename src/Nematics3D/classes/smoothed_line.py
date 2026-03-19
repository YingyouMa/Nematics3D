import numpy as np
from typing import Literal
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev, interp1d
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Any

from ..logging_decorator import logging_and_warning_decorator
from .host_base import OptsBase, HostBase
from .class_base import ClassBase
from .opts import cover_value
from ..datatypes import Number, as_Number, as_str, as_bool, UNSET, Unset, as_points

# fmt: off
@dataclass(slots=True, repr=False)
class OptsSmooth(OptsBase):
    
    window_ratio:               Number | None | Unset               = UNSET
    window_length:              int | None | Unset                  = UNSET
    order:                      int | Unset                         = UNSET
    N_out_ratio:                Number | Unset                      = UNSET
    mode:                       Literal["interp", "wrap"] | Unset   = UNSET
    min_line_length:            int | Unset                         = UNSET

    __attrs__ = {
        **(OptsBase.__attrs__),
        "window_ratio":         "window ratio for smoothing: line_length / window_length",
        "window_length":        "explicit window length for smoothing",
        "order":                "smoothing polynomial order",
        "N_out_ratio":          "ratio between output and input #points in smoothing",
        "mode":                 "smoothing mode (interp or wrap)",
        "min_line_length":      "minimum line length to be smoothed",
    }

    _validators = {
        **(OptsBase._validators),
        "window_ratio":         lambda v, d: None if v is None else as_Number(v, name=d),
        "window_length":        lambda v, d: None if v is None else as_Number(v, name=d, is_int=True),
        "order":                lambda v, d: as_Number(v, name=d, is_int=True, value_range=(3, np.inf)),
        "N_out_ratio":          lambda v, d: as_Number(v, name=d, value_range=(1e-12, np.inf)),
        "mode":                 lambda v, d: as_str(v, name=d, pool=("interp", "wrap")),
        "min_line_length":      lambda v, d: as_Number(v, name=d, is_int=True, value_range=(2, np.inf)),
    }
    
    _DEFAULTS_FROZEN = MappingProxyType({
        **(OptsBase._DEFAULTS_FROZEN),
        "tag":                  "smooth options",
        "window_ratio":         None,
        "window_length":        None,
        "order":                3,
        "N_out_ratio":          1,
        "mode":                 "interp",
        "min_line_length":      50,
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


# SmoothedLine keeps the HostBase commit pipeline but specializes it for
# one-dimensional line smoothing and spline-based tangent evaluation.
#
# Subclasses should preserve the distinction between raw input coordinates,
# processed coordinates entering the smoothing pipeline, and final smoothed
# output. If the smoothing stage is overridden, keep the fallback contract
# consistent so `_calc_result`, `_entity_tck`, and `_state_status` remain
# synchronized.
class SmoothedLine(HostBase):
    """
    SmoothedLine wraps a polyline and optionally produces a smoothed result.

    Normal users provide input coordinates, then inspect or change smoothing
    settings through `line.opts` or `line.act_commit(...)`. The resulting
    coordinates are available through `line.result`, NumPy conversion, or
    indexing/iteration on the object itself. Use
    `line.show_modifiable_attrs()` to inspect configurable options.
    """

    # fmt: off
    __attrs__ = {
        **dict(HostBase.__attrs__),
        "raw_name":                 "The name identifier of the original line",
        "raw_coords":               "Raw input line coordinates (shape: N x D)",
        "_calc_coords":             "The processed coordinates actually sent into the smoothing pipeline",
        "_calc_N_init":             "Property: Number of processed input points actually used by the smoothing pipeline",
        "_calc_N_out":              "Number of output points (after smoothing)",
        "_calc_result":             "The smoothed output coordinates (shape: M x D)",
        "_entity_tck":              "B-spline representation (tck) used for evaluating curve derivatives",
        "_state_is_smoothed":       "Boolean flag indicating whether smoothing was applied",
        "state_is_window_warning":  "Whether to present the warning when both window_length and window_ratio are provided.",
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
        for k, v in __attrs__.items()
        if not v.startswith("Property:") and k not in HostBase.__slots__
    )

    _impl_validators = {
        **HostBase._impl_validators,
        "coords": lambda v, d: as_points(v, name=d, dim=None),
    }

    _impl_attrs_reapply_opts_after_raw = {"coords"}

    # ==================== OVERRIDE ====================
    # SmoothedLine overrides HostBase.__init__ because it must validate and cache
    # raw coordinates before the host opts pipeline is initialized, then trigger
    # the first smoothing pass immediately after opts finalization.
    # ==================================================
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
            self.__attrs__["raw_coords"],
        )

        object.__setattr__(self, "raw_coords", line_coord_input)
        object.__setattr__(self, "_calc_coords", self.raw_coords)

        object.__setattr__(self, "_state_is_smoothed", False)
        object.__setattr__(self, "state_is_window_warning", True)
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

    def _helper_resolve_coords(self):
        object.__setattr__(self, "_calc_coords", self.raw_coords)

    @property
    def _calc_N_init(self):
        coords = getattr(self, "_calc_coords", None)
        if coords is None:
            coords = getattr(self, "raw_coords", None)
        return 0 if coords is None else len(coords)

    def _helper_fallback_no_smooth(self, reason: str) -> None:
        object.__setattr__(self, "_state_is_smoothed", False)
        object.__setattr__(self, "_calc_result", self._calc_coords)
        object.__setattr__(self, "_calc_N_out", self._calc_N_init)
        object.__setattr__(self, "_entity_tck", None)
        object.__setattr__(
            self,
            "_state_status",
            f"The line `{self.name}` is not smoothed, reason: {reason}.",
        )

    # ==================== OVERRIDE ====================
    # SmoothedLine overrides HostBase._helper_commit_apply_opts_main because
    # smoothing opts require custom normalization, validation, fallback, and
    # spline-cache updates that are specific to line processing.
    # ==================================================
    @logging_and_warning_decorator()
    def _helper_commit_apply_opts_main(
        self, is_reapply_opts=False, logger=None, **kwargs
    ):

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

        self._helper_resolve_coords()

        msg = f"Start to smooth line {self.name!r} with {self._calc_N_init} points.\n"
        msg += f"window length = {self.opts.window_length}\n"
        msg += f"window ratio = {self.opts.window_ratio}\n"
        msg += f"minimum smoothed line length = {self.opts.min_line_length}"
        logger.debug(msg)

        try:
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
                if self.opts.window_ratio is not None and self.state_is_window_warning:
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
                reason = (
                    f"the minimum length of line smoothing is set to be {self.opts.min_line_length} "
                    f"points, while the current line has {self._calc_N_init} points"
                )
                self._helper_fallback_no_smooth(reason)
                raise SmoothingConfigError(reason)

            if self.opts.window_length >= self._calc_N_init:
                reason = (
                    f"Filter window length {self.opts.window_length} should not be larger than "
                    f"line length {self._calc_N_init}"
                )
                raise SmoothingConfigError(reason)

            if self.opts.window_length <= self.opts.order:
                reason = (
                    f"Filter window length {self.opts.window_length} should not be smaller than "
                    f"filter order {self.opts.order}"
                )
                raise SmoothingConfigError(reason)

            logger.debug(
                f"Smoothing window length is finally chosen as {self.opts.window_length}"
            )

            object.__setattr__(
                self, "_calc_N_out", int(self._calc_N_init * self.opts.N_out_ratio)
            )

            line_points = savgol_filter(
                self._calc_coords,
                self.opts.window_length,
                self.opts.order,
                axis=0,
                mode=self.opts.mode,
            )

            uspline = np.arange(self._calc_N_init) / self._calc_N_init

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

    def act_calc_tangent(self, u_percent, is_return_coord=False):

        tck = getattr(self, "_entity_tck", None)
        if tck is None:
            raise RuntimeError(
                "Spline cache `_entity_tck` is missing."
                "Probably the line is not properly initialized or successfully smoothed."
            )

        u_percent = as_Number(
            u_percent,
            value_range=(0, 100),
            name="Continuous spline parameter along the curve",
        )
        u_percent /= 100
        dr_dx = np.asarray(splev(u_percent, self._entity_tck, der=1), dtype=float)

        length = float(np.linalg.norm(dr_dx))
        if (not np.isfinite(length)) or length < 1e-9:
            raise ValueError(
                f"Degenerate spline derivative at {u_percent}: ||dr/dx||={length}."
            )

        t_hat = dr_dx / length

        if is_return_coord:
            coord = np.asarray(splev(u_percent, self._entity_tck, der=0), dtype=float)

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


# SmoothedLineFunc samples a numerical function along the normalized parameter
# of a smoothed line and turns the sampled values into an interpolated
# one-dimensional function representation.
#
# Subclasses should treat this class as a staged sampling pipeline. Override
# the smallest helper that matches the customization you need: resolve owner-
# dependent defaults, preprocess query points, sample raw values, or prepare
# interpolation data for periodic behavior.


class SmoothedLineFunc(ClassBase):
    """
    SmoothedLineFunc represents a sampled function of `u_percent` along a
    smoothed line.

    Users provide a numerical sampling function and a set of `u_percent`
    sampling points. The class evaluates the function at those points, stores
    the sampled values and metrics, and builds an interpolator for later
    evaluation. Use `show_getattrs()` to inspect the available stored fields.
    """

    __attrs__ = {
        **(ClassBase.__attrs__),
        "raw_name": "The name identifier of this smoothed-line function.",
        "_raw_func": "Numerical function that maps a single u_percent sample to a value or to a (value, metric) pair.",
        "_raw_u_samples": "Sampling locations in u_percent used to evaluate the numerical function.",
        "_raw_func_kwargs": "Extra keyword arguments passed to the numerical function during sampling.",
        "_calc_values": "Values returned by the numerical function at each sampling location.",
        "_calc_metrics": "Per-sample metrics returned by the numerical function, or None if unavailable.",
        "_entity_interpolator": "Interpolator object built from the sampled values.",
    }

    __relations__ = {
        **(ClassBase.__relations__),
        "owner": "The SmoothedLine instance that this function is associated with.",
    }

    __slots__ = tuple(k for k in __attrs__.keys() if k not in ClassBase.__slots__)

    # ==================== OVERRIDE ====================
    # SmoothedLineFunc overrides ClassBase.__init__ because it must validate
    # inputs, sample the function immediately, and build the interpolator.
    # ==================================================
    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        func,
        u_samples,
        owner: SmoothedLine,
        func_kwargs: Mapping[str, Any] | None = None,
        name: str = "smoothed line function",
        logger=None,
    ):
        super().__init__(name=name, name_replace="smoothed line function")

        if not isinstance(owner, SmoothedLine):
            raise TypeError("`owner` for SmoothedLineFunc must be a SmoothedLine.")
        if not callable(func):
            raise TypeError("`func` for SmoothedLineFunc must be callable.")

        u_samples = np.asarray(u_samples, dtype=float).reshape(-1)
        if u_samples.ndim != 1 or len(u_samples) == 0:
            raise ValueError("`u_samples` must be a non-empty one-dimensional array.")
        if np.any(~np.isfinite(u_samples)):
            raise ValueError("`u_samples` must contain only finite values.")
        if np.min(u_samples) < 0 or np.max(u_samples) > 100:
            raise ValueError("`u_samples` must stay within the range [0, 100].")
        u_samples = np.unique(np.sort(u_samples))
        if len(u_samples) == 0:
            raise ValueError(
                "`u_samples` must remain non-empty after sorting and deduplication."
            )

        object.__setattr__(self, "_raw_func", func)
        object.__setattr__(self, "_raw_u_samples", u_samples)
        object.__setattr__(
            self,
            "_raw_func_kwargs",
            {} if func_kwargs is None else dict(func_kwargs),
        )
        object.__setattr__(self, "_calc_values", None)
        object.__setattr__(self, "_calc_metrics", None)
        object.__setattr__(self, "_entity_interpolator", None)
        self.act_bind_relation_base("owner", owner, is_weak=True)

        values = []
        metrics = []
        has_metric = False
        for u in self._raw_u_samples:
            sample_result = self._raw_func(float(u), **self._raw_func_kwargs)
            if isinstance(sample_result, tuple) and len(sample_result) == 2:
                value, metric = sample_result
            else:
                value, metric = sample_result, None
            values.append(np.asarray(value))
            metrics.append(metric)
            has_metric = has_metric or (metric is not None)

        values = np.stack(values, axis=0)
        metrics = metrics if has_metric else None

        owner_mode = getattr(owner.opts, "mode", None)
        mode = (
            "interp"
            if owner_mode is None
            else as_str(
                owner_mode, name="owner smoothing mode", pool=("interp", "wrap")
            )
        )

        if mode == "wrap":
            u_interp = np.concatenate(
                [
                    self._raw_u_samples - 100.0,
                    self._raw_u_samples,
                    self._raw_u_samples + 100.0,
                ]
            )
            values_interp = np.concatenate([values, values, values], axis=0)
        else:
            u_interp = self._raw_u_samples
            values_interp = values

        interpolator = interp1d(
            u_interp,
            values_interp,
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )

        object.__setattr__(self, "_calc_values", values)
        object.__setattr__(self, "_calc_metrics", metrics)
        object.__setattr__(self, "_entity_interpolator", interpolator)

    def interpolate(self, u_percent):
        u_percent = np.asarray(u_percent, dtype=float)
        owner_mode = getattr(self.owner.opts, "mode", None)
        mode = (
            "interp"
            if owner_mode is None
            else as_str(
                owner_mode, name="owner smoothing mode", pool=("interp", "wrap")
            )
        )
        if mode == "wrap":
            u_percent = np.mod(u_percent, 100.0)
        return self._entity_interpolator(u_percent)

    def __call__(self, u_percent):
        return self.interpolate(u_percent)

    # ==================== OVERRIDE ====================
    # SmoothedLineFunc overrides ClassBase.__repr__ because the sampled
    # function is most useful when summarized by sample count and owner mode.
    # ==================================================
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        owner_mode = getattr(self.owner.opts, "mode", None)
        mode = (
            "interp"
            if owner_mode is None
            else as_str(
                owner_mode, name="owner smoothing mode", pool=("interp", "wrap")
            )
        )
        return (
            f"{cls_name}({self.name!r}), num_samples={len(self._raw_u_samples)}, "
            f"mode={mode!r}"
        )

    # ==================== OVERRIDE ====================
    # SmoothedLineFunc overrides object.__str__ so plain string display stays
    # concise, matching the short ClassBase-style identity representation.
    # ==================================================
    def __str__(self) -> str:
        return ClassBase.__repr__(self)
