from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np
from scipy.interpolate import interp1d, splev, splprep
from scipy.signal import savgol_filter

from ..datatypes import Number, UNSET, Unset, as_Number, as_bool, as_points, as_str
from ..logging_decorator import logging_and_warning_decorator
from .class_base import ClassBase
from .host_base import HostBase, OptsBase
from .opts import cover_value, diff_dict_values

# fmt: off
@dataclass(slots=True, repr=False)
class OptsSmooth(OptsBase):
    """
    Options object controlling Savitzky-Golay-based line smoothing.

    OptsSmooth is the user-facing options container paired with SmoothedLine.
    It controls how the raw polyline is smoothed, resampled, and wrapped for
    downstream tangent evaluation and function sampling.

    Important readable attributes:

    - `host`: the SmoothedLine currently using this opts object, if any.
    - `window_ratio`, `window_length`: two coupled ways of setting the smoothing
      window size.
    - `order`: polynomial order used by the Savitzky-Golay filter.
    - `num_out_ratio`: output resampling density relative to the processed input.
    - `mode`: smoothing/interpolation mode, either `"interp"` or `"wrap"`.
    - `min_line_length`: minimum processed line length required before smoothing
      is allowed.

    Common user actions:

    - `act_finalize()`: validate defaults and lock the opts into functioning use.
    - `act_asdict()`: export the current opts values as a plain dictionary.
    - `act_save_json()`: save the current opts to JSON.
    - `act_load_json()`: load a JSON snapshot into this existing opts object.

    Representation:

    - `str(opts)` returns a short one-line identity.
    - `repr(opts)` returns the full current opts summary.
    """

    window_ratio:               Number | None | Unset               = UNSET
    window_length:              int | None | Unset                  = UNSET
    order:                      int | Unset                         = UNSET
    num_out_ratio:              Number | Unset                      = UNSET
    mode:                       Literal["interp", "wrap"] | Unset   = UNSET
    min_line_length:            int | Unset                         = UNSET

    __attrs__ = {
        **(OptsBase.__attrs__),
        "window_ratio":         "window ratio for smoothing: line_length / window_length",
        "window_length":        "explicit window length for smoothing",
        "order":                "smoothing polynomial order",
        "num_out_ratio":        "ratio between output and input #points in smoothing",
        "mode":                 "smoothing mode (interp or wrap)",
        "min_line_length":      "minimum line length to be smoothed",
    }

    impl_validators = {
        **(OptsBase.impl_validators),
        "window_ratio":         lambda v, d: None if v is None else as_Number(v, name=d),
        "window_length":        lambda v, d: None if v is None else as_Number(v, name=d, is_int=True),
        "order":                lambda v, d: as_Number(v, name=d, is_int=True, value_range=(3, np.inf)),
        "num_out_ratio":        lambda v, d: as_Number(v, name=d, value_range=(1e-12, np.inf)),
        "mode":                 lambda v, d: as_str(v, name=d, pool=("interp", "wrap")),
        "min_line_length":      lambda v, d: as_Number(v, name=d, is_int=True, value_range=(2, np.inf)),
    }

    impl_defaults_frozen = MappingProxyType({
        **(OptsBase.impl_defaults_frozen),
        "tag":                  "smooth options",
        "window_ratio":         None,
        "window_length":        None,
        "order":                3,
        "num_out_ratio":        1,
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


# SmoothedLine keeps the HostBase commit pipeline but specializes it for
# one-dimensional line smoothing and spline-based tangent evaluation.
#
# Subclasses should preserve the distinction between raw input coordinates,
# processed coordinates entering the smoothing pipeline, and final smoothed
# output. If the smoothing stage is overridden, keep the fallback contract
# consistent so `calc_result`, `entity_tck`, and `calc_status` remain
# synchronized.
class SmoothedLine(HostBase):
    """
    SmoothedLine wraps a polyline and optionally produces a smoothed result.

    This class keeps raw input coordinates together with a smoothing pipeline,
    a cached spline representation, and the final resampled output line.
    Normal users provide input coordinates, then inspect or change smoothing
    settings through `line.opts` or `line.act_commit(...)`.

    Important readable attributes:

    - `opts`: the paired OptsSmooth controlling the smoothing pipeline.
    - `raw_coords`: the original input polyline coordinates.
    - `calc_coords`: the processed coordinates currently entering the smoother.
    - `calc_num_init`: the number of processed input points currently used.
    - `calc_num_out`: the number of output points requested after smoothing.
    - `result`: the final output coordinates, either smoothed or fallback raw
      coordinates.
    - `entity_tck`: the spline cache used for tangent evaluation, or None when
      smoothing is unavailable.
    - `calc_is_smoothed`: whether smoothing completed successfully.
    - `calc_status`: a human-readable status string describing the pipeline
      outcome.

    Common inspection helpers:

    - `show_readable_attrs()`: show the main readable line attributes.
    - `show_modifiable_attrs()`: show which line or opts attributes can be changed.
    - `show_attr_desc(name)`: describe a specific readable attribute.
    - `show_relations()`: show object relations inherited from HostBase/ClassBase.

    Common user actions:

    - `act_commit(...)`: update smoothing parameters or other host inputs.
    - `line.name = ...` or `line.raw_name = ...`: rename the smoothed line object.
    - `act_calc_tangent(u_percent, ...)`: evaluate the unit tangent of the cached
      spline at a normalized location.

    Representation and array behavior:

    - `str(line)` returns the short host-style identity.
    - `repr(line)` returns the detailed host summary.
    - `np.asarray(line)`, indexing, iteration, and `len(line)` operate on the
      current `result` array.
    """

    # fmt: off
    __attr_defs__ = {
        **dict(HostBase.__attr_defs__),
        "raw_name": {
            **dict(HostBase.__attr_defs__["raw_name"]),
            "doc": "The name identifier of the original line",
        },
        "raw_coords": {
            "doc":                        "Raw input line coordinates (shape: N x D)",
            "validator":                  lambda v, d: as_points(v, name=d, dim=None),
            "is_public_settable":         True,
            "is_protected":               False,
            "is_reapply_opts_after_raw":  True,
        },
        "calc_coords": {
            "doc":                "The processed coordinates actually sent into the smoothing pipeline",
            "kind":               "calc",

        },
        "calc_num_init": {
            "doc":                "Read-only: Number of processed input points currently entering the smoothing pipeline.",
            "kind":               "property",

        },
        "calc_num_out": {
            "doc":                "Read-only: Number of output points requested after smoothing.",
            "kind":               "property",

        },
        "calc_result": {
            "doc":                "The smoothed output coordinates (shape: M x D)",
            "kind":               "calc",

        },
        "entity_tck": {
            "doc":                "B-spline representation (tck) used for evaluating curve derivatives",
            "kind":               "entity",

        },
        "calc_is_smoothed": {
            "doc":                "Boolean flag indicating whether smoothing was applied",
            "kind":               "calc",

        },
        "state_is_window_warning": {
            "doc":                "Whether to present the warning when both window_length and window_ratio are provided.",
            "validator":          lambda v, d: as_bool(v, name=d),
            "is_public_settable": True,
            "is_protected":       False,
        },
        "calc_status": {
            "doc": (
                "Status indicator of the smoothing pipeline. "
                "Set to 'success' if smoothing completes normally. "
                "If smoothing is skipped or disabled due to internally detected "
                "conditions (e.g. line too short, invalid window size, "
                "or numerical failures), this field stores a human-readable "
                "string describing the specific reason."
            ),
            "kind":               "calc",

        },
        "result": {
            "doc":                "Read-only: Final output coordinates produced by the smoothing pipeline.",
            "kind":               "property",

        },
    }
    # fmt: on

    __slots__ = (
        "raw_coords",
        "calc_coords",
        "calc_result",
        "entity_tck",
        "calc_is_smoothed",
        "state_is_window_warning",
        "calc_status",
    )
    # -------------------------------
    # Initialization
    # -------------------------------

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
        is_window_warning: bool = True,
        **kwargs,
    ):

        line_coord_input = type(self).__attr_defs__["raw_coords"]["validator"](
            line_coord_input,
            type(self).__attr_defs__["raw_coords"]["doc"],
        )

        is_window_warning = type(self).__attr_defs__["state_is_window_warning"][
            "validator"
        ](
            is_window_warning,
            type(self).__attr_defs__["state_is_window_warning"]["doc"],
        )
        object.__setattr__(self, "raw_coords", line_coord_input)
        object.__setattr__(self, "calc_coords", self.raw_coords)
        object.__setattr__(self, "calc_is_smoothed", False)
        object.__setattr__(self, "state_is_window_warning", is_window_warning)
        object.__setattr__(self, "calc_status", "Failure, reason unknown.")

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

    # -------------------------------
    # Coordinate and fallback helpers
    # -------------------------------

    def _helper_resolve_coords(self):
        object.__setattr__(self, "calc_coords", self.raw_coords)

    @property
    def calc_num_init(self):
        coords = getattr(self, "calc_coords", None)
        if coords is None:
            coords = getattr(self, "raw_coords", None)
        return 0 if coords is None else len(coords)

    def _helper_fallback_no_smooth(self, reason: str) -> None:
        object.__setattr__(self, "calc_is_smoothed", False)
        object.__setattr__(self, "calc_result", self.calc_coords)
        object.__setattr__(self, "entity_tck", None)
        object.__setattr__(
            self,
            "calc_status",
            f"The line `{self.name}` is not smoothed, reason: {reason}.",
        )

    # -------------------------------
    # Smoothing commit pipeline
    # -------------------------------

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

        with self.opts.act_internal_update():
            cover_value(
                self.opts,
                is_allow_cover_target_set=True,
                is_allow_unset_source=False,
                **kwargs,
            )

        self._helper_resolve_coords()

        msg = f"Start to smooth line {self.name!r} with {self.calc_num_init} points.\n"
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
                    int(self.calc_num_init / self.opts.window_ratio / 2) * 2 + 1,
                )
                object.__setattr__(
                    self.opts,
                    "window_ratio",
                    self.calc_num_init / self.opts.window_length,
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
                    self.calc_num_init / self.opts.window_length,
                )

            if self.calc_num_init < self.opts.min_line_length:
                reason = (
                    f"the minimum length of line smoothing is set to be {self.opts.min_line_length} "
                    f"points, while the current line has {self.calc_num_init} points"
                )
                self._helper_fallback_no_smooth(reason)
                raise SmoothingConfigError(reason)

            if self.opts.window_length >= self.calc_num_init:
                reason = (
                    f"Filter window length {self.opts.window_length} should not be larger than "
                    f"line length {self.calc_num_init}"
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

            line_points = savgol_filter(
                self.calc_coords,
                self.opts.window_length,
                self.opts.order,
                axis=0,
                mode=self.opts.mode,
            )

            is_periodic = self.opts.mode == "wrap"
            if is_periodic:
                # FITPACK periodic splines treat the last sample as the seam copy
                # of the first one. Provide that seam explicitly so no genuine
                # endpoint sample is overwritten in-place by `splprep(per=1)`.
                line_points_spline = np.concatenate((line_points, [line_points[0]]))
                uspline = np.linspace(0.0, 1.0, len(line_points_spline))
                u_out = np.linspace(0.0, 1.0, self.calc_num_out, endpoint=False)
            else:
                line_points_spline = line_points
                uspline = np.linspace(0.0, 1.0, self.calc_num_init)
                u_out = np.linspace(0.0, 1.0, self.calc_num_out)

            tck = splprep(
                line_points_spline.T.copy(),
                u=uspline,
                s=0,
                per=int(is_periodic),
            )[0]
            result = np.array(splev(u_out, tck)).T
            object.__setattr__(self, "entity_tck", tck)
            object.__setattr__(self, "calc_result", result)

            object.__setattr__(self, "calc_is_smoothed", True)
            object.__setattr__(self, "calc_status", "Success")

        except SmoothingConfigError as e:
            logger.exception("Smoothing aborted (manual check)")
            logger.recovery(
                "Fallback applied: smoothing disabled; using raw coordinates."
            )
            self._helper_fallback_no_smooth(str(e))

        except (TypeError, ValueError, RuntimeError):
            logger.exception("Smoothing aborted (system error)")
            logger.recovery(
                "Fallback applied: smoothing disabled; using raw coordinates."
            )
            self._helper_fallback_no_smooth("system error")

    # -------------------------------
    # Public smoothing actions
    # -------------------------------

    def act_calc_tangent(self, u_percent, is_return_coord=False):

        tck = getattr(self, "entity_tck", None)
        if tck is None:
            raise RuntimeError(
                "Spline cache `entity_tck` is missing."
                "Probably the line is not properly initialized or successfully smoothed."
            )

        u_percent = as_Number(
            u_percent,
            value_range=(0, 100),
            name="Continuous spline parameter along the curve",
        )
        u_percent /= 100
        if self.opts.mode == "wrap":
            u_percent = np.mod(u_percent, 1.0)
        dr_dx = np.asarray(splev(u_percent, self.entity_tck, der=1), dtype=float)
        length = float(np.linalg.norm(dr_dx))
        if (not np.isfinite(length)) or length < 1e-9:
            raise ValueError(
                f"Degenerate spline derivative at {u_percent}: ||dr/dx||={length}."
            )

        t_hat = dr_dx / length
        if not is_return_coord:
            return t_hat

        coord = np.asarray(splev(u_percent, self.entity_tck, der=0), dtype=float)
        return t_hat, coord

    def act_calc_pos(self, u_percent):
        tck = getattr(self, "entity_tck", None)
        if tck is None:
            raise RuntimeError(
                "Spline cache `entity_tck` is missing."
                "Probably the line is not properly initialized or successfully smoothed."
            )

        u_percent = as_Number(
            u_percent,
            value_range=(0, 100),
            name="Continuous spline parameter along the curve",
        )
        u_percent /= 100
        if self.opts.mode == "wrap":
            u_percent = np.mod(u_percent, 1.0)

        return np.asarray(splev(u_percent, self.entity_tck, der=0), dtype=float)

    # -------------------------------
    # Array-style access
    # -------------------------------

    def __array__(self, dtype=None):
        arr = self.calc_result
        return np.asarray(arr, dtype=dtype) if dtype is not None else arr

    def __getitem__(self, idx):
        return self.calc_result[idx]

    def __iter__(self):
        return iter(self.calc_result)

    def __len__(self) -> int:
        result = getattr(self, "calc_result", None)
        return 0 if result is None else len(result)

    # -------------------------------
    # Readable properties
    # -------------------------------

    @property
    def calc_num_out(self):
        return max(1, int(self.calc_num_init * self.opts.num_out_ratio))

    @property
    def result(self):
        return self.calc_result


# SmoothedLineFunc samples a numerical function along the normalized parameter
#
# Subclasses should treat this class as a staged sampling pipeline. Override
# the smallest helper that matches the customization you need: resolve owner-
# dependent defaults, preprocess query points, sample raw values, or prepare
# interpolation data for periodic behavior.


# NOTE:
# The legacy `SmoothedLineFunc` below still follows the old `ClassBase`
# protocol and has not yet been migrated to the new managed-attribute
# conventions. Keep it commented out for now and revisit it in a later pass.
## class SmoothedLineFunc(ClassBase):
#     """
#     SmoothedLineFunc represents a sampled function of `u_percent` along a
#     smoothed line.
#
#     Users provide a numerical sampling function and a set of `u_percent`
#     sampling points. The class evaluates the function at those points, stores
#     the sampled values and metrics, and builds an interpolator for later
#     evaluation.
#
#     Important readable attributes:
#
#     - `owner`: the SmoothedLine currently associated with this sampled function.
#     - `_raw_func`: the numerical sampling function.
#     - `_raw_u_samples`: the normalized sample locations in `u_percent`.
#     - `_raw_func_kwargs`: extra keyword arguments forwarded during sampling.
#     - `_raw_owner_opts_snapshot`: the owner opts snapshot recorded at the last
#       successful refresh.
#     - `_calc_values`: sampled values returned by the numerical function.
#     - `_calc_metrics`: optional per-sample metrics returned by the function.
#     - `_entity_interpolator`: the interpolator built from the sampled values.
#
#     Common inspection helpers:
#
#     - `show_readable_attrs()`: show the main readable stored fields.
#     - `show_attr_desc(name)`: describe a specific readable attribute.
#     - `show_relations()`: show object relations such as the bound owner.
#
#     Common user actions:
#
#     - `act_refresh(...)`: rebuild the sampled values and interpolator.
#     - `interpolate(u_percent)`: evaluate the interpolated function.
#     - `__call__(u_percent)`: shorthand for `interpolate(...)`.
#
#     Representation:
#
#     - `str(obj)` returns the short ClassBase-style identity.
#     - `repr(obj)` returns a compact summary including sample count and mode.
#     """
#
#     __attrs__ = {
#         **(ClassBase.__attrs__),
#         "raw_name": "The name identifier of this smoothed-line function.",
#         "_raw_func": "Numerical function that maps a single u_percent sample to a value or to a (value, metric) pair.",
#         "_raw_u_samples": "Sampling locations in u_percent used to evaluate the numerical function.",
#         "_raw_func_kwargs": "Extra keyword arguments passed to the numerical function during sampling.",
#         "_raw_owner_opts_snapshot": "Snapshot of owner.opts at the time this line function was last sampled.",
#         "_calc_values": "Values returned by the numerical function at each sampling location.",
#         "_calc_metrics": "Per-sample metrics returned by the numerical function, or None if unavailable.",
#         "_entity_interpolator": "Interpolator object built from the sampled values.",
#     }
#
#     __relations__ = {
#         **(ClassBase.__relations__),
#         "owner": "The SmoothedLine instance that this function is associated with.",
#     }
#
#     __slots__ = (
#         "raw_coords",
#         "calc_coords",
#         "impl_calc_num_out",
#         "calc_result",
#         "entity_tck",
#         "calc_is_smoothed",
#         "state_is_window_warning",
#         "calc_status",
#     )
#
#     # -------------------------------
#     # Validation and owner-state helpers
#     # -------------------------------
#
#     @staticmethod
#     def _helper_validate_u_samples(u_samples):
#         u_samples = np.asarray(u_samples, dtype=float).reshape(-1)
#         if u_samples.ndim != 1 or len(u_samples) == 0:
#             raise ValueError("`u_samples` must be a non-empty one-dimensional array.")
#         if np.any(~np.isfinite(u_samples)):
#             raise ValueError("`u_samples` must contain only finite values.")
#         if np.min(u_samples) < 0 or np.max(u_samples) > 100:
#             raise ValueError("`u_samples` must stay within the range [0, 100].")
#         u_samples = np.unique(np.sort(u_samples))
#         if len(u_samples) == 0:
#             raise ValueError(
#                 "`u_samples` must remain non-empty after sorting and deduplication."
#             )
#         return u_samples
#
#     def _helper_get_owner_mode_from(self, opts_dict):
#         owner_mode = None if opts_dict is None else opts_dict.get("mode", None)
#         return (
#             "interp"
#             if owner_mode is None
#             else as_str(
#                 owner_mode, name="owner smoothing mode", pool=("interp", "wrap")
#             )
#         )
#
#     # -------------------------------
#     # Initialization
#     # -------------------------------
#
#     @logging_and_warning_decorator(start_finish_level=5)
#     def __init__(
#         self,
#         func,
#         u_samples,
#         owner: SmoothedLine,
#         func_kwargs: Mapping[str, Any] | None = None,
#         name: str = "smoothed line function",
#         logger=None,
#     ):
#         super().__init__(name=name, name_replace="smoothed line function")
#
#         if not isinstance(owner, SmoothedLine):
#             raise TypeError("`owner` for SmoothedLineFunc must be a SmoothedLine.")
#         if not callable(func):
#             raise TypeError("`func` for SmoothedLineFunc must be callable.")
#
#         u_samples = self._helper_validate_u_samples(u_samples)
#
#         object.__setattr__(self, "_raw_func", func)
#         object.__setattr__(self, "_raw_u_samples", u_samples)
#         object.__setattr__(
#             self,
#             "_raw_func_kwargs",
#             {} if func_kwargs is None else dict(func_kwargs),
#         )
#         object.__setattr__(self, "_raw_owner_opts_snapshot", None)
#         object.__setattr__(self, "_calc_values", None)
#         object.__setattr__(self, "_calc_metrics", None)
#         object.__setattr__(self, "_entity_interpolator", None)
#         self.act_bind_relation_base("owner", owner, is_weak=True)
#
#         self.act_refresh(u_samples=u_samples)
#
#     # -------------------------------
#     # Sampling and refresh helpers
#     # -------------------------------
#
#     @logging_and_warning_decorator(start_finish_level=5)
#     def _helper_warn_if_owner_opts_changed(self, logger=None):
#         owner = self.owner
#         if owner is None:
#             return
#
#         opts_now = owner.opts.act_asdict()
#         opts_then = self._raw_owner_opts_snapshot
#         diff_then, diff_now = diff_dict_values(opts_then, opts_now)
#         if not diff_then and not diff_now:
#             return
#
#         logger.warning(
#             f"Smoothed line function {self.name!r} was sampled with stale owner opts.\n"
#             f"Recorded opts diff: {diff_then}\n"
#             f"Current owner opts diff: {diff_now}\n"
#             "Consider calling `act_refresh(...)` to rebuild the sampled interpolator."
#         )
#
#     @logging_and_warning_decorator(start_finish_level=5)
#     def act_refresh(self, u_samples=None, logger=None):
#         owner = self.owner
#         if owner is None:
#             raise RuntimeError(
#                 "Cannot refresh a SmoothedLineFunc without a live owner."
#             )
#
#         if u_samples is not None:
#             object.__setattr__(
#                 self,
#                 "_raw_u_samples",
#                 self._helper_validate_u_samples(u_samples),
#             )
#
#         values = []
#         metrics = []
#         has_metric = False
#         for u in self._raw_u_samples:
#             sample_result = self._raw_func(float(u), **self._raw_func_kwargs)
#             if isinstance(sample_result, tuple) and len(sample_result) == 2:
#                 value, metric = sample_result
#             else:
#                 value, metric = sample_result, None
#             values.append(np.asarray(value))
#             metrics.append(metric)
#             has_metric = has_metric or (metric is not None)
#
#         values = np.stack(values, axis=0)
#         metrics = metrics if has_metric else None
#
#         opts_snapshot = owner.opts.act_asdict()
#         mode = self._helper_get_owner_mode_from(opts_snapshot)
#
#         if mode == "wrap":
#             u_interp = np.concatenate(
#                 [
#                     self._raw_u_samples - 100.0,
#                     self._raw_u_samples,
#                     self._raw_u_samples + 100.0,
#                 ]
#             )
#             values_interp = np.concatenate([values, values, values], axis=0)
#         else:
#             u_interp = self._raw_u_samples
#             values_interp = values
#
#         interpolator = interp1d(
#             u_interp,
#             values_interp,
#             axis=0,
#             kind="linear",
#             bounds_error=False,
#             fill_value="extrapolate",
#             assume_sorted=True,
#         )
#
#         object.__setattr__(self, "_raw_owner_opts_snapshot", dict(opts_snapshot))
#         object.__setattr__(self, "_calc_values", values)
#         object.__setattr__(self, "_calc_metrics", metrics)
#         object.__setattr__(self, "_entity_interpolator", interpolator)
#
#     # -------------------------------
#     # Public evaluation actions
#     # -------------------------------
#
#     def interpolate(self, u_percent):
#         self._helper_warn_if_owner_opts_changed()
#         u_percent = np.asarray(u_percent, dtype=float)
#         mode = self._helper_get_owner_mode_from(self._raw_owner_opts_snapshot)
#         if mode == "wrap":
#             u_percent = np.mod(u_percent, 100.0)
#         return self._entity_interpolator(u_percent)
#
#     def __call__(self, u_percent):
#         return self.interpolate(u_percent)
#
#     # -------------------------------
#     # Representation
#     # -------------------------------
#
#     def __repr__(self) -> str:
#         cls_name = self.__class__.__name__
#         mode = self._helper_get_owner_mode_from(self._raw_owner_opts_snapshot)
#         return (
#             f"{cls_name}({self.name!r}), num_samples={len(self._raw_u_samples)}, "
#             f"mode={mode!r}"
#         )
#
#     # ==================== OVERRIDE ====================
#     # SmoothedLineFunc overrides the default string form to keep the short
#     # ClassBase-style identity view for compact display.
#     # ==================================================
#
#     def __str__(self) -> str:
#         return ClassBase.__str__(self)
#
