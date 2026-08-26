from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np
from scipy.interpolate import interp1d, splev, splprep
from scipy.signal import savgol_filter

from ..datatypes import Number, UNSET, Unset, as_number, as_bool, as_points, as_str
from ..logging_decorator import logging_and_warning_decorator
from .class_base import AttrDef, ClassBase
from .host_base import HostBase, OptsBase
from .opts import cover_value, diff_dict_values
from .registry_base import RegistryBase

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
        "window_ratio":         lambda v, d: None if v is None else as_number(v, name=d),
        "window_length":        lambda v, d: None if v is None else as_number(v, name=d, is_integer=True),
        "order":                lambda v, d: as_number(v, name=d, is_integer=True, value_range=(3, np.inf)),
        "num_out_ratio":        lambda v, d: as_number(v, name=d, value_range=(1e-12, np.inf)),
        "mode":                 lambda v, d: as_str(v, name=d, pool=("interp", "wrap")),
        "min_line_length":      lambda v, d: as_number(v, name=d, is_integer=True, value_range=(2, np.inf)),
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
    - `linefuncs`: registry of functions sampled and interpolated along this
      line.
    - `linefunc_mode`: interpolation mode used by functions sampled along this
      line. By default it follows `opts.mode`.

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
        "raw_coords": AttrDef(
            doc="Raw input line coordinates (shape: N x D)",
            kind="raw",
            validator=lambda v, d: as_points(v, name=d, dim=None),
            is_reapply_opts_after_raw=True,
        ),
        "calc_coords": AttrDef(
            doc="The processed coordinates actually sent into the smoothing pipeline",
            kind="calc",
        ),
        "calc_num_init": AttrDef(
            doc="Read-only: Number of processed input points currently entering the smoothing pipeline.",
            kind="property",
            is_public_settable=False,
        ),
        "calc_num_out": AttrDef(
            doc="Read-only: Number of output points requested after smoothing.",
            kind="property",
            is_public_settable=False,
        ),
        "calc_result": AttrDef(
            doc="The smoothed output coordinates (shape: M x D)",
            kind="calc",
        ),
        "entity_tck": AttrDef(
            doc="B-spline representation (tck) used for evaluating curve derivatives",
            kind="entity",
        ),
        "entity_linefuncs": AttrDef(
            doc="RegistryBase object managing functions sampled along this line.",
            kind="entity",
        ),
        "impl_linefunc_count": AttrDef(
            doc="Monotonic counter used to assign default line-function names.",
            kind="impl",
        ),
        "calc_is_smoothed": AttrDef(
            doc="Boolean flag indicating whether smoothing was applied",
            kind="calc",
        ),
        "state_is_window_warning": AttrDef(
            doc="Whether to present the warning when both window_length and window_ratio are provided.",
            kind="state",
            validator=lambda v, d: as_bool(v, name=d),
        ),
        "calc_status": AttrDef(
            doc=(
                "Status indicator of the smoothing pipeline. "
                "Set to 'success' if smoothing completes normally. "
                "If smoothing is skipped or disabled due to internally detected "
                "conditions (e.g. line too short, invalid window size, "
                "or numerical failures), this field stores a human-readable "
                "string describing the specific reason."
            ),
            kind="calc",
        ),
        "result": AttrDef(
            doc="Read-only: Final output coordinates produced by the smoothing pipeline.",
            kind="property",
            is_public_settable=False,
        ),
        "linefuncs": AttrDef(
            doc="Read-only: Registry of functions sampled along this line.",
            kind="property",
            is_public_settable=False,
        ),
        "linefunc_mode": AttrDef(
            doc="Read-only: Interpolation mode used by functions sampled along this line.",
            kind="property",
            is_public_settable=False,
        ),
    }
    # fmt: on

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
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

        line_coord_input = (
            type(self)
            .__attr_defs__["raw_coords"]
            .validator(
                line_coord_input,
                type(self).__attr_defs__["raw_coords"].doc,
            )
        )

        is_window_warning = (
            type(self)
            .__attr_defs__["state_is_window_warning"]
            .validator(
                is_window_warning,
                type(self).__attr_defs__["state_is_window_warning"].doc,
            )
        )
        object.__setattr__(self, "raw_coords", line_coord_input)
        object.__setattr__(self, "calc_coords", self.raw_coords)
        object.__setattr__(self, "calc_result", self.raw_coords)
        object.__setattr__(self, "entity_tck", None)
        object.__setattr__(self, "entity_linefuncs", None)
        object.__setattr__(self, "impl_linefunc_count", 0)
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

        linefuncs = RegistryBase(
            "line functions",
            info=f"functions sampled along smoothed line {self.name!r}",
        )
        linefuncs.act_bind_relation_base("owner", self, is_weak=True)
        object.__setattr__(self, "entity_linefuncs", linefuncs)

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

        u_percent = as_number(
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

        u_percent = as_number(
            u_percent,
            value_range=(0, 100),
            name="Continuous spline parameter along the curve",
        )
        u_percent /= 100
        if self.opts.mode == "wrap":
            u_percent = np.mod(u_percent, 1.0)

        return np.asarray(splev(u_percent, self.entity_tck, der=0), dtype=float)

    def act_create_linefunc(
        self,
        func,
        u_samples,
        func_kwargs: Mapping[str, Any] | None = None,
        is_follow_owner_opts: bool = True,
        name: str | None = None,
    ):
        if name is None:
            name = f"line_func_{self.impl_linefunc_count}"

        linefunc = SmoothedLineFunc(
            func=func,
            u_samples=u_samples,
            owner=self,
            func_kwargs=func_kwargs,
            is_follow_owner_opts=is_follow_owner_opts,
            name=name,
        )
        self.entity_linefuncs.act_register(linefunc)
        object.__setattr__(
            self,
            "impl_linefunc_count",
            self.impl_linefunc_count + 1,
        )
        return linefunc

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

    @property
    def linefuncs(self):
        return self.entity_linefuncs

    @property
    def linefunc_mode(self):
        return self.opts.mode


# SmoothedLineFunc samples a numerical function along the normalized parameter
#
# Subclasses should treat this class as a staged sampling pipeline. Override
# the smallest helper that matches the customization you need: resolve owner-
# dependent defaults, preprocess query points, sample raw values, or prepare
# interpolation data for periodic behavior.


def _raise_type_error(name: str, value: Any):
    raise TypeError(f"{name} must be callable, got {type(value).__name__}.")


def linefunc_window_span_percent(
    *,
    window_ratio: Number,
) -> float:
    """
    Convert a SmoothedLine window ratio to a u-percent window span.

    The returned span is the full window width in the normalized `[0, 100]`
    parameter domain. `SmoothedLine` keeps `window_length` and `window_ratio`
    synchronized, so the line-function smoother only needs the normalized ratio.
    """
    window_ratio = as_number(
        window_ratio,
        name="line function window_ratio",
        value_range=(1e-12, np.inf),
    )
    return 100.0 / float(window_ratio)


def linefunc_spacing_weights(
    u_samples,
    mode: Literal["interp", "wrap"] = "interp",
) -> np.ndarray:
    """
    Estimate quadrature weights for non-uniform u-percent samples.

    Each sample receives the local cell width it represents. In `"interp"`
    mode, end samples receive half of their nearest interval. In `"wrap"` mode,
    samples are treated as periodic around the `[0, 100]` domain.
    """
    u_samples = _linefunc_as_u_samples(u_samples)
    mode = as_str(mode, name="line function smoothing mode", pool=("interp", "wrap"))
    _linefunc_validate_wrap_endpoint(u_samples, mode)

    if len(u_samples) == 1:
        return np.array([100.0], dtype=float)

    if mode == "wrap":
        prev_samples = np.roll(u_samples, 1)
        next_samples = np.roll(u_samples, -1)
        left = np.mod(u_samples - prev_samples, 100.0)
        right = np.mod(next_samples - u_samples, 100.0)
        return 0.5 * (left + right)

    weights = np.empty_like(u_samples, dtype=float)
    weights[0] = 0.5 * (u_samples[1] - u_samples[0])
    weights[-1] = 0.5 * (u_samples[-1] - u_samples[-2])
    if len(u_samples) > 2:
        weights[1:-1] = 0.5 * (u_samples[2:] - u_samples[:-2])
    return weights


def linefunc_kernel_weights(
    delta,
    window_span_percent: Number,
    *,
    kernel: Literal["boxcar", "tricube", "triangular", "gaussian"] = "boxcar",
) -> np.ndarray:
    """
    Compute smoothing-kernel weights from u-percent deltas.

    `window_span_percent` is interpreted as a full window width. Compact kernels
    therefore use `window_span_percent / 2` as their support radius.
    """
    delta = np.asarray(delta, dtype=float)
    window_span_percent = as_number(
        window_span_percent,
        name="line function window span in percent",
        value_range=(1e-12, np.inf),
    )
    radius = 0.5 * float(window_span_percent)
    if radius <= 0:
        raise ValueError("window_span_percent must be positive.")

    kernel = as_str(
        kernel,
        name="line function smoothing kernel",
        pool=("boxcar", "tricube", "triangular", "gaussian"),
    )
    distance_scaled = np.abs(delta) / radius

    if kernel == "boxcar":
        return (distance_scaled <= 1.0).astype(float)

    if kernel == "tricube":
        weights = np.zeros_like(distance_scaled, dtype=float)
        mask = distance_scaled < 1.0
        weights[mask] = (1.0 - distance_scaled[mask] ** 3) ** 3
        return weights

    if kernel == "triangular":
        return np.maximum(1.0 - distance_scaled, 0.0)

    return np.exp(-0.5 * distance_scaled**2)


def linefunc_smooth_values(
    u_samples,
    values,
    *,
    window_ratio: Number,
    order: int,
    mode: Literal["interp", "wrap"] = "interp",
    spacing_weights=None,
    kernel: Literal["boxcar", "tricube", "triangular", "gaussian"] = "boxcar",
    min_weight: float = 1e-12,
) -> np.ndarray:
    """
    Smooth values at their own u-percent sample locations.

    This is the public value-smoothing helper used before building the final
    line-function interpolator. The owner SmoothedLine already synchronizes
    `window_length` and `window_ratio`, so only `window_ratio` is needed here.
    """
    u_samples = _linefunc_as_u_samples(u_samples)
    mode = as_str(mode, name="line function smoothing mode", pool=("interp", "wrap"))
    _linefunc_validate_wrap_endpoint(u_samples, mode)

    values = np.asarray(values)
    if values.shape[0] != len(u_samples):
        raise ValueError(
            "values must have the same first dimension as u_samples. "
            f"Got values.shape={values.shape} and len(u_samples)={len(u_samples)}."
        )

    window_span_percent = linefunc_window_span_percent(window_ratio=window_ratio)
    order = as_number(
        order,
        name="line function local polynomial order",
        is_integer=True,
        value_range=(0, np.inf),
    )
    if spacing_weights is None:
        spacing_weights = linefunc_spacing_weights(u_samples, mode=mode)
    else:
        spacing_weights = np.asarray(spacing_weights, dtype=float).reshape(-1)
        if spacing_weights.shape != u_samples.shape:
            raise ValueError(
                "spacing_weights must have the same shape as u_samples. "
                f"Got {spacing_weights.shape} and {u_samples.shape}."
            )
        if np.any(spacing_weights < 0) or np.any(~np.isfinite(spacing_weights)):
            raise ValueError("spacing_weights must be finite and non-negative.")

    values_flat = values.reshape(len(u_samples), -1)
    output = np.empty_like(values_flat, dtype=float)
    deltas_all = _linefunc_sample_delta_matrix(u_samples, mode=mode)

    for idx, delta in enumerate(deltas_all):
        kernel_weights = linefunc_kernel_weights(
            delta,
            window_span_percent,
            kernel=kernel,
        )
        weights = kernel_weights * spacing_weights
        is_active = weights > min_weight

        if not np.any(is_active):
            nearest_idx = int(np.argmin(np.abs(delta)))
            output[idx] = values_flat[nearest_idx]
            continue

        degree = min(int(order), int(np.count_nonzero(is_active)) - 1)
        if degree <= 0:
            active_weights = weights[is_active]
            output[idx] = np.average(
                values_flat[is_active],
                axis=0,
                weights=active_weights,
            )
            continue

        x_active = delta[is_active]
        y_active = values_flat[is_active]
        sqrt_weights = np.sqrt(weights[is_active])
        design = np.vander(x_active, N=degree + 1, increasing=True)
        design_weighted = design * sqrt_weights[:, np.newaxis]
        y_weighted = y_active * sqrt_weights[:, np.newaxis]
        try:
            coeffs = np.linalg.lstsq(design_weighted, y_weighted, rcond=None)[0]
            output[idx] = coeffs[0]
        except np.linalg.LinAlgError:
            output[idx] = np.average(
                y_active,
                axis=0,
                weights=weights[is_active],
            )

    return output.reshape(values.shape)


def linefunc_build_smoothed_interpolator(
    u_samples,
    values,
    *,
    window_ratio: Number,
    order: int,
    mode: Literal["interp", "wrap"] = "interp",
    spacing_weights=None,
    kernel: Literal["boxcar", "tricube", "triangular", "gaussian"] = "boxcar",
    interp_kind: str = "linear",
    min_weight: float = 1e-12,
) -> tuple[interp1d, np.ndarray]:
    """
    Build a smooth interpolator from non-uniform line-function samples.

    The returned tuple is `(interpolator, values_smooth)`. The interpolator
    accepts arbitrary u-percent query points. `values_smooth` is returned so the
    caller can cache or inspect the smoothed support values separately.
    """
    u_samples = _linefunc_as_u_samples(u_samples)
    mode = as_str(mode, name="line function smoothing mode", pool=("interp", "wrap"))
    _linefunc_validate_wrap_endpoint(u_samples, mode)

    values_smooth = linefunc_smooth_values(
        u_samples,
        values,
        window_ratio=window_ratio,
        order=order,
        mode=mode,
        spacing_weights=spacing_weights,
        kernel=kernel,
        min_weight=min_weight,
    )

    if mode == "wrap":
        u_interp = np.concatenate([u_samples - 100.0, u_samples, u_samples + 100.0])
        values_interp = np.concatenate(
            [values_smooth, values_smooth, values_smooth],
            axis=0,
        )
    else:
        u_interp = u_samples
        values_interp = values_smooth

    interpolator = interp1d(
        u_interp,
        values_interp,
        axis=0,
        kind=interp_kind,
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    return interpolator, values_smooth


def _linefunc_sample_delta_matrix(
    u_samples,
    mode: Literal["interp", "wrap"] = "interp",
) -> np.ndarray:
    """Return sample-to-sample deltas for smoothing at the sample locations."""
    u_samples = _linefunc_as_u_samples(u_samples)
    mode = as_str(mode, name="line function smoothing mode", pool=("interp", "wrap"))
    _linefunc_validate_wrap_endpoint(u_samples, mode)

    delta = u_samples[np.newaxis, :] - u_samples[:, np.newaxis]
    if mode == "wrap":
        delta = (delta + 50.0) % 100.0 - 50.0
    return delta


def _linefunc_as_u_samples(u_samples) -> np.ndarray:
    """Validate sorted, unique u-percent samples for line-function smoothing."""
    u_samples = np.asarray(u_samples, dtype=float).reshape(-1)
    if u_samples.ndim != 1 or len(u_samples) == 0:
        raise ValueError("u_samples must be a non-empty one-dimensional array.")
    if np.any(~np.isfinite(u_samples)):
        raise ValueError("u_samples must contain only finite values.")
    if np.min(u_samples) < 0 or np.max(u_samples) > 100:
        raise ValueError("u_samples must stay within the range [0, 100].")
    if np.any(np.diff(u_samples) <= 0):
        raise ValueError("u_samples must be strictly increasing with no duplicates.")
    return u_samples


def _linefunc_validate_wrap_endpoint(
    u_samples: np.ndarray,
    mode: Literal["interp", "wrap"],
) -> None:
    """Reject duplicate periodic endpoints in wrap mode."""
    if mode != "wrap" or len(u_samples) < 2:
        return
    if np.isclose(u_samples[0], 0.0) and np.isclose(u_samples[-1], 100.0):
        raise ValueError(
            "wrap mode treats u_percent=0 and u_percent=100 as the same point. "
            "Provide only one of these endpoints."
        )


class SmoothedLineFunc(ClassBase):
    """
    Sample and interpolate a numerical function along one SmoothedLine.

    Users provide a callable `func(u_percent, **func_kwargs)` together with
    normalized sample locations in `[0, 100]`. The object evaluates that
    callable on the current line parameter domain, stores sampled outputs, and
    exposes a linear interpolator for later reuse.

    The sampling mode follows the current owner opts mode:

    - `"interp"`: interpolate directly over the sampled range.
    - `"wrap"`: tile the samples across `[-100, 0, 100]` offsets so periodic
      evaluation remains continuous across the wrap boundary.
    """

    # fmt: off
    __attr_defs__ = {
        "owner": AttrDef(
            doc="The SmoothedLine instance that this function is associated with.",
            kind="relation",
            is_weak_by_default=True,
        ),
        "raw_func": AttrDef(
            doc=(
                "Numerical sampling function mapping one u_percent to a value "
                "or a (value, metric) / (value, metric, payload_samples) / "
                "(value, metric, payload_samples, payload_shared) tuple."
            ),
            kind="raw",
            validator=lambda v, d: v if callable(v) else (_raise_type_error(d, v)),
        ),
        "raw_u_samples": AttrDef(
            doc="Sampling locations in u_percent used to evaluate the numerical function.",
            kind="raw",
            validator=lambda v, d: SmoothedLineFunc._helper_validate_u_samples(v, name=d),
        ),
        "raw_func_kwargs": AttrDef(
            doc="Extra keyword arguments passed to the numerical function during sampling.",
            kind="raw",
            validator=lambda v, d: SmoothedLineFunc._helper_validate_func_kwargs(v, name=d),
        ),
        "state_is_follow_owner_opts": AttrDef(
            doc=(
                "Whether owner opts changes should automatically refresh this "
                "function before interpolation."
            ),
            kind="state",
            validator=lambda v, d: as_bool(v, name=d),
        ),
        "impl_owner_opts_snapshot": AttrDef(
            doc=(
                "Snapshot of owner opts and line-function mode at the time "
                "this line function was last sampled."
            ),
            kind="impl",
        ),
        "calc_values": AttrDef(
            doc="Values returned by the numerical function at each sampling location.",
            kind="calc",
        ),
        "calc_metrics": AttrDef(
            doc="Per-sample metrics returned by the numerical function, or None if unavailable.",
            kind="calc",
        ),
        "calc_payload_samples": AttrDef(
            doc="Per-sample payload objects returned by the numerical function, or None if unavailable.",
            kind="calc",
        ),
        "calc_payload_shared": AttrDef(
            doc="Shared payload returned for the full sampled function, or None if unavailable.",
            kind="calc",
        ),
        "entity_interpolator": AttrDef(
            doc="Interpolator object built from the sampled values.",
            kind="entity",
        ),
    }
    # fmt: on

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
        and name not in ClassBase.__slots__
    )

    # -------------------------------
    # Validation and owner-state helpers
    # -------------------------------

    @staticmethod
    def _helper_validate_u_samples(
        u_samples,
        *,
        name: str = "`u_samples`",
    ) -> np.ndarray:
        u_samples = np.asarray(u_samples, dtype=float).reshape(-1)
        if u_samples.ndim != 1 or len(u_samples) == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional array.")
        if np.any(~np.isfinite(u_samples)):
            raise ValueError(f"{name} must contain only finite values.")
        if np.min(u_samples) < 0 or np.max(u_samples) > 100:
            raise ValueError(f"{name} must stay within the range [0, 100].")
        u_samples = np.unique(np.sort(u_samples))
        if len(u_samples) == 0:
            raise ValueError(
                f"{name} must remain non-empty after sorting and deduplication."
            )
        return u_samples

    @staticmethod
    def _helper_validate_func_kwargs(
        func_kwargs,
        *,
        name: str = "`func_kwargs`",
    ) -> dict[str, Any]:
        if func_kwargs is None:
            return {}
        if not isinstance(func_kwargs, Mapping):
            raise TypeError(f"{name} must be a mapping or None.")
        return dict(func_kwargs)

    def _helper_get_owner_mode_from(self, opts_dict):
        owner_mode = None if opts_dict is None else opts_dict.get("mode", None)
        return (
            "interp"
            if owner_mode is None
            else as_str(
                owner_mode, name="owner smoothing mode", pool=("interp", "wrap")
            )
        )

    def _helper_get_owner_linefunc_mode_from(self, opts_dict):
        linefunc_mode = (
            None if opts_dict is None else opts_dict.get("linefunc_mode", None)
        )
        if linefunc_mode is None:
            return self._helper_get_owner_mode_from(opts_dict)
        return as_str(
            linefunc_mode,
            name="owner line-function interpolation mode",
            pool=("interp", "wrap"),
        )

    def _helper_get_owner_opts_snapshot(self, owner):
        opts_snapshot = dict(owner.opts.act_asdict())
        linefunc_mode = getattr(owner, "linefunc_mode", opts_snapshot.get("mode"))
        opts_snapshot["linefunc_mode"] = as_str(
            linefunc_mode,
            name="owner line-function interpolation mode",
            pool=("interp", "wrap"),
        )
        return opts_snapshot

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_normalize_u_samples_for_mode(self, u_samples, mode, logger=None):
        """Normalize user-facing u-sample endpoint duplication for wrap mode."""
        u_samples = self._helper_validate_u_samples(
            u_samples,
            name=type(self).__attr_defs__["raw_u_samples"].doc,
        )
        if mode == "wrap" and len(u_samples) >= 2:
            if np.isclose(u_samples[0], 0.0) and np.isclose(u_samples[-1], 100.0):
                logger.warning(
                    "wrap mode treats `u_percent=0` and `u_percent=100` as the "
                    "same point. Automatically removing the `100` endpoint from "
                    "the sampled line function."
                )
                u_samples = u_samples[:-1]
        return u_samples

    # -------------------------------
    # Initialization
    # -------------------------------

    def __init__(
        self,
        func,
        u_samples,
        owner: SmoothedLine,
        func_kwargs: Mapping[str, Any] | None = None,
        is_follow_owner_opts: bool = True,
        name: str = "smoothed line function",
    ):
        super().__init__(name=name, name_replace="smoothed line function")

        if not isinstance(owner, SmoothedLine):
            raise TypeError("`owner` for SmoothedLineFunc must be a SmoothedLine.")

        object.__setattr__(
            self,
            "raw_func",
            type(self)
            .__attr_defs__["raw_func"]
            .validator(
                func,
                type(self).__attr_defs__["raw_func"].doc,
            ),
        )
        object.__setattr__(
            self,
            "raw_u_samples",
            type(self)
            .__attr_defs__["raw_u_samples"]
            .validator(
                u_samples,
                type(self).__attr_defs__["raw_u_samples"].doc,
            ),
        )
        object.__setattr__(
            self,
            "raw_func_kwargs",
            type(self)
            .__attr_defs__["raw_func_kwargs"]
            .validator(
                func_kwargs,
                type(self).__attr_defs__["raw_func_kwargs"].doc,
            ),
        )
        object.__setattr__(
            self,
            "state_is_follow_owner_opts",
            type(self)
            .__attr_defs__["state_is_follow_owner_opts"]
            .validator(
                is_follow_owner_opts,
                type(self).__attr_defs__["state_is_follow_owner_opts"].doc,
            ),
        )
        object.__setattr__(self, "impl_owner_opts_snapshot", None)
        object.__setattr__(self, "calc_values", None)
        object.__setattr__(self, "calc_metrics", None)
        object.__setattr__(self, "calc_payload_samples", None)
        object.__setattr__(self, "calc_payload_shared", None)
        object.__setattr__(self, "entity_interpolator", None)

        self.act_bind_relation_base("owner", owner, is_weak=True)
        self.act_refresh()

    # ==================== OVERRIDE ====================
    # SmoothedLineFunc overrides ClassBase.__setattr__ so public raw_/state_
    # changes immediately rebuild sampled values and the dependent interpolator.
    # ==================================================
    def __setattr__(self, key, value):
        attr_defs = type(self).__attr_defs__
        try:
            object.__getattribute__(self, "impl_assign_state")
        except AttributeError:
            super().__setattr__(key, value)
            return

        target_key = key
        if target_key not in attr_defs and target_key not in self.impl_extra:
            raw_key = f"raw_{key}"
            if raw_key in attr_defs:
                target_key = raw_key

        super().__setattr__(key, value)

        if not target_key.startswith(("raw_", "state_")):
            return
        if target_key == "raw_name":
            return
        if getattr(self, "entity_interpolator", None) is None:
            return
        self.act_refresh()

    # -------------------------------
    # Sampling and refresh helpers
    # -------------------------------

    def _helper_get_owner_opts_comparison(self):
        owner = self.owner
        opts_then = self.impl_owner_opts_snapshot
        opts_now = (
            None if owner is None else self._helper_get_owner_opts_snapshot(owner)
        )

        if opts_then is None or opts_now is None:
            diff_then = {}
            diff_now = {}
            is_stale = False
        else:
            diff_then, diff_now = diff_dict_values(opts_then, opts_now)
            is_stale = bool(diff_then or diff_now)

        lines = [f"Smoothed line function {self.name!r} owner opts comparison:"]
        lines.append(f"Stored opts snapshot: {opts_then!r}")
        lines.append(f"Current owner opts: {opts_now!r}")
        if owner is None:
            lines.append("Owner relation is currently unavailable.")
        elif opts_then is None:
            lines.append(
                "No stored opts snapshot is available yet. Call `act_refresh(...)` first."
            )
        elif is_stale:
            lines.append(f"Stored opts diff: {diff_then}")
            lines.append(f"Current owner opts diff: {diff_now}")
            lines.append(
                "Stored snapshot differs from the current owner opts. "
                "Consider calling `act_refresh(...)`."
            )
        else:
            lines.append("Stored snapshot matches the current owner opts.")

        return {
            "owner": owner,
            "opts_then": opts_then,
            "opts_now": opts_now,
            "diff_then": diff_then,
            "diff_now": diff_now,
            "is_stale": is_stale,
            "message": "\n".join(lines),
        }

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_warn_if_owner_opts_changed(self, logger=None):
        comparison = self._helper_get_owner_opts_comparison()
        if not comparison["is_stale"]:
            return
        logger.warning(comparison["message"])

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_refresh_if_owner_opts_changed(self, logger=None):
        comparison = self._helper_get_owner_opts_comparison()
        if not comparison["is_stale"]:
            return
        if self.state_is_follow_owner_opts:
            logger.info(
                f"Owner opts changed for {self.name!r}; "
                "refreshing sampled values and interpolator."
            )
            self.act_refresh()
            return
        logger.warning(comparison["message"])

    @logging_and_warning_decorator(start_finish_level=5)
    def act_refresh(
        self,
        u_samples=None,
        func=None,
        func_kwargs: Mapping[str, Any] | None = None,
        logger=None,
    ):
        owner = self.owner
        if owner is None:
            raise RuntimeError(
                "Cannot refresh a SmoothedLineFunc without a live owner."
            )

        opts_snapshot = self._helper_get_owner_opts_snapshot(owner)
        mode = self._helper_get_owner_linefunc_mode_from(opts_snapshot)

        if u_samples is not None:
            object.__setattr__(
                self,
                "raw_u_samples",
                self._helper_normalize_u_samples_for_mode(
                    u_samples,
                    mode,
                ),
            )
        else:
            object.__setattr__(
                self,
                "raw_u_samples",
                self._helper_normalize_u_samples_for_mode(
                    self.raw_u_samples,
                    mode,
                ),
            )
        if func is not None:
            object.__setattr__(
                self,
                "raw_func",
                type(self)
                .__attr_defs__["raw_func"]
                .validator(
                    func,
                    type(self).__attr_defs__["raw_func"].doc,
                ),
            )
        if func_kwargs is not None:
            object.__setattr__(
                self,
                "raw_func_kwargs",
                self._helper_validate_func_kwargs(
                    func_kwargs,
                    name=type(self).__attr_defs__["raw_func_kwargs"].doc,
                ),
            )

        values = []
        metrics = []
        payload_samples = []
        is_has_metric = False
        is_has_payload_samples = False
        payload_shared = None
        is_has_payload_shared = False
        for u in self.raw_u_samples:
            sample_result = self.raw_func(float(u), **self.raw_func_kwargs)
            if isinstance(sample_result, tuple) and len(sample_result) == 4:
                value, metric, payload_sample, payload_shared_i = sample_result
            elif isinstance(sample_result, tuple) and len(sample_result) == 3:
                value, metric, payload_sample = sample_result
                payload_shared_i = None
            elif isinstance(sample_result, tuple) and len(sample_result) == 2:
                value, metric = sample_result
                payload_sample = None
                payload_shared_i = None
            else:
                value, metric, payload_sample, payload_shared_i = (
                    sample_result,
                    None,
                    None,
                    None,
                )
            values.append(np.asarray(value))
            metrics.append(metric)
            payload_samples.append(payload_sample)
            is_has_metric = is_has_metric or (metric is not None)
            is_has_payload_samples = is_has_payload_samples or (
                payload_sample is not None
            )
            if payload_shared_i is not None:
                if not is_has_payload_shared:
                    payload_shared = payload_shared_i
                    is_has_payload_shared = True
                elif payload_shared != payload_shared_i:
                    raise ValueError(
                        "Shared payload returned by `raw_func` must remain identical "
                        "across all sampled `u_percent` values."
                    )

        values = np.stack(values, axis=0)
        metrics = metrics if is_has_metric else None
        payload_samples = payload_samples if is_has_payload_samples else None
        payload_shared = payload_shared if is_has_payload_shared else None

        interpolator, values_smooth = linefunc_build_smoothed_interpolator(
            self.raw_u_samples,
            values,
            window_ratio=opts_snapshot["window_ratio"],
            order=opts_snapshot["order"],
            mode=mode,
        )

        object.__setattr__(self, "impl_owner_opts_snapshot", dict(opts_snapshot))
        object.__setattr__(self, "calc_values", values_smooth)
        object.__setattr__(self, "calc_metrics", metrics)
        object.__setattr__(self, "calc_payload_samples", payload_samples)
        object.__setattr__(self, "calc_payload_shared", payload_shared)
        object.__setattr__(self, "entity_interpolator", interpolator)
        return self

    # -------------------------------
    # Public evaluation actions
    # -------------------------------

    def interpolate(self, u_percent):
        if self.entity_interpolator is None:
            raise RuntimeError(
                "SmoothedLineFunc has no interpolator yet. Call `act_refresh()` first."
            )

        self._helper_refresh_if_owner_opts_changed()
        u_percent = np.asarray(u_percent, dtype=float)
        mode = self._helper_get_owner_linefunc_mode_from(self.impl_owner_opts_snapshot)
        if mode == "wrap":
            u_percent = np.mod(u_percent, 100.0)
        return self.entity_interpolator(u_percent)

    def __call__(self, u_percent):
        return self.interpolate(u_percent)

    @logging_and_warning_decorator(start_finish_level=5)
    def show_owner_opts_snapshot(self, is_return=False, logger=None):
        comparison = self._helper_get_owner_opts_comparison()
        logger.info(comparison["message"])
        if is_return:
            return comparison["message"]
        return None

    # -------------------------------
    # Representation
    # -------------------------------

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        mode = self._helper_get_owner_linefunc_mode_from(self.impl_owner_opts_snapshot)
        return (
            f"{cls_name}({self.name!r}), num_samples={len(self.raw_u_samples)}, "
            f"mode={mode!r}"
        )
