from typing import Any, Literal, Mapping

import numpy as np
from scipy.interpolate import interp1d

from ..datatypes import (
    Number,
    as_bool,
    as_number,
    as_str,
)
from ..core.class_base import AttrDef, ClassBase
from ..core.opts import diff_dict_values
from ..core.result_base import ResultBase
from ..geometry.smoothing.line import (
    LineSmoothingConfigError as LineSmoothingConfigError,
    OptsSmoothedLine as OptsSmoothedLine,
    SmoothedLine,
)
from ..logging_decorator import logging_and_warning_decorator

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

    for idx, u_center in enumerate(u_samples):
        delta = u_samples - u_center
        if mode == "wrap":
            delta = (delta + 50.0) % 100.0 - 50.0

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
    normalized sample locations in `[0, 100]`. The callable must return a
    ResultBase instance at every sample. The configured `result_value_attr`
    (default `"value"`) selects which result attribute is smoothed and
    interpolated, while the complete raw result objects remain available in
    `calc_results`.

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
                "Numerical sampling function mapping one u_percent to a "
                "ResultBase instance."
            ),
            kind="raw",
            validator=lambda v, d: v if callable(v) else (_raise_type_error(d, v)),
        ),
        "raw_result_value_attr": AttrDef(
            doc=(
                "ResultBase attribute whose per-sample value is smoothed and "
                "interpolated."
            ),
            kind="raw",
            validator=lambda v, d: SmoothedLineFunc._helper_validate_result_value_attr(
                v, name=d
            ),
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
        "calc_results": AttrDef(
            doc="Raw ResultBase objects returned at each sampling location.",
            kind="calc",
        ),
        "calc_values": AttrDef(
            doc=(
                "Smoothed sample values extracted from the configured "
                "ResultBase attribute."
            ),
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
    def _helper_validate_result_value_attr(
        result_value_attr,
        *,
        name: str = "`result_value_attr`",
    ) -> str:
        result_value_attr = as_str(result_value_attr, name=name)
        if not result_value_attr:
            raise ValueError(f"{name} must be a non-empty string.")
        return result_value_attr

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
        result_value_attr: str = "value",
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
            "raw_result_value_attr",
            type(self)
            .__attr_defs__["raw_result_value_attr"]
            .validator(
                result_value_attr,
                type(self).__attr_defs__["raw_result_value_attr"].doc,
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
        object.__setattr__(self, "calc_results", None)
        object.__setattr__(self, "calc_values", None)
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

        results = []
        values = []
        for u in self.raw_u_samples:
            u_float = float(u)
            sample_result = self.raw_func(u_float, **self.raw_func_kwargs)
            if not isinstance(sample_result, ResultBase):
                raise TypeError(
                    "SmoothedLineFunc `raw_func` must return a ResultBase instance "
                    f"at every sample; got {type(sample_result).__name__} at "
                    f"u_percent={u_float}."
                )
            if not hasattr(sample_result, self.raw_result_value_attr):
                raise AttributeError(
                    f"SmoothedLineFunc `raw_func` returned "
                    f"{type(sample_result).__name__} at u_percent={u_float}, but "
                    f"it has no attribute {self.raw_result_value_attr!r}. Set "
                    "`result_value_attr` to the ResultBase attribute that should "
                    "be smoothed."
                )

            results.append(sample_result)
            values.append(
                np.asarray(getattr(sample_result, self.raw_result_value_attr))
            )

        values = np.stack(values, axis=0)

        interpolator, values_smooth = linefunc_build_smoothed_interpolator(
            self.raw_u_samples,
            values,
            window_ratio=opts_snapshot["window_ratio"],
            order=opts_snapshot["order"],
            mode=mode,
        )

        object.__setattr__(self, "impl_owner_opts_snapshot", dict(opts_snapshot))
        object.__setattr__(self, "calc_results", tuple(results))
        object.__setattr__(self, "calc_values", values_smooth)
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
