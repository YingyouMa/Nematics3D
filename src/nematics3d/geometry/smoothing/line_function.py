"""Sampling, smoothing, and interpolation of values along a SmoothedLine."""

from __future__ import annotations

from typing import Any, Literal, Mapping

import numpy as np
from scipy.interpolate import interp1d

from ...core.class_base import AttrDef, ClassBase
from ...core.result_base import ResultBase
from ...datatypes import Number, as_bool, as_number, as_str
from ...logging_decorator import logging_and_warning_decorator
from .line import SmoothedLine


def _raise_type_error(name: str, value: Any):
    raise TypeError(f"{name} must be callable, got {type(value).__name__}.")


def _linefunc_as_u_samples(u_samples) -> np.ndarray:
    u_samples = np.asarray(u_samples, dtype=float)
    if u_samples.ndim != 1 or len(u_samples) == 0:
        raise ValueError("u_samples must be a non-empty one-dimensional array.")
    if np.any(~np.isfinite(u_samples)):
        raise ValueError("u_samples must contain only finite values.")
    if np.min(u_samples) < 0 or np.max(u_samples) > 100:
        raise ValueError("u_samples must stay within the range [0, 100].")
    if np.any(np.diff(u_samples) <= 0):
        raise ValueError("u_samples must be strictly increasing with no duplicates.")
    return u_samples


def _linefunc_validate_wrap_endpoint(u_samples, mode):
    if mode == "wrap" and len(u_samples) >= 2:
        if np.isclose(u_samples[0], 0.0) and np.isclose(u_samples[-1], 100.0):
            raise ValueError(
                "wrap mode treats u_percent=0 and u_percent=100 as the same point. "
                "Provide only one of these endpoints."
            )


def linefunc_window_span_percent(*, window_ratio: Number) -> float:
    window_ratio = as_number(
        window_ratio,
        name="line function window_ratio",
        value_range=(1e-12, np.inf),
    )
    return 100.0 / float(window_ratio)


def linefunc_spacing_weights(u_samples, mode: Literal["interp", "wrap"] = "interp"):
    u_samples = _linefunc_as_u_samples(u_samples)
    mode = as_str(mode, name="line function smoothing mode", pool=("interp", "wrap"))
    _linefunc_validate_wrap_endpoint(u_samples, mode)
    if len(u_samples) == 1:
        return np.array([100.0], dtype=float)
    if mode == "wrap":
        prev_samples = np.roll(u_samples, 1)
        next_samples = np.roll(u_samples, -1)
        return 0.5 * (
            np.mod(u_samples - prev_samples, 100.0)
            + np.mod(next_samples - u_samples, 100.0)
        )
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
):
    delta = np.asarray(delta, dtype=float)
    window_span_percent = as_number(
        window_span_percent,
        name="line function window span in percent",
        value_range=(1e-12, np.inf),
    )
    radius = 0.5 * float(window_span_percent)
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
):
    u_samples = _linefunc_as_u_samples(u_samples)
    mode = as_str(mode, name="line function smoothing mode", pool=("interp", "wrap"))
    _linefunc_validate_wrap_endpoint(u_samples, mode)
    values = np.asarray(values)
    if values.ndim == 0 or values.shape[0] != len(u_samples):
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
        spacing_weights = np.asarray(spacing_weights, dtype=float)
        if spacing_weights.shape != u_samples.shape:
            raise ValueError("spacing_weights must have the same shape as u_samples.")
        if np.any(spacing_weights < 0) or np.any(~np.isfinite(spacing_weights)):
            raise ValueError("spacing_weights must be finite and non-negative.")
    values_flat = values.reshape(len(u_samples), -1)
    output = np.empty_like(values_flat, dtype=float)
    for idx, u_center in enumerate(u_samples):
        delta = u_samples - u_center
        if mode == "wrap":
            delta = (delta + 50.0) % 100.0 - 50.0
        weights = (
            linefunc_kernel_weights(delta, window_span_percent, kernel=kernel)
            * spacing_weights
        )
        active = weights > min_weight
        if not np.any(active):
            output[idx] = values_flat[int(np.argmin(np.abs(delta)))]
            continue
        degree = min(int(order), int(np.count_nonzero(active)) - 1)
        if degree <= 0:
            output[idx] = np.average(
                values_flat[active], axis=0, weights=weights[active]
            )
            continue
        x_active = delta[active]
        y_active = values_flat[active]
        sqrt_weights = np.sqrt(weights[active])
        design = np.vander(x_active, N=degree + 1, increasing=True)
        try:
            coeffs = np.linalg.lstsq(
                design * sqrt_weights[:, None],
                y_active * sqrt_weights[:, None],
                rcond=None,
            )[0]
            output[idx] = coeffs[0]
        except np.linalg.LinAlgError:
            output[idx] = np.average(y_active, axis=0, weights=weights[active])
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
):
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
        values_interp = np.concatenate([values_smooth] * 3, axis=0)
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


class SmoothedLineFunc(ClassBase):
    """Sample, smooth, and interpolate a numerical function along one SmoothedLine."""

    __attr_defs__ = {
        "owner": AttrDef(
            doc="The SmoothedLine instance that this function is associated with.",
            kind="relation",
            is_weak_by_default=True,
        ),
        "raw_func": AttrDef(
            doc="Numerical sampling function evaluated at each u_percent.",
            kind="raw",
            validator=lambda v, d: v if callable(v) else (_raise_type_error(d, v)),
        ),
        "raw_result_value_attr": AttrDef(
            doc="ResultBase attribute selected when a sample returns ResultBase.",
            kind="raw",
            validator=lambda v, d: SmoothedLineFunc._helper_validate_result_value_attr(
                v, name=d
            ),
        ),
        "raw_u_samples": AttrDef(
            doc="Sampling locations in u_percent used to evaluate the numerical function.",
            kind="raw",
            validator=lambda v, d: SmoothedLineFunc._helper_validate_u_samples(
                v, name=d
            ),
        ),
        "raw_func_kwargs": AttrDef(
            doc="Extra keyword arguments passed to the numerical function during sampling.",
            kind="raw",
            validator=lambda v, d: SmoothedLineFunc._helper_validate_func_kwargs(
                v, name=d
            ),
        ),
        "state_is_follow_owner_opts": AttrDef(
            doc="Whether relevant owner-option changes trigger lazy refresh.",
            kind="state",
            validator=lambda v, d: as_bool(v, name=d),
        ),
        "impl_owner_opts_snapshot": AttrDef(
            doc="Relevant owner dependency signature at the last refresh.", kind="impl"
        ),
        "impl_is_stale": AttrDef(
            doc="Whether cached samples need refresh.", kind="impl"
        ),
        "calc_results": AttrDef(
            doc="Raw objects returned at each sample location.", kind="calc"
        ),
        "calc_values": AttrDef(doc="Smoothed numerical support values.", kind="calc"),
        "entity_interpolator": AttrDef(
            doc="Interpolator built from calc_values.", kind="entity"
        ),
    }

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
        and name not in ClassBase.__slots__
    )

    @staticmethod
    def _helper_validate_u_samples(u_samples, *, name="`u_samples`"):
        try:
            return _linefunc_as_u_samples(u_samples)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc

    @staticmethod
    def _helper_validate_result_value_attr(
        result_value_attr, *, name="`result_value_attr`"
    ):
        result_value_attr = as_str(result_value_attr, name=name)
        if not result_value_attr:
            raise ValueError(f"{name} must be a non-empty string.")
        return result_value_attr

    @staticmethod
    def _helper_validate_func_kwargs(func_kwargs, *, name="`func_kwargs`"):
        if func_kwargs is None:
            return {}
        if not isinstance(func_kwargs, Mapping):
            raise TypeError(f"{name} must be a mapping or None.")
        return dict(func_kwargs)

    def _helper_get_owner_opts_snapshot(self, owner):
        return {
            "mode": owner.opts.mode,
            "window_ratio": owner.opts.window_ratio,
            "order": owner.opts.order,
            "entity_tck_id": id(owner.entity_tck),
            "linefunc_mode": as_str(
                owner.linefunc_mode,
                name="owner line-function interpolation mode",
                pool=("interp", "wrap"),
            ),
        }

    def _helper_owner_opts_changed(self):
        owner = self.owner
        if owner is None or self.impl_owner_opts_snapshot is None:
            return False
        current = self._helper_get_owner_opts_snapshot(owner)
        relevant = ("window_ratio", "order", "entity_tck_id", "linefunc_mode")
        return any(
            current[key] != self.impl_owner_opts_snapshot.get(key) for key in relevant
        )

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_normalize_u_samples_for_mode(self, u_samples, mode, logger=None):
        u_samples = self._helper_validate_u_samples(
            u_samples, name=type(self).__attr_defs__["raw_u_samples"].doc
        )
        if mode == "wrap" and len(u_samples) >= 2:
            if np.isclose(u_samples[0], 0.0) and np.isclose(u_samples[-1], 100.0):
                logger.warning(
                    "wrap mode treats `u_percent=0` and `u_percent=100` as the "
                    "same point. Automatically removing the `100` endpoint."
                )
                u_samples = u_samples[:-1]
        return u_samples

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
            self.__attr_defs__["raw_func"].validator(
                func, self.__attr_defs__["raw_func"].doc
            ),
        )
        object.__setattr__(
            self,
            "raw_result_value_attr",
            self.__attr_defs__["raw_result_value_attr"].validator(
                result_value_attr, self.__attr_defs__["raw_result_value_attr"].doc
            ),
        )
        object.__setattr__(
            self,
            "raw_u_samples",
            self.__attr_defs__["raw_u_samples"].validator(
                u_samples, self.__attr_defs__["raw_u_samples"].doc
            ),
        )
        object.__setattr__(
            self,
            "raw_func_kwargs",
            self.__attr_defs__["raw_func_kwargs"].validator(
                func_kwargs, self.__attr_defs__["raw_func_kwargs"].doc
            ),
        )
        object.__setattr__(
            self,
            "state_is_follow_owner_opts",
            self.__attr_defs__["state_is_follow_owner_opts"].validator(
                is_follow_owner_opts,
                self.__attr_defs__["state_is_follow_owner_opts"].doc,
            ),
        )
        object.__setattr__(self, "impl_owner_opts_snapshot", None)
        object.__setattr__(self, "impl_is_stale", True)
        object.__setattr__(self, "calc_results", None)
        object.__setattr__(self, "calc_values", None)
        object.__setattr__(self, "entity_interpolator", None)
        self.act_bind_relation_base("owner", owner, is_weak=True)
        self.act_refresh()

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
        if target_key.startswith("raw_") and target_key != "raw_name":
            object.__setattr__(self, "impl_is_stale", True)

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_refresh_if_needed(self, logger=None):
        owner_changed = self._helper_owner_opts_changed()
        if owner_changed and not self.state_is_follow_owner_opts:
            logger.warning(
                f"Owner smoothing options changed for {self.name!r}; cached values "
                "are retained because state_is_follow_owner_opts=False."
            )
            owner_changed = False
        if self.impl_is_stale or owner_changed:
            self.act_refresh()

    @logging_and_warning_decorator(start_finish_level=5)
    def act_refresh(self, u_samples=None, func=None, func_kwargs=None, logger=None):
        owner = self.owner
        if owner is None:
            raise RuntimeError(
                "Cannot refresh a SmoothedLineFunc without a live owner."
            )
        opts_snapshot = self._helper_get_owner_opts_snapshot(owner)
        mode = opts_snapshot["linefunc_mode"]
        if u_samples is not None:
            object.__setattr__(
                self,
                "raw_u_samples",
                self._helper_normalize_u_samples_for_mode(u_samples, mode),
            )
        else:
            object.__setattr__(
                self,
                "raw_u_samples",
                self._helper_normalize_u_samples_for_mode(self.raw_u_samples, mode),
            )
        if func is not None:
            object.__setattr__(
                self,
                "raw_func",
                self.__attr_defs__["raw_func"].validator(
                    func, self.__attr_defs__["raw_func"].doc
                ),
            )
        if func_kwargs is not None:
            object.__setattr__(
                self,
                "raw_func_kwargs",
                self._helper_validate_func_kwargs(
                    func_kwargs, name=self.__attr_defs__["raw_func_kwargs"].doc
                ),
            )

        results = []
        values = []
        for u in self.raw_u_samples:
            sample_result = self.raw_func(float(u), **self.raw_func_kwargs)
            results.append(sample_result)
            if isinstance(sample_result, ResultBase):
                if not hasattr(sample_result, self.raw_result_value_attr):
                    raise AttributeError(
                        f"SmoothedLineFunc `raw_func` returned {type(sample_result).__name__} "
                        f"at u_percent={float(u)}, but it has no attribute "
                        f"{self.raw_result_value_attr!r}."
                    )
                sample_value = getattr(sample_result, self.raw_result_value_attr)
            else:
                sample_value = sample_result
            try:
                values.append(np.asarray(sample_value, dtype=float))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "SmoothedLineFunc samples must be numeric/array-like or ResultBase "
                    "instances containing a numeric selected attribute."
                ) from exc
        try:
            values = np.stack(values, axis=0)
        except ValueError as exc:
            raise ValueError(
                "SmoothedLineFunc sample values must have a consistent shape."
            ) from exc
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
        object.__setattr__(self, "impl_is_stale", False)
        return self

    def interpolate(self, u_percent):
        self._helper_refresh_if_needed()
        if self.entity_interpolator is None:
            raise RuntimeError(
                "SmoothedLineFunc has no interpolator yet. Call `act_refresh()` first."
            )
        u_percent = np.asarray(u_percent, dtype=float)
        if self.impl_owner_opts_snapshot["linefunc_mode"] == "wrap":
            u_percent = np.mod(u_percent, 100.0)
        return self.entity_interpolator(u_percent)

    def __call__(self, u_percent):
        return self.interpolate(u_percent)

    @logging_and_warning_decorator(start_finish_level=5)
    def show_owner_opts_snapshot(self, is_return=False, logger=None):
        owner = self.owner
        current = None if owner is None else self._helper_get_owner_opts_snapshot(owner)
        message = (
            f"Smoothed line function {self.name!r} owner dependency snapshot:\n"
            f"Stored: {self.impl_owner_opts_snapshot!r}\nCurrent: {current!r}"
        )
        logger.info(message)
        if is_return:
            return message
        return None

    def __repr__(self):
        mode = (
            None
            if self.impl_owner_opts_snapshot is None
            else self.impl_owner_opts_snapshot["linefunc_mode"]
        )
        return (
            f"{self.__class__.__name__}({self.name!r}), "
            f"num_samples={len(self.raw_u_samples)}, mode={mode!r}"
        )


__all__ = [
    "SmoothedLineFunc",
    "linefunc_build_smoothed_interpolator",
    "linefunc_kernel_weights",
    "linefunc_smooth_values",
    "linefunc_spacing_weights",
    "linefunc_window_span_percent",
]
