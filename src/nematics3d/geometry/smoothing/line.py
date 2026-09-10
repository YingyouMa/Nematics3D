"""Savitzky-Golay smoothing and spline parameterization for polylines."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.signal import savgol_filter

from ...core.class_base import AttrDef
from ...core.host_base import HostBase, OptsBase
from ...core.opts import cover_value
from ...core.registry_base import RegistryBase
from ...datatypes import (
    Number,
    UNSET,
    Unset,
    as_bool,
    as_number,
    as_points,
    as_readonly_array,
    as_str,
)
from ...logging_decorator import logging_and_warning_decorator


# fmt: off
@dataclass(slots=True, repr=False)
class OptsSmoothedLine(OptsBase):
    """Options controlling Savitzky-Golay-based line smoothing."""

    window_ratio:               Number | None | Unset               = UNSET
    window_length:              int | None | Unset                  = UNSET
    order:                      int | Unset                         = UNSET
    num_out_ratio:              Number | Unset                      = UNSET
    mode:                       Literal["interp", "wrap"] | Unset   = UNSET
    min_line_length:            int | Unset                         = UNSET

    __attrs__ = {
        **OptsBase.__attrs__,
        "window_ratio":         "window ratio for smoothing: line_length / window_length",
        "window_length":        "explicit window length for smoothing",
        "order":                "smoothing polynomial order",
        "num_out_ratio":        "ratio between output and input #points in smoothing",
        "mode":                 "smoothing mode (interp or wrap)",
        "min_line_length":      "minimum line length to be smoothed",
    }

    impl_validators = {
        **OptsBase.impl_validators,
        "window_ratio":         lambda v, d: None if v is None else as_number(v, name=d, value_range=(1e-12, np.inf)),
        "window_length":        lambda v, d: None if v is None else as_number(v, name=d, is_integer=True),
        "order":                lambda v, d: as_number(v, name=d, is_integer=True, value_range=(3, np.inf)),
        "num_out_ratio":        lambda v, d: as_number(v, name=d, value_range=(1e-12, np.inf)),
        "mode":                 lambda v, d: as_str(v, name=d, pool=("interp", "wrap")),
        "min_line_length":      lambda v, d: as_number(v, name=d, is_integer=True, value_range=(2, np.inf)),
    }

    impl_defaults_frozen = MappingProxyType({
        **OptsBase.impl_defaults_frozen,
        "tag":                  "smooth options",
        "window_ratio":         None,
        "window_length":        None,
        "order":                3,
        "num_out_ratio":        1,
        "mode":                 "interp",
        "min_line_length":      50,
    })
# fmt: on


class LineSmoothingConfigError(ValueError):
    """Recoverable, user-fixable line-smoothing configuration error."""


class SmoothedLine(HostBase):
    """Smooth and parameterize a polyline while retaining its raw coordinates."""

    # fmt: off
    __attr_defs__ = {
        "raw_coords": AttrDef(
            doc="Raw input line coordinates (shape: N x D)",
            kind="raw",
            validator=lambda v, d: as_points(v, name=d, d=None),
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
            doc="Status indicator of the smoothing pipeline.",
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

    def __init__(
        self,
        coords: np.ndarray,
        name: str | None = None,
        opts: OptsSmoothedLine | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        is_window_warning: bool = True,
        **kwargs,
    ):
        super().__init__(
            OptsSmoothedLine,
            opts,
            opts_defaults_override,
            name=name,
            name_replace="line",
            raw_coords=coords,
            state_is_window_warning=is_window_warning,
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
        object.__setattr__(self, "impl_linefunc_count", 0)

    def _helper_resolve_coords(self):
        object.__setattr__(self, "calc_coords", self.raw_coords)

    def _helper_set_result(self, result) -> None:
        result_readonly = as_readonly_array(result, dtype=None, copy=False)
        object.__setattr__(self, "calc_result", result_readonly)

    @property
    def calc_num_init(self):
        coords = getattr(self, "calc_coords", None)
        if coords is None:
            coords = getattr(self, "raw_coords", None)
        return 0 if coords is None else len(coords)

    def _helper_fallback_no_smooth(self, reason: str) -> None:
        object.__setattr__(self, "calc_is_smoothed", False)
        self._helper_set_result(self.calc_coords)
        object.__setattr__(self, "entity_tck", None)
        object.__setattr__(
            self,
            "calc_status",
            f"The line `{self.name}` is not smoothed, reason: {reason}.",
        )

    def _helper_resolve_window_opts(self, *, logger=None) -> None:
        window_length = self.opts.window_length
        window_ratio = self.opts.window_ratio

        if window_length is None:
            if window_ratio is None:
                raise LineSmoothingConfigError(
                    "No input value provided for smooth window length."
                )
            if self.calc_num_init <= 0:
                raise LineSmoothingConfigError("Cannot smooth an empty line.")
            window_length = int(self.calc_num_init / window_ratio / 2) * 2 + 1
        else:
            if (
                window_ratio is not None
                and self.state_is_window_warning
                and logger is not None
            ):
                logger.warning(
                    f"Window_length is manual input as {window_length}. "
                    f"window_ratio ({window_ratio}) would be ignored and reset."
                )
            window_length = int(window_length)
            if window_length % 2 == 0:
                window_length += 1

        resolved_ratio = self.calc_num_init / window_length
        with self.opts.act_internal_update():
            self.opts.window_length = window_length
            self.opts.window_ratio = resolved_ratio

    def _helper_sample_spline_result(self, tck) -> np.ndarray:
        is_periodic = self.opts.mode == "wrap"
        u_out = np.linspace(
            0.0,
            1.0,
            self.calc_num_out,
            endpoint=not is_periodic,
        )
        knots, coefficients, degree = tck
        result = np.empty((len(u_out), len(coefficients)), dtype=float)
        for axis, coefficient in enumerate(coefficients):
            result[:, axis] = splev(u_out, (knots, coefficient, degree))
        return result

    def _helper_resolve_spline_u(self, u_percent) -> float:
        if getattr(self, "entity_tck", None) is None:
            raise RuntimeError(
                "Spline cache `entity_tck` is missing. "
                "Probably the line is not properly initialized or successfully smoothed."
            )
        u_percent = as_number(
            u_percent,
            value_range=(0, 100),
            name="Continuous spline parameter along the curve",
        )
        u = float(u_percent) / 100.0
        if self.opts.mode == "wrap":
            u = float(np.mod(u, 1.0))
        return u

    @logging_and_warning_decorator()
    def _helper_commit_apply_opts_main(
        self, is_reapply_opts=False, logger=None, **kwargs
    ):
        if not is_reapply_opts and not kwargs:
            return

        with self.opts.act_internal_update():
            if kwargs:
                if "window_ratio" in kwargs and "window_length" not in kwargs:
                    self.opts.window_length = None
                if "window_ratio" not in kwargs and "window_length" in kwargs:
                    self.opts.window_ratio = None
            cover_value(
                self.opts,
                is_allow_cover_target_set=True,
                is_allow_unset_source=False,
                **kwargs,
            )

        self._helper_resolve_coords()
        is_resample_only = (
            not is_reapply_opts
            and set(kwargs) == {"num_out_ratio"}
            and getattr(self, "entity_tck", None) is not None
            and getattr(self, "calc_is_smoothed", False)
        )

        msg = f"Start to smooth line {self.name!r} with {self.calc_num_init} points.\n"
        msg += f"window length = {self.opts.window_length}\n"
        msg += f"window ratio = {self.opts.window_ratio}\n"
        msg += f"minimum smoothed line length = {self.opts.min_line_length}"
        logger.debug(msg)

        try:
            if is_resample_only:
                logger.debug("Reusing cached spline for output-only resampling.")
                self._helper_set_result(
                    self._helper_sample_spline_result(self.entity_tck)
                )
                object.__setattr__(self, "calc_status", "Success")
                return

            self._helper_resolve_window_opts(logger=logger)
            if self.calc_num_init < self.opts.min_line_length:
                raise LineSmoothingConfigError(
                    f"the minimum length of line smoothing is set to be {self.opts.min_line_length} "
                    f"points, while the current line has {self.calc_num_init} points"
                )
            if self.opts.window_length >= self.calc_num_init:
                raise LineSmoothingConfigError(
                    f"Filter window length {self.opts.window_length} should not be larger than "
                    f"line length {self.calc_num_init}"
                )
            if self.opts.window_length <= self.opts.order:
                raise LineSmoothingConfigError(
                    f"Filter window length {self.opts.window_length} should not be smaller than "
                    f"filter order {self.opts.order}"
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
                line_points_spline = np.concatenate((line_points, [line_points[0]]))
                u_spline = np.linspace(0.0, 1.0, len(line_points_spline))
            else:
                line_points_spline = line_points
                u_spline = np.linspace(0.0, 1.0, self.calc_num_init)

            tck = splprep(
                line_points_spline.T,
                u=u_spline,
                s=0,
                per=int(is_periodic),
            )[0]
            del line_points_spline
            del line_points
            del u_spline

            object.__setattr__(self, "entity_tck", tck)
            self._helper_set_result(self._helper_sample_spline_result(tck))
            object.__setattr__(self, "calc_is_smoothed", True)
            object.__setattr__(self, "calc_status", "Success")

        except LineSmoothingConfigError as exc:
            logger.exception("Smoothing aborted (manual check)")
            logger.recovery(
                "Fallback applied: smoothing disabled; using raw coordinates."
            )
            self._helper_fallback_no_smooth(str(exc))
        except (TypeError, ValueError, RuntimeError):
            logger.exception("Smoothing aborted (system error)")
            logger.recovery(
                "Fallback applied: smoothing disabled; using raw coordinates."
            )
            self._helper_fallback_no_smooth("system error")

    def act_calc_tangent(self, u_percent, is_return_coord=False):
        u = self._helper_resolve_spline_u(u_percent)
        is_return_coord = as_bool(
            is_return_coord,
            name="Whether to return the spline coordinate",
        )
        dr_du = np.asarray(splev(u, self.entity_tck, der=1), dtype=float)
        length = float(np.linalg.norm(dr_du))
        if (not np.isfinite(length)) or length < 1e-9:
            raise ValueError(
                f"Degenerate spline derivative at {u}: ||dr/du||={length}."
            )
        tangent = dr_du / length
        if not is_return_coord:
            return tangent
        coord = np.asarray(splev(u, self.entity_tck, der=0), dtype=float)
        return tangent, coord

    def act_calc_pos(self, u_percent):
        u = self._helper_resolve_spline_u(u_percent)
        return np.asarray(splev(u, self.entity_tck, der=0), dtype=float)

    def act_create_linefunc(
        self,
        func,
        u_samples,
        func_kwargs: Mapping[str, Any] | None = None,
        result_value_attr: str = "value",
        is_follow_owner_opts: bool = True,
        name: str | None = None,
    ):
        from .line_function import SmoothedLineFunc

        if name is None:
            name = f"line_func_{self.impl_linefunc_count}"
        linefunc = SmoothedLineFunc(
            func=func,
            u_samples=u_samples,
            owner=self,
            func_kwargs=func_kwargs,
            result_value_attr=result_value_attr,
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

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.calc_result, dtype=dtype, copy=copy)

    def __getitem__(self, idx):
        return self.calc_result[idx]

    def __iter__(self):
        return iter(self.calc_result)

    def __len__(self) -> int:
        result = getattr(self, "calc_result", None)
        return 0 if result is None else len(result)

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


__all__ = ["LineSmoothingConfigError", "OptsSmoothedLine", "SmoothedLine"]
