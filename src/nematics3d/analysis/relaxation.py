"""Relaxation-length estimators for one-dimensional correlation curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.optimize import curve_fit

from nematics3d.classes.result_base import ResultBase
from nematics3d.datatypes import as_Number


@dataclass(slots=True, frozen=True, repr=False)
class ThresholdRelaxationResult(ResultBase):
    """Relaxation length measured by threshold crossing."""

    __result_name__: ClassVar[str] = "Threshold relaxation length"
    __field_docs__: ClassVar[dict[str, str]] = {
        "length": (
            "Interpolated coordinate where the correlation first crosses the "
            "threshold value; None when no crossing is found."
        ),
        "threshold": "Threshold value applied to the working correlation curve.",
        "index": (
            "Index of the first sampled point at or below the threshold; None "
            "when no crossing is found."
        ),
        "is_crossed": "Whether the correlation curve crosses the threshold.",
        "message": "Short status message describing the threshold measurement.",
    }

    length: float | None
    threshold: float
    index: int | None
    is_crossed: bool
    message: str


@dataclass(slots=True, frozen=True, repr=False)
class FitRelaxationResult(ResultBase):
    """Relaxation length measured by fitting a decay model."""

    __result_name__: ClassVar[str] = "Fitted relaxation length"
    __field_docs__: ClassVar[dict[str, str]] = {
        "model_name": "Name of the fitted decay model.",
        "length": "Relaxation length from the fitted model; None when unavailable.",
        "rmse": "Root-mean-square fitting error; None when fitting is unavailable.",
        "fit_head_factor": (
            "Optional lower fitting bound as a multiple of the current relaxation "
            "length."
        ),
        "fit_tail_factor": (
            "Optional upper fitting bound as a multiple of the current relaxation "
            "length."
        ),
        "fit_x_min": "Minimum coordinate included in the final fit range.",
        "fit_x_max": "Maximum coordinate included in the final fit range.",
        "iteration_num": "Number of fitting-range refinement iterations performed.",
        "is_converged": "Whether the iterative fitting procedure converged.",
        "message": "Short status message describing the fit result.",
    }

    model_name: str
    length: float | None
    rmse: float | None
    fit_head_factor: float | None
    fit_tail_factor: float | None
    fit_x_min: float | None
    fit_x_max: float | None
    iteration_num: int
    is_converged: bool
    message: str


@dataclass(slots=True, frozen=True, repr=False)
class RelaxationLengthResult(ResultBase):
    """Container returned by :func:`act_relaxation_length`."""

    __result_name__: ClassVar[str] = "Correlation relaxation length"
    __field_docs__: ClassVar[dict[str, str]] = {
        "threshold": "Relaxation-length result from direct threshold crossing.",
        "exponential": "Relaxation-length result from exponential-model fitting.",
        "gaussian": "Relaxation-length result from Gaussian-model fitting.",
    }

    threshold: ThresholdRelaxationResult
    exponential: FitRelaxationResult
    gaussian: FitRelaxationResult


def _as_correlation_curve(
    correlation,
    coordinate_axis=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize one-dimensional correlation curve inputs."""
    correlation = np.asarray(correlation, dtype=float)
    if coordinate_axis is None:
        coordinate_axis = np.arange(correlation.size, dtype=float)
    else:
        coordinate_axis = np.asarray(coordinate_axis, dtype=float)

    if correlation.ndim != 1:
        raise ValueError("`correlation` must be a one-dimensional array.")
    if coordinate_axis.ndim != 1:
        raise ValueError("`coordinate_axis` must be a one-dimensional array.")
    if coordinate_axis.shape != correlation.shape:
        raise ValueError(
            "`coordinate_axis` and `correlation` must have the same shape."
        )
    if correlation.size < 2:
        raise ValueError("`correlation` must contain at least two points.")
    if not np.all(np.isfinite(coordinate_axis)):
        raise ValueError("`coordinate_axis` must contain only finite values.")
    if not np.all(np.isfinite(correlation)):
        raise ValueError("`correlation` must contain only finite values.")
    if not np.all(np.diff(coordinate_axis) > 0):
        raise ValueError("`coordinate_axis` must be strictly increasing.")
    if correlation[0] == 0:
        raise ValueError("`correlation[0]` must be non-zero.")
    correlation = correlation / correlation[0]

    return coordinate_axis, correlation


def _threshold_result(
    x: np.ndarray,
    correlation: np.ndarray,
    *,
    threshold: float,
) -> ThresholdRelaxationResult:
    """Return the first interpolated location where correlation crosses threshold."""
    if correlation[0] <= threshold:
        return ThresholdRelaxationResult(
            length=float(x[0]),
            threshold=threshold,
            index=0,
            is_crossed=True,
            message="The first correlation value is already below the threshold.",
        )

    below_threshold = correlation <= threshold
    crossing_indices = np.flatnonzero(below_threshold)
    crossing_indices = crossing_indices[crossing_indices > 0]

    if crossing_indices.size == 0:
        return ThresholdRelaxationResult(
            length=None,
            threshold=threshold,
            index=None,
            is_crossed=False,
            message="The correlation curve does not cross the threshold.",
        )

    index = int(crossing_indices[0])
    x_left = float(x[index - 1])
    x_right = float(x[index])
    y_left = float(correlation[index - 1])
    y_right = float(correlation[index])

    if y_right == y_left:
        length = x_right
    else:
        fraction = (threshold - y_left) / (y_right - y_left)
        length = x_left + fraction * (x_right - x_left)

    return ThresholdRelaxationResult(
        length=float(length),
        threshold=threshold,
        index=index,
        is_crossed=True,
        message="The correlation curve crosses the threshold.",
    )


def _empty_fit_result(model_name: str) -> FitRelaxationResult:
    """Return a placeholder fit result before model fitting is implemented."""
    return FitRelaxationResult(
        model_name=model_name,
        length=None,
        rmse=None,
        fit_head_factor=None,
        fit_tail_factor=None,
        fit_x_min=None,
        fit_x_max=None,
        iteration_num=0,
        is_converged=False,
        message="Fitting is not implemented yet.",
    )


def _exponential_model(
    x: np.ndarray,
    length: float,
) -> np.ndarray:
    """Return exponential decay values for curve fitting."""
    return np.exp(-x / length)


def _gaussian_model(
    x: np.ndarray,
    length: float,
) -> np.ndarray:
    """Return Gaussian decay values for curve fitting."""
    return np.exp(-((x / length) ** 2))


def _as_optional_nonnegative_number(value, *, name: str) -> float | None:
    """Return a non-negative float or None."""
    if value is None:
        return None
    return float(
        as_Number(
            value,
            name=name,
            value_range=(0.0, np.inf),
        )
    )


def _as_positive_integer(value, *, name: str) -> int:
    """Return a strictly positive integer."""
    value = int(
        as_Number(
            value,
            name=name,
            is_int=True,
            value_range=(0, np.inf),
        )
    )
    if value <= 0:
        raise ValueError(f"`{name}` must be positive.")
    return value


def _initial_fit_length(
    x: np.ndarray,
    threshold_result: ThresholdRelaxationResult,
) -> float:
    """Return the first fitting length estimate."""
    if threshold_result.length is not None and threshold_result.length > 0:
        return float(threshold_result.length)

    span = float(x[-1] - x[0])
    if span > 0:
        return span / 5.0
    return 1.0


def _fit_range_mask(
    x: np.ndarray,
    *,
    length: float,
    fit_head_factor: float | None,
    fit_tail_factor: float | None,
) -> np.ndarray:
    """Return the fitting-range mask implied by length multiples."""
    mask = np.ones_like(x, dtype=bool)
    if fit_head_factor is not None:
        mask &= x >= fit_head_factor * length
    if fit_tail_factor is not None:
        mask &= x <= fit_tail_factor * length
    return mask


def _fit_range_limits(
    x: np.ndarray, mask: np.ndarray
) -> tuple[float | None, float | None]:
    """Return actual minimum and maximum coordinates used for fitting."""
    if not np.any(mask):
        return None, None
    return float(np.min(x[mask])), float(np.max(x[mask]))


def _fit_rmse(
    x: np.ndarray,
    y: np.ndarray,
    *,
    model_func,
    length: float,
) -> float:
    """Return root-mean-square fitting error."""
    residual = y - model_func(x, length)
    return float(np.sqrt(np.mean(residual**2)))


def _fit_decay_once(
    x: np.ndarray,
    y: np.ndarray,
    *,
    model_func,
    initial_length: float,
) -> float:
    """Fit one single-parameter decay model and return the relaxation length."""
    length_init = float(initial_length)
    tiny = float(np.finfo(float).tiny)

    popt, _ = curve_fit(
        model_func,
        x,
        y,
        p0=(length_init,),
        bounds=((tiny,), (np.inf,)),
        maxfev=10000,
    )

    return float(popt[0])


def _fit_decay_result(
    x: np.ndarray,
    correlation: np.ndarray,
    threshold_result: ThresholdRelaxationResult,
    *,
    model_name: str,
    model_func,
    fit_head_factor: float | None,
    fit_tail_factor: float | None,
    max_iteration_num: int,
    fit_tolerance: float,
    min_fit_point_num: int,
) -> FitRelaxationResult:
    """Return an iterative single-parameter decay-fit relaxation result."""
    length_current = _initial_fit_length(x, threshold_result)
    rmse = None
    fit_x_min = None
    fit_x_max = None
    message = f"The {model_name} fit did not converge."

    for iteration_index in range(max_iteration_num):
        mask = _fit_range_mask(
            x,
            length=length_current,
            fit_head_factor=fit_head_factor,
            fit_tail_factor=fit_tail_factor,
        )
        fit_x_min, fit_x_max = _fit_range_limits(x, mask)
        point_num = int(np.count_nonzero(mask))
        if point_num < min_fit_point_num:
            return FitRelaxationResult(
                model_name=model_name,
                length=None,
                rmse=None,
                fit_head_factor=fit_head_factor,
                fit_tail_factor=fit_tail_factor,
                fit_x_min=fit_x_min,
                fit_x_max=fit_x_max,
                iteration_num=iteration_index,
                is_converged=False,
                message=(
                    f"The {model_name} fit range contains fewer than "
                    f"{min_fit_point_num} point(s)."
                ),
            )

        try:
            length_new = _fit_decay_once(
                x[mask],
                correlation[mask],
                model_func=model_func,
                initial_length=length_current,
            )
        except Exception as exc:
            return FitRelaxationResult(
                model_name=model_name,
                length=None,
                rmse=None,
                fit_head_factor=fit_head_factor,
                fit_tail_factor=fit_tail_factor,
                fit_x_min=fit_x_min,
                fit_x_max=fit_x_max,
                iteration_num=iteration_index,
                is_converged=False,
                message=f"The {model_name} fit failed: {exc}",
            )

        if not np.isfinite(length_new) or length_new <= 0:
            return FitRelaxationResult(
                model_name=model_name,
                length=None,
                rmse=None,
                fit_head_factor=fit_head_factor,
                fit_tail_factor=fit_tail_factor,
                fit_x_min=fit_x_min,
                fit_x_max=fit_x_max,
                iteration_num=iteration_index + 1,
                is_converged=False,
                message=f"The {model_name} fit returned a non-positive length.",
            )

        rmse = _fit_rmse(
            x[mask],
            correlation[mask],
            model_func=model_func,
            length=length_new,
        )
        relative_change = abs(length_new - length_current) / length_current
        length_current = length_new
        if relative_change <= fit_tolerance:
            return FitRelaxationResult(
                model_name=model_name,
                length=length_current,
                rmse=rmse,
                fit_head_factor=fit_head_factor,
                fit_tail_factor=fit_tail_factor,
                fit_x_min=fit_x_min,
                fit_x_max=fit_x_max,
                iteration_num=iteration_index + 1,
                is_converged=True,
                message=f"The {model_name} fit converged.",
            )

    return FitRelaxationResult(
        model_name=model_name,
        length=length_current,
        rmse=rmse,
        fit_head_factor=fit_head_factor,
        fit_tail_factor=fit_tail_factor,
        fit_x_min=fit_x_min,
        fit_x_max=fit_x_max,
        iteration_num=max_iteration_num,
        is_converged=False,
        message=message,
    )


def act_relaxation_length(
    correlation,
    *,
    coordinate_axis=None,
    threshold: float = np.exp(-1),
    fit_head_factor: float | None = None,
    fit_tail_factor: float | None = 10.0,
    max_iteration_num: int = 20,
    fit_tolerance: float = 1e-3,
    min_fit_point_num: int = 4,
) -> RelaxationLengthResult:
    """Estimate relaxation length from a one-dimensional correlation curve.

    This initial implementation measures the threshold-crossing length. It uses
    ``coordinate_axis`` for physical distances when provided; otherwise, grid
    indices are used as the coordinate. The correlation curve is normalized by
    its first value before estimating the length. Exponential and Gaussian
    fitting iterate their fitting ranges as multiples of the current fitted
    length.
    """
    threshold = float(
        as_Number(
            threshold,
            name="threshold",
            value_range=(0.0, np.inf),
        )
    )
    fit_head_factor = _as_optional_nonnegative_number(
        fit_head_factor,
        name="fit_head_factor",
    )
    fit_tail_factor = _as_optional_nonnegative_number(
        fit_tail_factor,
        name="fit_tail_factor",
    )
    max_iteration_num = _as_positive_integer(
        max_iteration_num,
        name="max_iteration_num",
    )
    fit_tolerance = float(
        as_Number(
            fit_tolerance,
            name="fit_tolerance",
            value_range=(0.0, np.inf),
        )
    )
    min_fit_point_num = _as_positive_integer(
        min_fit_point_num,
        name="min_fit_point_num",
    )

    coordinate_axis, correlation = _as_correlation_curve(
        correlation,
        coordinate_axis=coordinate_axis,
    )
    threshold_result = _threshold_result(
        coordinate_axis,
        correlation,
        threshold=threshold,
    )

    return RelaxationLengthResult(
        threshold=threshold_result,
        exponential=_fit_decay_result(
            coordinate_axis,
            correlation,
            threshold_result,
            model_name="exponential",
            model_func=_exponential_model,
            fit_head_factor=fit_head_factor,
            fit_tail_factor=fit_tail_factor,
            max_iteration_num=max_iteration_num,
            fit_tolerance=fit_tolerance,
            min_fit_point_num=min_fit_point_num,
        ),
        gaussian=_fit_decay_result(
            coordinate_axis,
            correlation,
            threshold_result,
            model_name="gaussian",
            model_func=_gaussian_model,
            fit_head_factor=fit_head_factor,
            fit_tail_factor=fit_tail_factor,
            max_iteration_num=max_iteration_num,
            fit_tolerance=fit_tolerance,
            min_fit_point_num=min_fit_point_num,
        ),
    )
