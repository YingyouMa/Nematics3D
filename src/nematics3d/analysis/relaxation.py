"""Relaxation-length estimators for one-dimensional correlation curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

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
        "threshold": "Relative threshold multiplier applied to the first value.",
        "threshold_value": "Absolute correlation value used as the crossing threshold.",
        "index": (
            "Index of the first sampled point at or below the threshold; None "
            "when no crossing is found."
        ),
        "is_crossed": "Whether the correlation curve crosses the threshold.",
        "message": "Short status message describing the threshold measurement.",
    }

    length: float | None
    threshold: float
    threshold_value: float
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
        "amplitude": "Fitted decay amplitude; None when fitting is unavailable.",
        "offset": "Fitted background offset; None when fitting is unavailable.",
        "rmse": "Root-mean-square fitting error; None when fitting is unavailable.",
        "fit_range_factor": (
            "Requested fitting range as a multiple of the current relaxation length."
        ),
        "fit_x_max": "Maximum coordinate included in the final fit range.",
        "iteration_num": "Number of fitting-range refinement iterations performed.",
        "is_converged": "Whether the iterative fitting procedure converged.",
        "message": "Short status message describing the fit result.",
    }

    model_name: str
    length: float | None
    amplitude: float | None
    offset: float | None
    rmse: float | None
    fit_range_factor: float | None
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

    return coordinate_axis, correlation


def _threshold_result(
    x: np.ndarray,
    correlation: np.ndarray,
    *,
    threshold: float,
) -> ThresholdRelaxationResult:
    """Return the first interpolated location where correlation crosses threshold."""
    threshold_value = float(threshold * correlation[0])

    if correlation[0] <= threshold_value:
        return ThresholdRelaxationResult(
            length=float(x[0]),
            threshold=threshold,
            threshold_value=threshold_value,
            index=0,
            is_crossed=True,
            message="The first correlation value is already below the threshold.",
        )

    below_threshold = correlation <= threshold_value
    crossing_indices = np.flatnonzero(below_threshold)
    crossing_indices = crossing_indices[crossing_indices > 0]

    if crossing_indices.size == 0:
        return ThresholdRelaxationResult(
            length=None,
            threshold=threshold,
            threshold_value=threshold_value,
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
        fraction = (threshold_value - y_left) / (y_right - y_left)
        length = x_left + fraction * (x_right - x_left)

    return ThresholdRelaxationResult(
        length=float(length),
        threshold=threshold,
        threshold_value=threshold_value,
        index=index,
        is_crossed=True,
        message="The correlation curve crosses the threshold.",
    )


def _empty_fit_result(model_name: str) -> FitRelaxationResult:
    """Return a placeholder fit result before model fitting is implemented."""
    return FitRelaxationResult(
        model_name=model_name,
        length=None,
        amplitude=None,
        offset=None,
        rmse=None,
        fit_range_factor=None,
        fit_x_max=None,
        iteration_num=0,
        is_converged=False,
        message="Fitting is not implemented yet.",
    )


def act_relaxation_length(
    correlation,
    *,
    coordinate_axis=None,
    threshold: float = np.exp(-1),
) -> RelaxationLengthResult:
    """Estimate relaxation length from a one-dimensional correlation curve.

    This initial implementation measures the threshold-crossing length. It uses
    ``coordinate_axis`` for physical distances when provided; otherwise, grid
    indices are used as the coordinate. Exponential and Gaussian fit results are
    placeholders so the final result object already has its stable three-part
    shape.
    """
    threshold = float(
        as_Number(
            threshold,
            name="threshold",
            value_range=(0.0, np.inf),
        )
    )
    coordinate_axis, correlation = _as_correlation_curve(
        correlation,
        coordinate_axis=coordinate_axis,
    )

    return RelaxationLengthResult(
        threshold=_threshold_result(
            coordinate_axis,
            correlation,
            threshold=threshold,
        ),
        exponential=_empty_fit_result("exponential"),
        gaussian=_empty_fit_result("gaussian"),
    )
