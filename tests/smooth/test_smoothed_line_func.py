import sys
from pathlib import Path
import types
from dataclasses import dataclass

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
PKG_DIR = SRC_DIR / "nematics3d"

sys.path.insert(0, str(SRC_DIR))

if "nematics3d" not in sys.modules:
    pkg = types.ModuleType("nematics3d")
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules["nematics3d"] = pkg

from nematics3d.core.result_base import ResultBase
from nematics3d.classes.smoothed_line import (
    SmoothedLine,
    linefunc_kernel_weights,
    linefunc_smooth_values,
    linefunc_spacing_weights,
    linefunc_window_span_percent,
)


def _reference_full_matrix_smooth(
    u_samples,
    values,
    *,
    window_ratio,
    order,
    mode,
    kernel,
    min_weight=1e-12,
):
    """Reference the pre-optimization full N x N delta-matrix algorithm."""
    u_samples = np.asarray(u_samples, dtype=float)
    values = np.asarray(values)
    spacing_weights = linefunc_spacing_weights(u_samples, mode=mode)
    window_span_percent = linefunc_window_span_percent(window_ratio=window_ratio)

    values_flat = values.reshape(len(u_samples), -1)
    output = np.empty_like(values_flat, dtype=float)

    deltas_all = u_samples[np.newaxis, :] - u_samples[:, np.newaxis]
    if mode == "wrap":
        deltas_all = (deltas_all + 50.0) % 100.0 - 50.0

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


def test_linefunc_streamed_delta_matches_full_matrix_interp():
    u_samples = np.array([0.0, 4.0, 11.0, 19.0, 31.0, 48.0, 70.0, 100.0])
    values = np.column_stack(
        (
            np.sin(u_samples / 13.0),
            np.cos(u_samples / 17.0),
        )
    )

    expected = _reference_full_matrix_smooth(
        u_samples,
        values,
        window_ratio=3.0,
        order=2,
        mode="interp",
        kernel="tricube",
    )
    actual = linefunc_smooth_values(
        u_samples,
        values,
        window_ratio=3.0,
        order=2,
        mode="interp",
        kernel="tricube",
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_linefunc_streamed_delta_matches_full_matrix_wrap():
    u_samples = np.array([3.0, 12.0, 28.0, 47.0, 66.0, 82.0, 94.0])
    values = np.sin(2.0 * np.pi * u_samples / 100.0)

    expected = _reference_full_matrix_smooth(
        u_samples,
        values,
        window_ratio=4.0,
        order=2,
        mode="wrap",
        kernel="boxcar",
    )
    actual = linefunc_smooth_values(
        u_samples,
        values,
        window_ratio=4.0,
        order=2,
        mode="wrap",
        kernel="boxcar",
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


@dataclass(repr=False)
class ValueResult(ResultBase):
    value: float
    diagnostic: float


@dataclass(repr=False)
class AngleResult(ResultBase):
    angle: float
    diagnostic: float


def _build_protocol_line():
    x = np.linspace(0.0, 1.0, 60)
    coords = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
    return SmoothedLine(coords, window_length=5, min_line_length=2)


def test_linefunc_stores_full_resultbase_samples_and_uses_default_value_attr():
    line = _build_protocol_line()
    linefunc = line.act_create_linefunc(
        lambda u: ValueResult(value=u, diagnostic=u + 1.0),
        [0.0, 25.0, 50.0, 75.0, 100.0],
    )

    assert linefunc.raw_result_value_attr == "value"
    assert isinstance(linefunc.calc_results, tuple)
    assert len(linefunc.calc_results) == 5
    assert all(isinstance(result, ValueResult) for result in linefunc.calc_results)
    assert linefunc.calc_results[2].value == 50.0
    assert linefunc.calc_results[2].diagnostic == 51.0
    assert linefunc.calc_values.shape == (5,)


def test_linefunc_can_select_custom_result_value_attr():
    line = _build_protocol_line()
    linefunc = line.act_create_linefunc(
        lambda u: AngleResult(angle=2.0 * u, diagnostic=-u),
        [0.0, 25.0, 50.0, 75.0, 100.0],
        result_value_attr="angle",
    )

    assert linefunc.raw_result_value_attr == "angle"
    assert linefunc.calc_results[2].angle == 100.0
    assert linefunc.calc_results[2].diagnostic == -50.0
    assert linefunc.calc_values.shape == (5,)


def test_linefunc_rejects_non_resultbase_sample_return():
    line = _build_protocol_line()
    with pytest.raises(TypeError, match="must return a ResultBase instance"):
        line.act_create_linefunc(lambda u: u, [0.0, 50.0, 100.0])


def test_linefunc_rejects_missing_configured_result_attribute():
    line = _build_protocol_line()
    with pytest.raises(AttributeError, match="has no attribute 'value'"):
        line.act_create_linefunc(
            lambda u: AngleResult(angle=u, diagnostic=0.0),
            [0.0, 50.0, 100.0],
        )
