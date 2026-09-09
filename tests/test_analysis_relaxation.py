import numpy as np
import pytest

from nematics3d.analysis.relaxation import act_relaxation_length


def test_relaxation_threshold_crossing_is_linearly_interpolated():
    result = act_relaxation_length(
        [1.0, 0.5, 0.2, 0.1],
        coordinate_axis=[0.0, 2.0, 4.0, 6.0],
        threshold=0.35,
    )

    assert result.threshold.is_crossed
    assert result.threshold.index == 2
    assert result.threshold.length == pytest.approx(3.0)


def test_relaxation_threshold_reports_missing_crossing():
    result = act_relaxation_length(
        [2.0, 1.8, 1.6, 1.4],
        threshold=0.5,
    )

    assert not result.threshold.is_crossed
    assert result.threshold.index is None
    assert result.threshold.length is None


def test_relaxation_recovers_exact_exponential_length():
    x = np.linspace(0.0, 20.0, 101)
    expected_length = 3.5
    correlation = np.exp(-x / expected_length)

    result = act_relaxation_length(correlation, coordinate_axis=x)

    assert result.exponential.is_converged
    assert result.exponential.length == pytest.approx(expected_length, rel=1e-6)
    assert result.exponential.rmse == pytest.approx(0.0, abs=1e-10)


def test_relaxation_recovers_exact_gaussian_length():
    x = np.linspace(0.0, 20.0, 101)
    expected_length = 4.25
    correlation = np.exp(-((x / expected_length) ** 2))

    result = act_relaxation_length(correlation, coordinate_axis=x)

    assert result.gaussian.is_converged
    assert result.gaussian.length == pytest.approx(expected_length, rel=1e-6)
    assert result.gaussian.rmse == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize(
    ("correlation", "coordinate_axis", "message"),
    [
        ([[1.0, 0.5]], None, "one-dimensional"),
        ([1.0], None, "at least two points"),
        ([0.0, 0.5], None, "non-zero"),
        ([1.0, np.nan], None, "finite"),
        ([1.0, 0.5], [0.0, 0.0], "strictly increasing"),
        ([1.0, 0.5], [0.0, 1.0, 2.0], "same shape"),
    ],
)
def test_relaxation_rejects_invalid_curve_inputs(correlation, coordinate_axis, message):
    with pytest.raises(ValueError, match=message):
        act_relaxation_length(correlation, coordinate_axis=coordinate_axis)


def test_relaxation_fit_reports_too_few_points_without_raising():
    x = np.arange(5.0)
    correlation = np.exp(-x / 2.0)

    result = act_relaxation_length(
        correlation,
        coordinate_axis=x,
        fit_head_factor=100.0,
        min_fit_point_num=4,
    )

    assert not result.exponential.is_converged
    assert result.exponential.length is None
    assert "fewer than 4 point" in result.exponential.message
    assert not result.gaussian.is_converged
    assert result.gaussian.length is None
