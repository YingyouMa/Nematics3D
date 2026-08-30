import sys
from pathlib import Path
import types

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
PKG_DIR = SRC_DIR / "nematics3d"

sys.path.insert(0, str(SRC_DIR))

# Keep this regression focused on SmoothedLine itself while bypassing the
# top-level nematics3d package import, matching the existing smooth tests.
if "nematics3d" not in sys.modules:
    pkg = types.ModuleType("nematics3d")
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules["nematics3d"] = pkg

from nematics3d.classes.smoothed_line import SmoothedLine


def _build_noisy_line(num_points=121, noise_scale=0.08, seed=7):
    """Return a deterministic noisy 3D curve with a known smooth backbone."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 4.0 * np.pi, num_points)
    clean = np.column_stack(
        (
            t,
            1.4 * np.sin(t) + 0.25 * np.sin(3.0 * t),
            0.6 * np.cos(0.5 * t) + 0.15 * np.sin(2.0 * t),
        )
    )
    noisy = clean + noise_scale * rng.normal(size=clean.shape)
    return clean, noisy


def test_smoothed_line_current_smoothing_regression():
    """Freeze the current successful SmoothedLine smoothing behavior.

    This is intentionally a behavior-level regression test rather than a
    reimplementation of the Savitzky-Golay/spline algorithm. It establishes a
    stable baseline before SmoothedLine/HostBase cleanup work: construction must
    smooth successfully, preserve the raw input, produce the requested output
    size, reduce deterministic input noise, and expose a usable spline cache.
    """
    clean, noisy = _build_noisy_line()
    noisy_before = noisy.copy()

    line = SmoothedLine(
        noisy,
        window_length=9,
        order=3,
        num_out_ratio=1,
        min_line_length=2,
        mode="interp",
    )

    assert line.calc_is_smoothed is True
    assert line.calc_status == "Success"
    assert line.entity_tck is not None

    np.testing.assert_array_equal(noisy, noisy_before)
    np.testing.assert_array_equal(line.raw_coords, noisy_before)
    assert line.calc_num_init == len(noisy_before)
    assert line.calc_num_out == len(noisy_before)
    assert line.result.shape == noisy_before.shape
    assert np.all(np.isfinite(line.result))

    raw_rmse = np.sqrt(np.mean((noisy_before - clean) ** 2))
    smooth_rmse = np.sqrt(np.mean((line.result - clean) ** 2))
    assert smooth_rmse < raw_rmse

    tangent, coord = line.act_calc_tangent(50, is_return_coord=True)
    np.testing.assert_allclose(np.linalg.norm(tangent), 1.0, atol=1e-12)
    np.testing.assert_allclose(coord, line.act_calc_pos(50), atol=1e-12)
    assert np.all(np.isfinite(tangent))
    assert np.all(np.isfinite(coord))


def test_smoothed_line_window_resolution_and_fallback_contract():
    """Cover window normalization plus representative recoverable fallbacks."""
    _, noisy = _build_noisy_line()

    ratio_line = SmoothedLine(
        noisy,
        window_ratio=15,
        order=3,
        num_out_ratio=1,
        min_line_length=2,
        mode="interp",
    )
    assert ratio_line.calc_is_smoothed is True
    assert ratio_line.opts.window_length == 9
    np.testing.assert_allclose(
        ratio_line.opts.window_ratio,
        ratio_line.calc_num_init / ratio_line.opts.window_length,
    )

    even_line = SmoothedLine(
        noisy,
        window_length=8,
        order=3,
        num_out_ratio=1,
        min_line_length=2,
        mode="interp",
    )
    assert even_line.calc_is_smoothed is True
    assert even_line.opts.window_length == 9

    short_line = SmoothedLine(
        noisy[:8],
        window_length=5,
        order=3,
        min_line_length=50,
        mode="interp",
    )
    assert short_line.calc_is_smoothed is False
    assert short_line.entity_tck is None
    np.testing.assert_array_equal(short_line.result, short_line.calc_coords)
    assert "minimum length" in short_line.calc_status.lower()


def test_smoothed_line_query_helpers_and_wrap_boundary():
    """Keep position/tangent parameter handling shared and periodic at 100%."""
    _, noisy = _build_noisy_line()
    line = SmoothedLine(
        noisy,
        window_length=9,
        order=3,
        num_out_ratio=1,
        min_line_length=2,
        mode="wrap",
    )

    np.testing.assert_allclose(line.act_calc_pos(100), line.act_calc_pos(0), atol=1e-12)
    tangent, coord = line.act_calc_tangent(100, is_return_coord=True)
    np.testing.assert_allclose(coord, line.act_calc_pos(0), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(tangent), 1.0, atol=1e-12)


def test_smoothed_line_zero_window_ratio_does_not_reach_division():
    """A non-positive ratio must be rejected before smoothing window arithmetic."""
    _, noisy = _build_noisy_line()
    line = SmoothedLine(
        noisy,
        window_ratio=0,
        order=3,
        min_line_length=2,
        mode="interp",
    )
    assert line.calc_is_smoothed is False
    assert line.entity_tck is None
    assert "no input value" in line.calc_status.lower()
