import numpy as np
import nematics3d.classes.smoothed_line as smoothed_line_module
from nematics3d.classes.smoothed_line import SmoothedLine
from scipy.interpolate import splev


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


def test_smoothed_line_preallocated_sampling_matches_scipy_vector_output():
    """The lower-memory sampling helper must preserve the previous spline values."""
    _, noisy = _build_noisy_line()
    for mode in ("interp", "wrap"):
        line = SmoothedLine(
            noisy,
            window_length=9,
            order=3,
            num_out_ratio=2,
            min_line_length=2,
            mode=mode,
        )
        u_out = np.linspace(
            0.0,
            1.0,
            line.calc_num_out,
            endpoint=mode != "wrap",
        )
        expected = np.array(splev(u_out, line.entity_tck)).T
        np.testing.assert_allclose(line.result, expected, rtol=0.0, atol=0.0)


def test_smoothed_line_num_out_ratio_reuses_cached_spline(monkeypatch):
    """Changing only output density must not rerun filtering or spline fitting."""
    _, noisy = _build_noisy_line()
    line = SmoothedLine(
        noisy,
        window_length=9,
        order=3,
        num_out_ratio=1,
        min_line_length=2,
        mode="interp",
    )
    tck_before = line.entity_tck

    original_savgol_filter = smoothed_line_module.savgol_filter
    original_splprep = smoothed_line_module.splprep

    def _unexpected_recompute(*args, **kwargs):
        raise AssertionError("output-only resampling unexpectedly rebuilt the spline")

    monkeypatch.setattr(smoothed_line_module, "savgol_filter", _unexpected_recompute)
    monkeypatch.setattr(smoothed_line_module, "splprep", _unexpected_recompute)

    line.act_commit(num_out_ratio=2)

    assert line.entity_tck is tck_before
    assert line.calc_is_smoothed is True
    assert line.calc_status == "Success"
    assert line.result.shape == (2 * len(noisy), noisy.shape[1])

    monkeypatch.setattr(smoothed_line_module, "savgol_filter", original_savgol_filter)
    monkeypatch.setattr(smoothed_line_module, "splprep", original_splprep)
    fresh = SmoothedLine(
        noisy,
        window_length=9,
        order=3,
        num_out_ratio=2,
        min_line_length=2,
        mode="interp",
    )
    np.testing.assert_allclose(line.result, fresh.result, rtol=0.0, atol=0.0)


def test_smoothed_line_numpy_array_protocol():
    """Support NumPy 2.x dtype/copy requests through ``__array__``."""
    _, noisy = _build_noisy_line()
    line = SmoothedLine(
        noisy,
        window_length=9,
        order=3,
        num_out_ratio=1,
        min_line_length=2,
        mode="interp",
    )

    no_copy = np.asarray(line, copy=False)
    assert np.shares_memory(no_copy, line.calc_result)

    copied = np.asarray(line, copy=True)
    assert not np.shares_memory(copied, line.calc_result)
    np.testing.assert_array_equal(copied, line.calc_result)

    converted = np.asarray(line, dtype=np.float32)
    assert converted.dtype == np.float32
    np.testing.assert_allclose(converted, line.calc_result, rtol=1e-6, atol=1e-6)

    with np.testing.assert_raises(ValueError):
        np.asarray(line, dtype=np.float32, copy=False)


def test_smoothed_line_result_is_readonly_without_extra_data_copy():
    """Canonical successful output should be a read-only zero-copy view."""
    _, noisy = _build_noisy_line()
    line = SmoothedLine(
        noisy,
        window_length=9,
        order=3,
        num_out_ratio=1,
        min_line_length=2,
        mode="interp",
    )

    assert line.calc_result.flags.writeable is False
    assert line.result.flags.writeable is False
    assert np.asarray(line, copy=False).flags.writeable is False

    with np.testing.assert_raises(ValueError):
        line.result[0, 0] = 0.0
    with np.testing.assert_raises(ValueError):
        np.asarray(line, copy=False)[0, 0] = 0.0
    with np.testing.assert_raises(ValueError):
        line[0][0] = 0.0

    copied = np.asarray(line, copy=True)
    assert copied.flags.writeable is True
    copied[0, 0] = copied[0, 0] + 1.0


def test_smoothed_line_fallback_result_is_readonly_but_raw_coords_stay_writable():
    """Fallback protection must not mark the canonical raw-coordinate array read-only."""
    _, noisy = _build_noisy_line()
    line = SmoothedLine(
        noisy[:8],
        window_length=5,
        order=3,
        min_line_length=50,
        mode="interp",
    )

    assert line.calc_is_smoothed is False
    assert line.calc_result.flags.writeable is False
    assert line.raw_coords.flags.writeable is True
    assert line.calc_coords.flags.writeable is True
    assert np.shares_memory(line.calc_result, line.calc_coords)

    original = float(line.raw_coords[0, 0])
    line.raw_coords[0, 0] = original + 1.0
    assert line.calc_coords[0, 0] == original + 1.0
    assert line.calc_result[0, 0] == original + 1.0

    with np.testing.assert_raises(ValueError):
        line.calc_result[0, 0] = original


def test_smoothed_line_resample_fast_path_keeps_result_readonly(monkeypatch):
    """Output-only resampling should preserve the canonical read-only contract."""
    _, noisy = _build_noisy_line()
    line = SmoothedLine(
        noisy,
        window_length=9,
        order=3,
        num_out_ratio=1,
        min_line_length=2,
        mode="interp",
    )

    def _unexpected_recompute(*args, **kwargs):
        raise AssertionError("output-only resampling unexpectedly rebuilt the spline")

    monkeypatch.setattr(smoothed_line_module, "savgol_filter", _unexpected_recompute)
    monkeypatch.setattr(smoothed_line_module, "splprep", _unexpected_recompute)
    line.act_commit(num_out_ratio=2)

    assert line.calc_result.flags.writeable is False
    with np.testing.assert_raises(ValueError):
        line.result[0, 0] = 0.0


def test_smoothed_line_raw_coords_commit_reapplies_smoothing():
    """Changing raw coordinates must rebuild all dependent smoothing state."""
    _, noisy = _build_noisy_line(num_points=121, seed=7)
    _, replacement = _build_noisy_line(num_points=81, seed=19)
    line = SmoothedLine(
        noisy,
        window_length=9,
        order=3,
        num_out_ratio=1,
        min_line_length=2,
        mode="interp",
    )
    tck_before = line.entity_tck

    line.act_commit(coords=replacement)

    np.testing.assert_array_equal(line.raw_coords, replacement)
    np.testing.assert_array_equal(line.calc_coords, replacement)
    assert line.calc_num_init == len(replacement)
    assert line.calc_num_out == len(replacement)
    assert line.entity_tck is not None
    assert line.entity_tck is not tck_before
    assert line.calc_is_smoothed is True
    assert line.calc_status == "Success"
    assert line.result.shape == replacement.shape
    assert line.result.flags.writeable is False


def test_smoothed_line_can_recover_from_fallback_after_valid_commit():
    """A recoverable fallback must return to a complete success state."""
    _, noisy = _build_noisy_line(num_points=31)
    line = SmoothedLine(
        noisy,
        window_length=5,
        order=3,
        min_line_length=50,
        mode="interp",
    )
    assert line.calc_is_smoothed is False
    assert line.entity_tck is None

    line.act_commit(min_line_length=2)

    assert line.calc_is_smoothed is True
    assert line.calc_status == "Success"
    assert line.entity_tck is not None
    assert line.result.shape == noisy.shape
    assert line.result.flags.writeable is False


def test_smoothed_line_tiny_output_ratio_produces_one_sample():
    """Output density is clamped to at least one sampled spline point."""
    _, noisy = _build_noisy_line()
    line = SmoothedLine(
        noisy,
        window_length=9,
        order=3,
        num_out_ratio=1e-6,
        min_line_length=2,
        mode="interp",
    )

    assert line.calc_num_out == 1
    assert line.result.shape == (1, noisy.shape[1])
    assert np.all(np.isfinite(line.result))
