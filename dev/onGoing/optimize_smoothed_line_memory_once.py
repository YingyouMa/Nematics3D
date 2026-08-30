from pathlib import Path

source_path = Path("src/nematics3d/classes/smoothed_line.py")
test_path = Path("tests/smooth/test_smoothed_line.py")

source = source_path.read_text()

old_init = '''
        object.__setattr__(self, "entity_tck", None)
        object.__setattr__(self, "entity_linefuncs", None)
        object.__setattr__(self, "impl_linefunc_count", 0)
        object.__setattr__(self, "calc_is_smoothed", False)
        object.__setattr__(self, "calc_status", "Failure, reason unknown.")

        super().__init__(
'''
new_init = '''
        super().__init__(
'''
if old_init not in source:
    raise RuntimeError("Expected SmoothedLine bootstrap block not found")
source = source.replace(old_init, new_init, 1)

old_registry = '''        linefuncs.act_bind_relation_base("owner", self, is_weak=True)
        object.__setattr__(self, "entity_linefuncs", linefuncs)
'''
new_registry = '''        linefuncs.act_bind_relation_base("owner", self, is_weak=True)
        object.__setattr__(self, "entity_linefuncs", linefuncs)
        object.__setattr__(self, "impl_linefunc_count", 0)
'''
if old_registry not in source:
    raise RuntimeError("Expected linefunc registry block not found")
source = source.replace(old_registry, new_registry, 1)

marker = '''    def _helper_resolve_spline_u(self, u_percent) -> float:
'''
helper = '''    def _helper_sample_spline_result(self, tck) -> np.ndarray:
        """Sample a cached parametric spline into the configured output array."""
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

'''
if marker not in source:
    raise RuntimeError("Spline helper insertion point not found")
source = source.replace(marker, helper + marker, 1)

old_after_cover = '''        self._helper_resolve_coords()

        msg = f"Start to smooth line {self.name!r} with {self.calc_num_init} points.\\n"
'''
new_after_cover = '''        self._helper_resolve_coords()
        is_resample_only = (
            not is_reapply_opts
            and set(kwargs) == {"num_out_ratio"}
            and getattr(self, "entity_tck", None) is not None
            and getattr(self, "calc_is_smoothed", False)
        )

        msg = f"Start to smooth line {self.name!r} with {self.calc_num_init} points.\\n"
'''
if old_after_cover not in source:
    raise RuntimeError("Expected post-cover block not found")
source = source.replace(old_after_cover, new_after_cover, 1)

old_try = '''        try:
            self._helper_resolve_window_opts(logger=logger)
'''
new_try = '''        try:
            if is_resample_only:
                logger.debug("Reusing cached spline for output-only resampling.")
                result = self._helper_sample_spline_result(self.entity_tck)
                object.__setattr__(self, "calc_result", result)
                object.__setattr__(self, "calc_status", "Success")
                return

            self._helper_resolve_window_opts(logger=logger)
'''
if old_try not in source:
    raise RuntimeError("Expected smoothing try block not found")
source = source.replace(old_try, new_try, 1)

old_periodic = '''            is_periodic = self.opts.mode == "wrap"
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
'''
new_periodic = '''            is_periodic = self.opts.mode == "wrap"
            if is_periodic:
                # FITPACK periodic splines treat the last sample as the seam copy
                # of the first one. Provide that seam explicitly so no genuine
                # endpoint sample is overwritten in-place by `splprep(per=1)`.
                line_points_spline = np.concatenate((line_points, [line_points[0]]))
                uspline = np.linspace(0.0, 1.0, len(line_points_spline))
            else:
                line_points_spline = line_points
                uspline = np.linspace(0.0, 1.0, self.calc_num_init)

            tck = splprep(
                line_points_spline.T.copy(),
                u=uspline,
                s=0,
                per=int(is_periodic),
            )[0]

            # FITPACK has already consumed the filtered input. Drop these large
            # temporaries before allocating the final resampled output.
            del line_points_spline
            del line_points
            del uspline

            result = self._helper_sample_spline_result(tck)
            object.__setattr__(self, "entity_tck", tck)
'''
if old_periodic not in source:
    raise RuntimeError("Expected spline construction block not found")
source = source.replace(old_periodic, new_periodic, 1)

source_path.write_text(source)

tests = test_path.read_text()

old_import = '''from nematics3d.classes.smoothed_line import SmoothedLine
'''
new_import = '''import nematics3d.classes.smoothed_line as smoothed_line_module
from nematics3d.classes.smoothed_line import SmoothedLine
from scipy.interpolate import splev
'''
if old_import not in tests:
    raise RuntimeError("Expected SmoothedLine import not found")
tests = tests.replace(old_import, new_import, 1)

append = r'''


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
'''
if "test_smoothed_line_num_out_ratio_reuses_cached_spline" in tests:
    raise RuntimeError("Optimization tests already exist")
tests += append

test_path.write_text(tests)
