from pathlib import Path

source_path = Path("src/nematics3d/classes/smoothed_line.py")
test_path = Path("tests/smooth/test_smoothed_line.py")

source = source_path.read_text()

old_import = "from ..datatypes import Number, UNSET, Unset, as_number, as_bool, as_points, as_str\n"
new_import = "from ..datatypes import (\n    Number,\n    UNSET,\n    Unset,\n    as_bool,\n    as_number,\n    as_points,\n    as_readonly_array,\n    as_str,\n)\n"
if old_import not in source:
    raise RuntimeError("Expected datatypes import not found")
source = source.replace(old_import, new_import, 1)

marker = '''    def _helper_resolve_coords(self):\n        object.__setattr__(self, "calc_coords", self.raw_coords)\n\n'''
helper = '''    def _helper_resolve_coords(self):\n        object.__setattr__(self, "calc_coords", self.raw_coords)\n\n    def _helper_set_result(self, result) -> None:\n        """Store the canonical output as a zero-copy read-only array view."""\n        result_readonly = as_readonly_array(result, dtype=None, copy=False)\n        object.__setattr__(self, "calc_result", result_readonly)\n\n'''
if marker not in source:
    raise RuntimeError("Expected resolve-coords block not found")
source = source.replace(marker, helper, 1)

source = source.replace(
    '        object.__setattr__(self, "calc_result", self.calc_coords)\n',
    '        self._helper_set_result(self.calc_coords)\n',
    1,
)
source = source.replace(
    '                object.__setattr__(self, "calc_result", result)\n',
    '                self._helper_set_result(result)\n',
    1,
)
source = source.replace(
    '            object.__setattr__(self, "calc_result", result)\n',
    '            self._helper_set_result(result)\n',
    1,
)

source_path.write_text(source)

tests = test_path.read_text()
append = r'''


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
'''
if "test_smoothed_line_result_is_readonly_without_extra_data_copy" in tests:
    raise RuntimeError("Read-only tests already present")
test_path.write_text(tests + append)
