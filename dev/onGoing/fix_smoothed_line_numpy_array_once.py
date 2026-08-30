from pathlib import Path

source_path = Path("src/nematics3d/classes/smoothed_line.py")
test_path = Path("tests/smooth/test_smoothed_line.py")

source = source_path.read_text()
old = '''    def __array__(self, dtype=None):
        arr = self.calc_result
        return np.asarray(arr, dtype=dtype) if dtype is not None else arr
'''
new = '''    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.calc_result, dtype=dtype, copy=copy)
'''
if old not in source:
    raise RuntimeError("Expected SmoothedLine __array__ implementation not found")
source_path.write_text(source.replace(old, new, 1))

tests = test_path.read_text()
append = r'''


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
'''
if "def test_smoothed_line_numpy_array_protocol():" in tests:
    raise RuntimeError("NumPy array protocol test already exists")
test_path.write_text(tests + append)
