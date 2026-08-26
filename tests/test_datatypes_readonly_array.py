import numpy as np
import pytest

from nematics3d.datatypes import as_readonly_array


def test_readonly_array_default_copy_is_independent_and_readonly():
    source = np.array([1.0, 2.0, 3.0])
    result = as_readonly_array(source)

    assert result.flags.writeable is False
    assert not np.shares_memory(result, source)

    source[0] = 10.0
    assert result[0] == 1.0

    with pytest.raises(ValueError):
        result[0] = 5.0


def test_readonly_array_copy_false_shares_memory_without_freezing_source():
    source = np.array([1.0, 2.0, 3.0])
    result = as_readonly_array(source, copy=False)

    assert result.flags.writeable is False
    assert source.flags.writeable is True
    assert np.shares_memory(result, source)
    assert result is not source

    source[0] = 10.0
    assert result[0] == 10.0

    with pytest.raises(ValueError):
        result[0] = 5.0


def test_readonly_array_default_dtype_is_float():
    result = as_readonly_array([1, 2, 3])
    assert result.dtype == np.dtype(float)


def test_readonly_array_respects_explicit_dtype():
    result = as_readonly_array([1, 2, 3], dtype=np.int64)
    assert result.dtype == np.dtype(np.int64)


def test_readonly_array_copy_must_be_bool():
    with pytest.raises((TypeError, ValueError)):
        as_readonly_array([1, 2, 3], copy=1)
    with pytest.raises((TypeError, ValueError)):
        as_readonly_array([1, 2, 3], copy="False")
