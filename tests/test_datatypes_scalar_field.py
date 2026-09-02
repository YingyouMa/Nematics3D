import numpy as np
import pytest

from nematics3d.datatypes import as_scalar_field


def test_as_scalar_field_preserves_floating_dtype_and_storage():
    values = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    result = as_scalar_field(values)

    assert result is values
    assert result.dtype == np.float32


def test_as_scalar_field_converts_integer_values_to_float():
    result = as_scalar_field(np.array([1, 2, 3], dtype=np.int16))

    assert np.issubdtype(result.dtype, np.floating)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_as_scalar_field_rejects_complex_and_non_real_object_values():
    with pytest.raises(TypeError, match="real numbers"):
        as_scalar_field(np.array([1.0 + 2.0j]))

    with pytest.raises(TypeError, match="real numbers"):
        as_scalar_field(np.array([1.0, "invalid"], dtype=object))
