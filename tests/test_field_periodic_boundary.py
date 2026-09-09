import numpy as np
import pytest

from nematics3d.misc import add_periodic_boundary


def test_add_periodic_boundary_appends_first_slice_on_selected_axes():
    data = np.arange(2 * 3 * 4 * 2).reshape(2, 3, 4, 2)

    result = add_periodic_boundary(data, (True, False, True))

    assert result.shape == (3, 3, 5, 2)
    np.testing.assert_array_equal(result[:2, :, :4], data)
    np.testing.assert_array_equal(result[-1, :, :4], result[0, :, :4])
    np.testing.assert_array_equal(result[:, :, -1], result[:, :, 0])


def test_add_periodic_boundary_without_periodicity_returns_independent_copy():
    data = np.arange(24).reshape(2, 3, 4)

    result = add_periodic_boundary(data, False)

    np.testing.assert_array_equal(result, data)
    assert not np.shares_memory(result, data)


def test_add_periodic_boundary_preserves_dtype_and_input():
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    before = data.copy()

    result = add_periodic_boundary(data, (True, True, True))

    assert result.dtype == data.dtype
    np.testing.assert_array_equal(data, before)


def test_add_periodic_boundary_requires_three_spatial_dimensions():
    with pytest.raises(ValueError, match="at least three spatial dimensions"):
        add_periodic_boundary(np.ones((3, 4)), True)
