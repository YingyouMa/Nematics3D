import numpy as np
import pytest

from nematics3d.analysis.disclination import defect_vicinity_grid


def test_defect_vicinity_grid_covers_all_three_plaquette_orientations():
    defects = np.array(
        [
            [1.0, 2.5, 3.5],
            [1.5, 2.0, 3.5],
            [1.5, 2.5, 3.0],
        ]
    )
    result = defect_vicinity_grid(defects, num_shell=1)

    assert result.shape == (3, 4, 3)
    assert np.issubdtype(result.dtype, np.integer)
    np.testing.assert_array_equal(result[0, :, 0], np.ones(4, dtype=int))
    np.testing.assert_array_equal(result[1, :, 1], np.full(4, 2, dtype=int))
    np.testing.assert_array_equal(result[2, :, 2], np.full(4, 3, dtype=int))


def test_defect_vicinity_grid_empty_shape_and_num_shell_validation():
    empty = defect_vicinity_grid(np.empty((0, 3)), num_shell=3)
    assert empty.shape == (0, 36, 3)
    assert np.issubdtype(empty.dtype, np.integer)

    with pytest.raises(ValueError, match="positive integer"):
        defect_vicinity_grid(np.empty((0, 3)), num_shell=0)
    with pytest.raises(TypeError, match="positive integer"):
        defect_vicinity_grid(np.empty((0, 3)), num_shell=1.5)
