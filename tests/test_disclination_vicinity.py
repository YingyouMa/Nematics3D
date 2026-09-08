import numpy as np
import pytest

from nematics3d.analysis.disclination import defect_vicinity_grid
from nematics3d.analysis.disclination.line import get_square, get_square_each


def test_get_square_each_has_expected_boundary_points():
    square = get_square_each(1.0, 2, dim=2)
    np.testing.assert_allclose(
        square,
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    )


def test_get_square_validates_geometry_contract():
    with pytest.raises(ValueError, match="positive finite"):
        get_square_each(0.0, 2)
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        get_square_each(1.0, 1)
    with pytest.raises(ValueError, match="either 2 or 3"):
        get_square_each(1.0, 2, dim=4)
    with pytest.raises(ValueError, match="shape"):
        get_square([1.0], [2], origin_list=[[0.0, 0.0]], dim=3)


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
