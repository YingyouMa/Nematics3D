import numpy as np
import pyvista as pv
import pytest

from nematics3d.analysis.disclination import defect_detect_surface


def _two_triangle_square():
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    faces = np.array([3, 0, 1, 2, 3, 0, 2, 3])
    return pv.PolyData(points, faces)


def test_defect_detect_surface_uniform_directors_have_no_defect():
    surface = _two_triangle_square()
    directors = np.tile([1.0, 0.0, 0.0], (4, 1))

    defects, mask = defect_detect_surface(
        surface, directors, is_simplify=False, is_return_mask=True
    )

    assert defects.shape == (0, 3)
    np.testing.assert_array_equal(mask, np.zeros(4, dtype=bool))


def test_defect_detect_surface_detects_planar_quad_winding():
    surface = _two_triangle_square()
    angles = np.deg2rad([0.0, 45.0, 90.0, 135.0])
    directors = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(4)))

    defects, mask = defect_detect_surface(
        surface, directors, is_simplify=False, is_return_mask=True
    )

    np.testing.assert_allclose(defects, [[0.5, 0.5, 0.0]])
    np.testing.assert_array_equal(mask, np.ones(4, dtype=bool))


def test_defect_detect_surface_normalizes_directors_and_rejects_zero():
    surface = _two_triangle_square()
    directors = np.tile([2.0, 0.0, 0.0], (4, 1))
    defects = defect_detect_surface(surface, directors)
    assert defects.shape == (0, 3)

    directors[0] = 0.0
    with pytest.raises(ValueError, match="zero directors"):
        defect_detect_surface(surface, directors)
