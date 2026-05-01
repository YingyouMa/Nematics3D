import numpy as np
import pytest

from nematics3d.datatypes import as_points
from nematics3d.geometry import compute_convex_hull_points


def test_as_points_can_deduplicate_and_check_min_num():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    unique_points = as_points(points, is_unique=True, min_num=2)

    np.testing.assert_allclose(
        unique_points,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )


def test_as_points_raises_when_unique_points_are_too_few():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    with pytest.raises(TypeError, match="at least 2 point"):
        as_points(points, is_unique=True, min_num=2)


def test_compute_convex_hull_points_removes_interior_point():
    cube = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
        ]
    )

    hull_points = compute_convex_hull_points(cube)

    assert hull_points.shape == (8, 3)
    assert not np.any(np.all(np.isclose(hull_points, [0.5, 0.5, 0.5]), axis=1))


def test_compute_convex_hull_points_deduplicates_small_input():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    hull_points = compute_convex_hull_points(points)

    np.testing.assert_allclose(hull_points, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def test_compute_convex_hull_points_falls_back_for_coplanar_points():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )

    hull_points = compute_convex_hull_points(points)

    np.testing.assert_allclose(hull_points, np.unique(points, axis=0))
