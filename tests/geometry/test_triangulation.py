import numpy as np
import pytest

from nematics3d.geometry import triangulate_surface_points


def _canonical_faces(mesh):
    faces = np.asarray(mesh.faces, dtype=int).reshape(-1, 4)[:, 1:]
    return sorted(tuple(sorted(face)) for face in faces)


def _fibonacci_sphere(n):
    indices = np.arange(n, dtype=float)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / n
    radius_xy = np.sqrt(1.0 - z * z)
    phi = golden_angle * indices
    return np.column_stack(
        [
            radius_xy * np.cos(phi),
            radius_xy * np.sin(phi),
            z,
        ]
    )


def test_triangulate_surface_points_tetrahedron():
    points = np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ]
    )

    mesh = triangulate_surface_points(points)

    assert mesh.n_points == 4
    assert mesh.n_cells == 4
    np.testing.assert_allclose(mesh.points, points)
    assert np.all(np.asarray(mesh.faces).reshape(-1, 4)[:, 0] == 3)


def test_triangulate_surface_points_sphere():
    n_points = 100
    points = _fibonacci_sphere(n_points)

    mesh = triangulate_surface_points(points)

    assert mesh.n_points == n_points
    assert mesh.n_cells == 2 * n_points - 4
    np.testing.assert_allclose(mesh.points, points)
    assert np.all(np.asarray(mesh.faces).reshape(-1, 4)[:, 0] == 3)


def test_triangulate_surface_points_translation_invariant():
    points = _fibonacci_sphere(64)
    translation = np.array([10.0, -7.0, 3.0])

    mesh = triangulate_surface_points(points)
    translated_mesh = triangulate_surface_points(points + translation)

    assert _canonical_faces(mesh) == _canonical_faces(translated_mesh)
    np.testing.assert_allclose(translated_mesh.points, points + translation)


def test_triangulate_surface_points_centroid_collision():
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="centroid"):
        triangulate_surface_points(points)
