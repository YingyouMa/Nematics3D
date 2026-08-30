"""Focused tests for physical plane-axis resolution."""

import numpy as np
import pytest

from nematics3d.grid import resolve_plane_physical_axes


def _assert_orthonormal_plane_basis(normal, axis1, axis2):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)

    np.testing.assert_allclose(np.linalg.norm(axis1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(axis2), 1.0, atol=1e-12)
    np.testing.assert_allclose(normal @ axis1, 0.0, atol=1e-12)
    np.testing.assert_allclose(normal @ axis2, 0.0, atol=1e-12)
    np.testing.assert_allclose(axis1 @ axis2, 0.0, atol=1e-12)
    np.testing.assert_allclose(np.cross(normal, axis1), axis2, atol=1e-12)


def test_resolve_plane_physical_axes_keeps_valid_axis_and_normalizes_inputs():
    axis1, axis2 = resolve_plane_physical_axes(
        normal=[0.0, 0.0, 4.0],
        axis1=[3.0, 0.0, 0.0],
        is_warn=False,
    )

    np.testing.assert_allclose(axis1, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(axis2, [0.0, 1.0, 0.0])
    _assert_orthonormal_plane_basis([0.0, 0.0, 4.0], axis1, axis2)


def test_resolve_plane_physical_axes_generates_axis_when_axis1_missing():
    normal = np.array([1.0, 2.0, 3.0])

    axis1, axis2 = resolve_plane_physical_axes(normal, is_warn=False)

    _assert_orthonormal_plane_basis(normal, axis1, axis2)


def test_resolve_plane_physical_axes_projects_nonperpendicular_axis1():
    axis1, axis2 = resolve_plane_physical_axes(
        normal=[0.0, 0.0, 1.0],
        axis1=[1.0, 0.0, 1.0],
        is_warn=False,
    )

    np.testing.assert_allclose(axis1, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(axis2, [0.0, 1.0, 0.0], atol=1e-12)


def test_resolve_plane_physical_axes_falls_back_for_collinear_axis1():
    axis1, axis2 = resolve_plane_physical_axes(
        normal=[0.0, 0.0, 2.0],
        axis1=[0.0, 0.0, -5.0],
        is_warn=False,
    )

    np.testing.assert_allclose(axis1, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(axis2, [0.0, 1.0, 0.0], atol=1e-12)


def test_resolve_plane_physical_axes_rejects_invalid_is_warn():
    with pytest.raises(TypeError):
        resolve_plane_physical_axes([0.0, 0.0, 1.0], is_warn="False")


@pytest.mark.parametrize(
    "normal, axis1",
    [
        ([0.0, 0.0, 0.0], None),
        ([0.0, np.nan, 1.0], None),
        ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, 1.0], ["1", "0", "0"]),
        ([0.0, 0.0], None),
    ],
)
def test_resolve_plane_physical_axes_rejects_invalid_vectors(normal, axis1):
    with pytest.raises((TypeError, ValueError)):
        resolve_plane_physical_axes(normal, axis1, is_warn=False)
