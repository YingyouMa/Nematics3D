import numpy as np
import pytest

from nematics3d.classes.bounds import (
    Bounds,
    OptsBounds,
    bounds_expanded,
    bounds_minimal_wrapping_points,
    bounds_sample_points,
    obb_bounds_from_fit,
)
from nematics3d.geometry import OBBFit, box_corners_from_center_axes_radii


def _match_points_unordered(points_a, points_b, tol=1e-8):
    used = np.zeros(len(points_b), dtype=bool)
    for pa in points_a:
        diff = np.linalg.norm(points_b - pa, axis=1)
        idx = int(np.argmin(diff))
        if used[idx] or diff[idx] > tol:
            return False
        used[idx] = True
    return True


def test_obb_bounds_from_fit_builds_center_aligned_bounds():
    axes = np.eye(3)
    fit = OBBFit(
        axes=axes,
        center=np.array([10.0, -2.0, 4.0]),
        lengths=np.array([4.0, 2.0, 6.0]),
        local_min=np.array([-2.0, -1.0, -3.0]),
        local_max=np.array([2.0, 1.0, 3.0]),
        volume=48.0,
    )

    bounds = obb_bounds_from_fit(fit, name="seed bounds")

    assert isinstance(bounds, Bounds)
    assert bounds.name == "seed bounds"
    assert bounds.opts.alignment == "center"
    np.testing.assert_allclose(bounds.opts.origin, fit.center)
    np.testing.assert_allclose(bounds.opts.axis1, axes[:, 0])
    np.testing.assert_allclose(bounds.opts.axis2, axes[:, 1])
    assert bounds.opts.length1 == pytest.approx(4.0)
    assert bounds.opts.length2 == pytest.approx(2.0)
    assert bounds.opts.length3 == pytest.approx(6.0)
    np.testing.assert_allclose(bounds.corners.min(axis=0), [8.0, -3.0, 1.0])
    np.testing.assert_allclose(bounds.corners.max(axis=0), [12.0, -1.0, 7.0])


def test_obb_bounds_from_fit_requires_obb_fit():
    with pytest.raises(TypeError, match="OBBFit"):
        obb_bounds_from_fit(object())


def test_bounds_minimal_wrapping_points_uses_supplied_axes():
    angle = np.deg2rad(30.0)
    axes = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    origin = np.array([10.0, -3.0, 2.0])
    local_x = [-1.0, 3.0]
    local_y = [2.0, 5.0]
    local_z = [-2.0, 4.0]
    local_corners = np.array(
        [[x, y, z] for x in local_x for y in local_y for z in local_z]
    )
    points = origin + local_corners @ axes.T

    bounds = bounds_minimal_wrapping_points(
        points,
        axes,
        origin=origin,
        name="minimal test bounds",
    )

    expected_center = origin + axes @ np.array([1.0, 3.5, 1.0])
    assert bounds.name == "minimal test bounds"
    assert bounds.opts.alignment == "center"
    np.testing.assert_allclose(bounds.opts.origin, expected_center)
    np.testing.assert_allclose(bounds.opts.axis1, axes[:, 0])
    np.testing.assert_allclose(bounds.opts.axis2, axes[:, 1])
    np.testing.assert_allclose(
        [bounds.opts.length1, bounds.opts.length2, bounds.opts.length3],
        [4.0, 3.0, 6.0],
    )
    assert _match_points_unordered(bounds.corners, points)


def test_bounds_minimal_wrapping_points_floors_degenerate_lengths():
    point = np.array([1.0, 2.0, 3.0])

    bounds = bounds_minimal_wrapping_points(
        point,
        np.eye(3),
        min_lengths=(0.5, 0.25, 0.125),
    )

    np.testing.assert_allclose(bounds.opts.origin, point)
    np.testing.assert_allclose(
        [bounds.opts.length1, bounds.opts.length2, bounds.opts.length3],
        [0.5, 0.25, 0.125],
    )

    scalar_bounds = bounds_minimal_wrapping_points(
        point,
        np.eye(3),
        min_lengths=0.75,
    )

    np.testing.assert_allclose(
        [
            scalar_bounds.opts.length1,
            scalar_bounds.opts.length2,
            scalar_bounds.opts.length3,
        ],
        [0.75, 0.75, 0.75],
    )


def test_box_corners_from_center_axes_radii_builds_axis_aligned_box():
    corners = box_corners_from_center_axes_radii(
        center=(10.0, 20.0, 30.0),
        axes=np.eye(3),
        radii=(2.0, 3.0, 4.0),
    )

    assert corners.shape == (8, 3)
    np.testing.assert_allclose(corners.min(axis=0), [8.0, 17.0, 26.0])
    np.testing.assert_allclose(corners.max(axis=0), [12.0, 23.0, 34.0])


def test_box_corners_from_center_axes_radii_uses_supplied_axes():
    angle = np.deg2rad(30.0)
    axes = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    center = np.array([10.0, -3.0, 2.0])
    radii = np.array([2.0, 1.5, 3.0])

    corners = box_corners_from_center_axes_radii(center, axes, radii)
    expected = (
        center
        + np.array(
            [
                [-2.0, -1.5, -3.0],
                [2.0, -1.5, -3.0],
                [-2.0, 1.5, -3.0],
                [-2.0, -1.5, 3.0],
                [2.0, 1.5, -3.0],
                [2.0, -1.5, 3.0],
                [-2.0, 1.5, 3.0],
                [2.0, 1.5, 3.0],
            ]
        )
        @ axes.T
    )

    np.testing.assert_allclose(corners, expected)


def test_box_corners_from_center_axes_radii_validates_inputs():
    with pytest.raises(ValueError, match="radii"):
        box_corners_from_center_axes_radii(
            center=(0.0, 0.0, 0.0),
            axes=np.eye(3),
            radii=(1.0, 0.0, 1.0),
        )

    with pytest.raises(ValueError, match="orthonormal"):
        box_corners_from_center_axes_radii(
            center=(0.0, 0.0, 0.0),
            axes=np.array(
                [
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            radii=(1.0, 1.0, 1.0),
        )


def test_bounds_expanded_scales_lengths_about_existing_center():
    base = Bounds(
        name="base",
        opts=OptsBounds(
            origin=(1.0, 2.0, 3.0),
            axis1=(1.0, 0.0, 0.0),
            axis2=(0.0, 1.0, 0.0),
            length1=2.0,
            length2=4.0,
            length3=6.0,
            alignment="center",
        ),
    )

    expanded = bounds_expanded(
        base,
        expand_factors=(2.0, 0.5, 1.5),
        name="expanded",
    )

    assert expanded.name == "expanded"
    assert expanded.opts.alignment == "center"
    np.testing.assert_allclose(expanded.opts.origin, base.opts.origin)
    np.testing.assert_allclose(expanded.opts.axis1, base.opts.axis1)
    np.testing.assert_allclose(expanded.opts.axis2, base.opts.axis2)
    np.testing.assert_allclose(
        [expanded.opts.length1, expanded.opts.length2, expanded.opts.length3],
        [4.0, 2.0, 9.0],
    )


def test_bounds_expanded_applies_min_lengths():
    base = Bounds(
        opts=OptsBounds(
            origin=(0.0, 0.0, 0.0),
            axis1=(1.0, 0.0, 0.0),
            axis2=(0.0, 1.0, 0.0),
            length1=2.0,
            length2=3.0,
            length3=4.0,
            alignment="center",
        ),
    )

    expanded = bounds_expanded(
        base,
        expand_factors=(0.5, 0.5, 0.5),
        min_lengths=(5.0, 1.0, 3.0),
    )

    np.testing.assert_allclose(
        [expanded.opts.length1, expanded.opts.length2, expanded.opts.length3],
        [5.0, 1.5, 3.0],
    )

    scalar_expanded = bounds_expanded(
        base,
        expand_factors=(0.5, 0.5, 0.5),
        min_lengths=3.5,
    )

    np.testing.assert_allclose(
        [
            scalar_expanded.opts.length1,
            scalar_expanded.opts.length2,
            scalar_expanded.opts.length3,
        ],
        [3.5, 3.5, 3.5],
    )


def test_bounds_expanded_validates_inputs():
    base = Bounds(
        opts=OptsBounds(
            origin=(0.0, 0.0, 0.0),
            axis1=(1.0, 0.0, 0.0),
            axis2=(0.0, 1.0, 0.0),
            length1=1.0,
            alignment="center",
        ),
    )

    with pytest.raises(TypeError, match="Bounds"):
        bounds_expanded(object(), expand_factors=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="expand_factors"):
        bounds_expanded(base, expand_factors=(1.0, 1.0))
    with pytest.raises(ValueError, match="positive"):
        bounds_expanded(base, expand_factors=(1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="min_lengths"):
        bounds_expanded(base, expand_factors=(1.0, 1.0, 1.0), min_lengths=(1.0, 1.0))


def test_bounds_sample_points_uses_center_aligned_bounds():
    base = Bounds(
        opts=OptsBounds(
            origin=(10.0, 20.0, 30.0),
            axis1=(1.0, 0.0, 0.0),
            axis2=(0.0, 1.0, 0.0),
            length1=2.0,
            length2=4.0,
            length3=6.0,
            alignment="center",
        ),
    )

    points, local_points = bounds_sample_points(
        base,
        spacing=(1.0, 2.0, 3.0),
        is_return_local=True,
    )

    assert points.shape == (27, 3)
    assert local_points.shape == (27, 3)
    np.testing.assert_allclose(local_points.min(axis=0), [-1.0, -2.0, -3.0])
    np.testing.assert_allclose(local_points.max(axis=0), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(points.min(axis=0), [9.0, 18.0, 27.0])
    np.testing.assert_allclose(points.max(axis=0), [11.0, 22.0, 33.0])


def test_bounds_sample_points_uses_min_corner_aligned_bounds():
    base = Bounds(
        opts=OptsBounds(
            origin=(1.0, 2.0, 3.0),
            axis1=(1.0, 0.0, 0.0),
            axis2=(0.0, 1.0, 0.0),
            length1=2.0,
            length2=2.0,
            length3=2.0,
            alignment="min_corner",
        ),
    )

    points = bounds_sample_points(base, spacing=1.0)

    assert points.shape == (27, 3)
    np.testing.assert_allclose(points.min(axis=0), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(points.max(axis=0), [3.0, 4.0, 5.0])


def test_bounds_sample_points_validates_inputs():
    base = Bounds(
        opts=OptsBounds(
            origin=(0.0, 0.0, 0.0),
            axis1=(1.0, 0.0, 0.0),
            axis2=(0.0, 1.0, 0.0),
            length1=1.0,
            alignment="center",
        ),
    )

    with pytest.raises(TypeError, match="Bounds"):
        bounds_sample_points(object(), spacing=1.0)
    with pytest.raises(ValueError, match="spacing"):
        bounds_sample_points(base, spacing=(1.0, 1.0))
    with pytest.raises(ValueError, match="positive"):
        bounds_sample_points(base, spacing=0.0)
