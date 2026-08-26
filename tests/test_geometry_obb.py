import numpy as np
import pytest

from nematics3d.datatypes import as_points
from nematics3d.geometry import (
    canonicalize_axes,
    compute_convex_hull_points,
    obb_fit_approx,
    obb_fit_pca,
    obb_refine_random_search,
)


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

    with pytest.raises(ValueError, match="at least 2 point"):
        as_points(points, is_unique=True, min_num=2)


def test_canonicalize_axes_makes_signs_deterministic_and_right_handed():
    axes = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    canonical_axes = canonicalize_axes(axes)

    np.testing.assert_allclose(canonical_axes, np.eye(3))
    assert np.linalg.det(canonical_axes) == pytest.approx(1.0)


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


def test_obb_fit_pca_axis_aligned_box():
    x = [-2.0, 2.0]
    y = [-1.5, 1.5]
    z = [-1.0, 1.0]
    points = np.array([[xi, yi, zi] for xi in x for yi in y for zi in z])

    fit = obb_fit_pca(points)

    np.testing.assert_allclose(np.sort(fit.lengths), [2.0, 3.0, 4.0])
    np.testing.assert_allclose(fit.center, [0.0, 0.0, 0.0], atol=1e-12)
    assert fit.volume == pytest.approx(24.0)
    np.testing.assert_allclose(fit.axes.T @ fit.axes, np.eye(3), atol=1e-12)
    assert np.linalg.det(fit.axes) == pytest.approx(1.0)


def test_obb_fit_supports_result_base_inspection():
    fit = obb_fit_pca([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    assert fit.keys() == (
        "axes",
        "center",
        "lengths",
        "local_min",
        "local_max",
        "volume",
    )
    assert fit["volume"] == pytest.approx(0.0)
    assert fit.get("missing", "fallback") == "fallback"
    assert "center" in fit
    assert set(fit.asdict()) == set(fit.keys())

    fit_repr = repr(fit)
    assert fit_repr.startswith(
        "OBBFit: The parameters of an oriented bounding-box fit\n"
    )
    assert "  axes      =" in fit_repr
    assert "too many elements to display" not in fit_repr
    assert "[[ 1.," in fit_repr
    assert "\n               [ 0.," in fit_repr
    assert "  local_min =" in fit_repr
    assert "  volume    = 0," in fit_repr


def test_obb_fit_pca_tracks_translated_center():
    offset = np.array([10.0, -2.0, 4.5])
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [2.0, 4.0, 0.0],
        ]
    )

    fit = obb_fit_pca(points + offset)

    np.testing.assert_allclose(fit.center, offset + [1.0, 2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.sort(fit.lengths), [0.0, 2.0, 4.0], atol=1e-12)
    assert fit.volume == pytest.approx(0.0)


def test_obb_fit_pca_handles_single_point():
    fit = obb_fit_pca([1.0, 2.0, 3.0])

    np.testing.assert_allclose(fit.center, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(fit.lengths, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(fit.axes, np.eye(3))
    assert fit.volume == pytest.approx(0.0)


def test_obb_refine_random_search_does_not_worsen_initial_fit():
    rng = np.random.default_rng(123)
    points = rng.normal(size=(30, 3))
    initial_fit = obb_fit_pca(points)

    refined_fit = obb_refine_random_search(
        points,
        initial_fit,
        angle_scales_deg=(10.0, 2.0),
        trials_per_scale=8,
        seed=456,
    )

    assert refined_fit.volume <= initial_fit.volume
    np.testing.assert_allclose(refined_fit.axes.T @ refined_fit.axes, np.eye(3))
    assert np.linalg.det(refined_fit.axes) == pytest.approx(1.0)


def test_obb_refine_random_search_is_reproducible_with_seed():
    rng = np.random.default_rng(789)
    points = rng.normal(size=(24, 3))
    initial_fit = obb_fit_pca(points)

    refined_fit_a = obb_refine_random_search(
        points,
        initial_fit,
        angle_scales_deg=(12.0, 3.0),
        trials_per_scale=10,
        seed=100,
    )
    refined_fit_b = obb_refine_random_search(
        points,
        initial_fit,
        angle_scales_deg=(12.0, 3.0),
        trials_per_scale=10,
        seed=100,
    )

    np.testing.assert_allclose(refined_fit_a.axes, refined_fit_b.axes)
    np.testing.assert_allclose(refined_fit_a.center, refined_fit_b.center)
    np.testing.assert_allclose(refined_fit_a.lengths, refined_fit_b.lengths)
    assert refined_fit_a.volume == pytest.approx(refined_fit_b.volume)


def test_obb_fit_approx_uses_hull_pca_and_refinement():
    rng = np.random.default_rng(321)
    points = rng.normal(size=(40, 3))
    hull_points = compute_convex_hull_points(points)
    initial_fit = obb_fit_pca(hull_points)

    fit = obb_fit_approx(
        points,
        angle_scales_deg=(8.0, 2.0),
        trials_per_scale=8,
        seed=654,
    )

    assert fit.volume <= initial_fit.volume
    np.testing.assert_allclose(fit.axes.T @ fit.axes, np.eye(3))
    assert np.linalg.det(fit.axes) == pytest.approx(1.0)


def test_obb_fit_approx_is_reproducible_with_seed():
    rng = np.random.default_rng(987)
    points = rng.normal(size=(32, 3))

    fit_a = obb_fit_approx(
        points,
        angle_scales_deg=(9.0, 3.0),
        trials_per_scale=8,
        seed=111,
    )
    fit_b = obb_fit_approx(
        points,
        angle_scales_deg=(9.0, 3.0),
        trials_per_scale=8,
        seed=111,
    )

    np.testing.assert_allclose(fit_a.axes, fit_b.axes)
    np.testing.assert_allclose(fit_a.center, fit_b.center)
    np.testing.assert_allclose(fit_a.lengths, fit_b.lengths)
    assert fit_a.volume == pytest.approx(fit_b.volume)
