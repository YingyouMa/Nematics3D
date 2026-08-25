import itertools
from unittest.mock import patch

import numpy as np
import pytest

from nematics3d.disclination import defect_detect
from nematics3d.field import add_periodic_boundary, align_stack


def _legacy_xy_detector(n, threshold):
    a = n[:-1, :-1]
    b = n[1:, :-1]
    c = n[1:, 1:]
    d = n[:-1, 1:]
    aligned = align_stack(np.stack((a, b, c, d), axis=0))
    closure = np.einsum("...i,...i->...", aligned[0], aligned[-1])
    coordinates = np.argwhere(closure < threshold).astype(float)
    coordinates[:, :2] += 0.5
    return coordinates


def _legacy_defect_detect(n, threshold, is_boundary_periodic, planes):
    extended = add_periodic_boundary(n, is_boundary_periodic)
    chunks = []
    permutations = ((2, 1, 0), (0, 2, 1), (0, 1, 2))
    for axis, is_selected in enumerate(planes):
        if not is_selected:
            continue
        permutation = permutations[axis]
        rotated = np.moveaxis(extended, (0, 1, 2), permutation)
        coordinates = _legacy_xy_detector(rotated, threshold)
        chunks.append(coordinates[:, permutation])

    if not chunks:
        return np.empty((0, 3), dtype=float)

    coordinates = np.concatenate(chunks)
    for axis, is_periodic in enumerate(is_boundary_periodic):
        if is_periodic:
            coordinates[:, axis] %= n.shape[axis]
    return np.unique(coordinates, axis=0)


def _sorted_rows(values):
    return values[np.lexsort(values.T[::-1])]


def _director_from_xy_angles(angles):
    angles = np.asarray(angles)
    director = np.zeros((*angles.shape, 1, 3))
    director[..., 0] = np.cos(angles)[..., None]
    director[..., 1] = np.sin(angles)[..., None]
    return director


def test_defect_detect_finds_one_known_xy_defect():
    director = _director_from_xy_angles([[0.0, 3 * np.pi / 4], [np.pi / 4, np.pi / 2]])

    defects = defect_detect(
        director,
        planes=(False, False, True),
        worker_count=1,
    )

    np.testing.assert_array_equal(defects, [[0.5, 0.5, 0.0]])


def test_defect_detect_returns_empty_for_uniform_field():
    director = np.zeros((3, 3, 2, 3))
    director[..., 0] = 1.0

    defects = defect_detect(director, worker_count=1)

    assert defects.shape == (0, 3)


def test_defect_detect_finds_defect_crossing_periodic_x_boundary():
    director = _director_from_xy_angles(
        [
            [np.pi / 4, np.pi / 2],
            [0.0, 3 * np.pi / 4],
            [0.0, 3 * np.pi / 4],
        ]
    )

    nonperiodic = defect_detect(
        director,
        planes=(False, False, True),
        worker_count=1,
    )
    periodic = defect_detect(
        director,
        is_boundary_periodic=(True, False, False),
        planes=(False, False, True),
        worker_count=1,
    )

    boundary_defect = np.array([2.5, 0.5, 0.0])
    assert not np.any(np.all(nonperiodic == boundary_defect, axis=1))
    assert np.any(np.all(periodic == boundary_defect, axis=1))
    assert np.all(periodic[:, 0] < director.shape[0])
    assert len(periodic) == len(np.unique(periodic, axis=0))


def test_defect_detect_finds_defect_crossing_periodic_y_boundary():
    director = _director_from_xy_angles(
        [
            [np.pi / 4, np.pi / 2],
            [0.0, 3 * np.pi / 4],
            [0.0, 3 * np.pi / 4],
        ]
    )

    nonperiodic = defect_detect(
        director,
        planes=(False, False, True),
        worker_count=1,
    )
    periodic = defect_detect(
        director,
        is_boundary_periodic=(False, True, False),
        planes=(False, False, True),
        worker_count=1,
    )

    boundary_defect = np.array([0.5, 1.5, 0.0])
    assert not np.any(np.all(nonperiodic == boundary_defect, axis=1))
    assert np.any(np.all(periodic == boundary_defect, axis=1))
    assert np.all(periodic[:, 1] < director.shape[1])
    assert len(periodic) == len(np.unique(periodic, axis=0))


def test_defect_detect_finds_defect_crossing_periodic_z_boundary():
    xy_director = _director_from_xy_angles(
        [
            [np.pi / 4, np.pi / 2],
            [0.0, 3 * np.pi / 4],
            [0.0, 3 * np.pi / 4],
        ]
    )
    director = np.moveaxis(xy_director, (0, 1, 2), (2, 0, 1))

    nonperiodic = defect_detect(
        director,
        planes=(False, True, False),
        worker_count=1,
    )
    periodic = defect_detect(
        director,
        is_boundary_periodic=(False, False, True),
        planes=(False, True, False),
        worker_count=1,
    )

    boundary_defect = np.array([0.5, 0.0, 2.5])
    assert not np.any(np.all(nonperiodic == boundary_defect, axis=1))
    assert np.any(np.all(periodic == boundary_defect, axis=1))
    assert np.all(periodic[:, 2] < director.shape[2])
    assert len(periodic) == len(np.unique(periodic, axis=0))


def test_periodic_plaquette_normal_does_not_duplicate_defects():
    director = _director_from_xy_angles([[0.0, 3 * np.pi / 4], [np.pi / 4, np.pi / 2]])

    nonperiodic = defect_detect(
        director,
        planes=(False, False, True),
        worker_count=1,
    )
    periodic_normal = defect_detect(
        director,
        is_boundary_periodic=(False, False, True),
        planes=(False, False, True),
        worker_count=1,
    )

    np.testing.assert_array_equal(periodic_normal, nonperiodic)


def test_defect_detect_matches_legacy_for_periodic_boundaries_and_planes():
    rng = np.random.default_rng(1827)
    director = rng.normal(size=(4, 5, 3, 3))
    director /= np.linalg.norm(director, axis=-1, keepdims=True)

    periodic_cases = ((False, False, False), (True, False, True))
    plane_cases = ((True, True, True), (True, False, True))
    for periodic, planes in itertools.product(periodic_cases, plane_cases):
        expected = _legacy_defect_detect(director, 0.0, periodic, planes)
        actual = defect_detect(
            director,
            is_boundary_periodic=periodic,
            planes=planes,
            worker_count=1,
        )
        np.testing.assert_array_equal(_sorted_rows(actual), _sorted_rows(expected))


def test_defect_detect_validated_and_trusted_paths_agree():
    rng = np.random.default_rng(29)
    director = rng.normal(size=(5, 4, 3, 3)).astype(np.float32)

    validated = defect_detect(director, worker_count=1)
    trusted = defect_detect(
        director,
        worker_count=1,
        is_input_validated=True,
    )

    np.testing.assert_array_equal(validated, trusted)


def test_defect_detect_trusted_path_skips_director_validation():
    director = np.zeros((2, 2, 1, 3))
    director[..., 0] = 1.0

    with patch(
        "nematics3d.disclination.as_director_field",
        side_effect=AssertionError("director validation was called"),
    ):
        defects = defect_detect(
            director,
            worker_count=1,
            is_input_validated=True,
        )

    assert defects.shape == (0, 3)


def test_defect_detect_rejects_invalid_shape_by_default():
    with pytest.raises(ValueError, match="Nx, Ny, Nz, 3"):
        defect_detect(np.ones((4, 4, 3)), worker_count=1)


def test_defect_detect_rejects_nonfinite_field_by_default():
    director = np.ones((2, 2, 1, 3))
    director[0, 0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        defect_detect(director, worker_count=1)


@pytest.mark.parametrize("is_input_validated", [0, 1, "yes", None])
def test_defect_detect_requires_boolean_validation_flag(is_input_validated):
    director = np.ones((2, 2, 1, 3))

    with pytest.raises(TypeError, match="must be a boolean"):
        defect_detect(director, is_input_validated=is_input_validated)


@pytest.mark.parametrize(
    ("worker_count", "error_type"),
    [(0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError)],
)
def test_defect_detect_rejects_invalid_worker_count(worker_count, error_type):
    director = np.ones((2, 2, 1, 3))

    with pytest.raises(error_type, match="worker_count"):
        defect_detect(director, worker_count=worker_count)
