import numpy as np
import pytest

from nematics3d.analysis.disclination.classification import (
    _build_defect_edges,
    _canonicalize_defect_indices_2x,
    defect_classify_into_lines,
)


def _line_point_sets(lines, box_size_periodic=(np.inf, np.inf, np.inf)):
    point_sets = []
    for line in lines:
        points = np.asarray(line.raw_defect_indices).copy()
        for axis, period in enumerate(box_size_periodic):
            if np.isfinite(period):
                points[:, axis] %= period
        point_sets.append(frozenset(map(tuple, points)))
    return set(point_sets)


def test_classify_open_chain():
    defects = np.array([[x, 0.5, 0.5] for x in range(5)])

    lines = defect_classify_into_lines(defects, log_mode="none")

    assert len(lines) == 1
    np.testing.assert_array_equal(lines[0].raw_defect_indices, defects)


def test_classify_connects_across_periodic_boundary():
    defects = np.array([[0.0, 0.5, 0.5], [3.0, 0.5, 0.5]])
    box_size = (4.0, np.inf, np.inf)

    nonperiodic = defect_classify_into_lines(defects, log_mode="none")
    periodic = defect_classify_into_lines(
        defects,
        box_size_periodic=box_size,
        log_mode="none",
    )

    assert nonperiodic == []
    assert len(periodic) == 1
    assert _line_point_sets(periodic, box_size) == {frozenset(map(tuple, defects))}
    assert abs(np.diff(periodic[0].raw_defect_indices[:, 0])[0]) == 1.0


def test_classify_separates_disconnected_components():
    first = np.array([[x, 0.5, 0.5] for x in range(4)])
    second = np.array([[x, 10.5, 10.5] for x in range(3)])
    defects = np.concatenate((first, second))

    lines = defect_classify_into_lines(defects, log_mode="none")

    assert _line_point_sets(lines) == {
        frozenset(map(tuple, first)),
        frozenset(map(tuple, second)),
    }


def test_branching_graph_consumes_every_edge_once():
    defects = np.array(
        [
            [0.0, 0.5, 0.5],
            [-1.0, 0.5, 0.5],
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
        ]
    )
    box_size = np.full(3, np.inf)
    points_2x = _canonicalize_defect_indices_2x(defects, box_size)
    edge_u, _ = _build_defect_edges(points_2x, box_size)

    lines = defect_classify_into_lines(defects, log_mode="none")

    assert sum(len(line.raw_defect_indices) - 1 for line in lines) == len(edge_u)
    assert set.union(
        *(set(map(tuple, line.raw_defect_indices)) for line in lines)
    ) == set(map(tuple, defects))


def test_classify_returns_no_lines_without_edges():
    isolated = np.array([[0.0, 0.5, 0.5], [20.0, 0.5, 0.5]])

    assert defect_classify_into_lines(isolated, log_mode="none") == []
    assert defect_classify_into_lines(np.empty((0, 3)), log_mode="none") == []


def test_classify_rejects_off_lattice_coordinates():
    defects = np.array([[0.6, 1.0, 0.5]])

    with pytest.raises(ValueError, match="integer or half-integer"):
        defect_classify_into_lines(defects, log_mode="none")


def test_classify_rejects_wrong_coordinate_parity():
    defects = np.array([[0.0, 1.0, 0.5]])

    with pytest.raises(ValueError, match="exactly one integer"):
        defect_classify_into_lines(defects, log_mode="none")


def test_classify_rejects_duplicate_periodic_points():
    defects = np.array([[0.0, 0.5, 0.5], [4.0, 0.5, 0.5]])

    with pytest.raises(ValueError, match="Duplicate defect indices"):
        defect_classify_into_lines(
            defects,
            box_size_periodic=(4.0, np.inf, np.inf),
            log_mode="none",
        )


@pytest.mark.parametrize(
    "defects",
    [np.ones((2, 2)), np.ones((2, 3, 1)), np.array([[np.nan, 0.5, 0.5]])],
)
def test_classify_rejects_invalid_arrays(defects):
    with pytest.raises(ValueError):
        defect_classify_into_lines(defects, log_mode="none")
