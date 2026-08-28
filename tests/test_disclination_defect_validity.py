import numpy as np
import pytest

from nematics3d.analysis.disclination import defect_validity_from_mask


PLAQUETTES = (
    (
        (2.0, 3.5, 4.5),
        ((2, 3, 4), (2, 3, 5), (2, 4, 4), (2, 4, 5)),
    ),
    (
        (2.5, 3.0, 4.5),
        ((2, 3, 4), (2, 3, 5), (3, 3, 4), (3, 3, 5)),
    ),
    (
        (2.5, 3.5, 4.0),
        ((2, 3, 4), (2, 4, 4), (3, 3, 4), (3, 4, 4)),
    ),
)


@pytest.mark.parametrize(("defect", "supporting_corners"), PLAQUETTES)
def test_defect_validity_requires_all_four_plaquette_corners(
    defect, supporting_corners
):
    mask = np.ones((6, 6, 6), dtype=bool)
    defect_indices = np.array([defect])

    assert defect_validity_from_mask(defect_indices, mask).tolist() == [True]

    for corner in supporting_corners:
        mask_with_invalid_corner = mask.copy()
        mask_with_invalid_corner[corner] = False
        assert defect_validity_from_mask(
            defect_indices, mask_with_invalid_corner
        ).tolist() == [False]

    mask[0, 0, 0] = False
    assert defect_validity_from_mask(defect_indices, mask).tolist() == [True]


def test_defect_validity_preserves_order_dtype_shape_and_inputs():
    defects = np.array(
        [
            (1.0, 1.5, 1.5),
            (2.5, 2.0, 2.5),
            (3.5, 3.5, 3.0),
        ]
    )
    mask = np.ones((5, 5, 5), dtype=bool)
    mask[1, 1, 1] = False
    defects_before = defects.copy()
    mask_before = mask.copy()

    result = defect_validity_from_mask(defects, mask)

    assert result.tolist() == [False, True, True]
    assert result.shape == (3,)
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(defects, defects_before)
    np.testing.assert_array_equal(mask, mask_before)


def test_defect_validity_wraps_periodic_plaquette_corners():
    defects = np.array([(1.0, 3.5, 3.5)])
    mask = np.ones((4, 4, 4), dtype=bool)

    assert defect_validity_from_mask(
        defects, mask, is_boundary_periodic=(False, True, True)
    ).tolist() == [True]

    mask[1, 0, 0] = False
    assert defect_validity_from_mask(
        defects, mask, is_boundary_periodic=(False, True, True)
    ).tolist() == [False]


def test_defect_validity_reports_each_nonperiodic_out_of_bounds_axis():
    defects = np.array([(1.0, 3.5, 3.5)])
    mask = np.ones((4, 4, 4), dtype=bool)

    with pytest.raises(ValueError) as exc_info:
        defect_validity_from_mask(defects, mask)

    message = str(exc_info.value)
    assert "axis 1" in message
    assert "axis 2" in message
    assert "valid range [0, 3]" in message


def test_defect_validity_accepts_empty_input():
    result = defect_validity_from_mask(np.empty((0, 3)), np.ones((2, 2, 2), dtype=bool))

    assert result.shape == (0,)
    assert result.dtype == np.bool_
