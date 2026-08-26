import numpy as np
import pytest

from nematics3d.datatypes import as_defect_index


def test_as_defect_index_canonicalizes_half_grid_coordinates():
    values = np.array(
        [
            [1.0 + 1e-10, 2.5 - 1e-10, 3.5],
            [4.5, 5.0, 6.5],
        ]
    )

    result = as_defect_index(values)

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [1.0, 2.5, 3.5],
                [4.5, 5.0, 6.5],
            ]
        ),
    )


def test_as_defect_index_accepts_an_empty_collection():
    result = as_defect_index(np.empty((0, 3)))

    assert result.shape == (0, 3)
    assert np.issubdtype(result.dtype, np.floating)


@pytest.mark.parametrize(
    ("values", "error_type", "message"),
    [
        (np.array([1.0, 2.5, 3.5]), ValueError, r"shape \(N, 3\)"),
        (np.array([[1.0, 2.5, np.nan]]), ValueError, "finite"),
        (np.array([[1.0, 2.4, 3.5]]), ValueError, "integer or half-integer"),
        (np.array([[1.0, 2.0, 3.5]]), ValueError, "exactly one integer"),
        (np.array([[1.0 + 1.0j, 2.5, 3.5]]), TypeError, "real numbers"),
        (np.array([["1", "2.5", "3.5"]]), TypeError, "real numbers"),
    ],
)
def test_as_defect_index_rejects_invalid_input(values, error_type, message):
    with pytest.raises(error_type, match=message):
        as_defect_index(values)


@pytest.mark.parametrize("tolerance", [True, -1.0, np.inf, "small"])
def test_as_defect_index_rejects_invalid_tolerance(tolerance):
    with pytest.raises((TypeError, ValueError), match="tolerance"):
        as_defect_index(np.empty((0, 3)), tolerance=tolerance)
