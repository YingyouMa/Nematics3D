import numpy as np
import pytest

from nematics3d.datatypes import as_lattice_mask, as_real_lattice_field
from nematics3d.classes.grid_field import GridFieldDataset, InputGridField


def test_real_lattice_field_scalar_and_component_shapes():
    scalar = np.arange(24).reshape(2, 3, 4)
    vector = np.zeros((2, 3, 4, 3))

    out_scalar = as_real_lattice_field(scalar, extra_ndim=0)
    out_vector = as_real_lattice_field(vector, extra_ndim=1)

    assert out_scalar.shape == (2, 3, 4)
    assert out_vector.shape == (2, 3, 4, 3)
    assert np.issubdtype(out_scalar.dtype, np.floating)


def test_real_lattice_field_requires_three_nonempty_lattice_axes():
    with pytest.raises(ValueError):
        as_real_lattice_field(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        as_real_lattice_field(np.zeros((2, 0, 3)))


def test_real_lattice_field_extra_ndim_validation():
    values = np.zeros((2, 3, 4, 3))
    assert as_real_lattice_field(values, extra_ndim=1).shape == values.shape

    with pytest.raises(ValueError):
        as_real_lattice_field(values, extra_ndim=0)
    with pytest.raises(ValueError):
        as_real_lattice_field(values, extra_ndim=-1)
    with pytest.raises((TypeError, ValueError)):
        as_real_lattice_field(values, extra_ndim=1.5)


def test_real_lattice_field_exact_shape_is_strict():
    values = np.zeros((2, 3, 4))
    assert as_real_lattice_field(values, shape=(2, 3, 4)).shape == values.shape

    with pytest.raises(ValueError):
        as_real_lattice_field(values, shape=(2, 3, 5))
    with pytest.raises((TypeError, ValueError)):
        as_real_lattice_field(values, shape=(2.5, 3, 4))
    with pytest.raises((TypeError, ValueError)):
        as_real_lattice_field(values, shape=(True, 3, 4))
    with pytest.raises(ValueError):
        as_real_lattice_field(values, shape=(0, 3, 4))


def test_real_lattice_field_rejects_non_numeric_and_complex():
    with pytest.raises(TypeError):
        as_real_lattice_field(np.full((2, 2, 2), "x"))
    with pytest.raises(TypeError):
        as_real_lattice_field(np.ones((2, 2, 2), dtype=complex))


def test_real_lattice_field_finite_contract():
    values = np.zeros((2, 2, 2))
    values[0, 0, 0] = np.nan

    with pytest.raises(ValueError):
        as_real_lattice_field(values)

    out = as_real_lattice_field(values, is_finite=False)
    assert np.isnan(out[0, 0, 0])

    with pytest.raises((TypeError, ValueError)):
        as_real_lattice_field(values, is_finite="False")


def test_real_lattice_field_value_range_rejects_or_clips():
    values = np.array([[[[-1.0, 0.5, 2.0]]]])

    with pytest.raises(ValueError):
        as_real_lattice_field(values, value_range=(0.0, 1.0))

    out = as_real_lattice_field(values, value_range=(0.0, 1.0), bounded=True)
    np.testing.assert_array_equal(out, np.array([[[[0.0, 0.5, 1.0]]]]))

    with pytest.raises((TypeError, ValueError)):
        as_real_lattice_field(values, bounded=1)


def test_real_lattice_field_nonfinite_with_range_preserves_nan():
    values = np.array([[[np.nan, 0.5]]])
    out = as_real_lattice_field(values, is_finite=False, value_range=(0.0, 1.0))
    assert np.isnan(out[0, 0, 0])
    assert out[0, 0, 1] == 0.5


def test_real_lattice_field_avoids_unnecessary_copy_for_float_input():
    values = np.zeros((2, 2, 2), dtype=float)
    out = as_real_lattice_field(values)
    assert np.shares_memory(out, values)


def test_lattice_mask_accepts_boolean_and_returns_independent_bool_array():
    mask = np.array([[[True, False], [False, True]], [[False, True], [True, False]]])
    out = as_lattice_mask(mask)

    assert out.dtype == np.bool_
    np.testing.assert_array_equal(out, mask)
    assert not np.shares_memory(out, mask)


def test_lattice_mask_accepts_numeric_zero_one():
    mask = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
    out = as_lattice_mask(mask)
    assert out.dtype == np.bool_
    np.testing.assert_array_equal(out, mask.astype(bool))
    assert not np.shares_memory(out, mask)


def test_lattice_mask_rejects_non_binary_numeric_values():
    mask = np.zeros((2, 2, 2))
    mask[0, 0, 0] = 0.5
    with pytest.raises(ValueError):
        as_lattice_mask(mask)


def test_lattice_mask_requires_exactly_three_axes_and_expected_shape():
    with pytest.raises(ValueError):
        as_lattice_mask(True)
    with pytest.raises(ValueError):
        as_lattice_mask(np.zeros((2, 2), dtype=bool))
    with pytest.raises(ValueError):
        as_lattice_mask(np.zeros((2, 2, 2, 1), dtype=bool))
    with pytest.raises(ValueError):
        as_lattice_mask(np.zeros((2, 2, 2), dtype=bool), shape=(2, 2, 3))
    with pytest.raises(ValueError):
        as_lattice_mask(np.zeros((2, 0, 2), dtype=bool))


@pytest.mark.parametrize(
    "invalid_shape",
    ((2, 2), (2, 2, 2, 1), (2.0, 2, 2), (True, 2, 2), (2, 0, 2)),
)
def test_lattice_mask_expected_shape_uses_standard_grid_shape_contract(
    invalid_shape,
):
    with pytest.raises((TypeError, ValueError)):
        as_lattice_mask(
            np.ones((2, 2, 2), dtype=bool),
            shape=invalid_shape,
        )


def test_lattice_mask_rejects_nonfinite_complex_and_non_numeric():
    nan_mask = np.zeros((2, 2, 2))
    nan_mask[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        as_lattice_mask(nan_mask)
    with pytest.raises(TypeError):
        as_lattice_mask(np.ones((2, 2, 2), dtype=complex))
    with pytest.raises(TypeError):
        as_lattice_mask(np.full((2, 2, 2), "x"))


def test_lattice_mask_dataset_initialization_preserves_physical_convention():
    numeric_mask = np.array(
        [[[1, 0], [1, 1]], [[0, 1], [1, 0]]],
        dtype=np.uint8,
    )

    dataset = GridFieldDataset(
        inputValue=InputGridField(shape=(2, 2, 2), mask=numeric_mask)
    )

    dataset_mask = dataset._helper_read_validity_mask()
    assert dataset_mask.dtype == np.bool_
    np.testing.assert_array_equal(dataset_mask, numeric_mask.astype(bool))


def test_lattice_mask_dataset_initialization_rejects_grid_shape_mismatch():
    with pytest.raises(ValueError, match="Field grid shape must match"):
        GridFieldDataset(
            inputValue=InputGridField(
                shape=(2, 2, 2),
                mask=np.ones((2, 2, 3), dtype=bool),
            )
        )
