import numpy as np
import pytest

from nematics3d.analysis.sampling import sample_van_der_corput


def test_sample_van_der_corput_standard_sequence():
    result = sample_van_der_corput(7)

    np.testing.assert_array_equal(
        result,
        np.array([0.0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375]),
    )


def test_sample_van_der_corput_including_one():
    result = sample_van_der_corput(7, is_include_one=True)

    np.testing.assert_array_equal(
        result,
        np.array([0.0, 1.0, 0.5, 0.25, 0.75, 0.125, 0.625]),
    )


def test_sample_van_der_corput_including_one_short_sequences():
    np.testing.assert_array_equal(
        sample_van_der_corput(0, is_include_one=True), np.array([])
    )
    np.testing.assert_array_equal(
        sample_van_der_corput(1, is_include_one=True), np.array([0.0])
    )
    np.testing.assert_array_equal(
        sample_van_der_corput(2, is_include_one=True), np.array([0.0, 1.0])
    )


@pytest.mark.parametrize("is_include_one", [False, True])
def test_sample_van_der_corput_output_range_and_dtype(is_include_one):
    result = sample_van_der_corput(257, is_include_one=is_include_one)

    assert result.dtype == np.dtype(float)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0 if is_include_one else result < 1.0)
    assert np.count_nonzero(result == 1.0) == int(is_include_one)


@pytest.mark.parametrize("num", [-1, 1.5, True])
def test_sample_van_der_corput_rejects_invalid_num(num):
    with pytest.raises((TypeError, ValueError)):
        sample_van_der_corput(num)


@pytest.mark.parametrize("is_include_one", [2, -1, "yes", None])
def test_sample_van_der_corput_rejects_invalid_include_one(is_include_one):
    with pytest.raises((TypeError, ValueError)):
        sample_van_der_corput(4, is_include_one=is_include_one)
