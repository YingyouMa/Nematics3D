import numpy as np

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
