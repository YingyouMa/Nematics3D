import numpy as np
import pytest

from nematics3d.datatypes import as_director_field


def test_as_director_field_keeps_zero_director_zero_when_normalizing():
    director = np.array(
        [
            [[[0.0, 0.0, 0.0], [3.0, 0.0, 4.0]]],
        ]
    )

    normalized = as_director_field(
        director,
        is_spatial_3d_required=True,
        log_mode="none",
    )

    np.testing.assert_allclose(normalized[0, 0, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(normalized[0, 0, 1], [0.6, 0.0, 0.8])
    assert np.isfinite(normalized).all()


def test_as_director_field_preserves_floating_dtype_without_normalization():
    director = np.ones((2, 2, 2, 3), dtype=np.float32)

    validated = as_director_field(
        director,
        is_spatial_3d_required=True,
        is_normalized=False,
        log_mode="none",
    )

    assert validated is director
    assert validated.dtype == np.float32


def test_as_director_field_rejects_nonreal_numpy_dtype():
    director = np.ones((2, 2, 2, 3), dtype=np.complex64)

    with np.testing.assert_raises(TypeError):
        as_director_field(director, log_mode="none")


def test_as_director_field_converts_integer_input_to_float():
    director = np.ones((2, 2, 1, 3), dtype=np.int16)

    validated = as_director_field(
        director,
        is_normalized=False,
        log_mode="none",
    )

    assert np.issubdtype(validated.dtype, np.floating)
    np.testing.assert_array_equal(validated, director)


def test_as_director_field_accepts_real_object_array():
    director = np.empty((1, 1, 1, 3), dtype=object)
    director[0, 0, 0] = [1, np.float32(2), 3.0]

    validated = as_director_field(
        director,
        is_normalized=False,
        log_mode="none",
    )

    np.testing.assert_array_equal(validated, [[[[1.0, 2.0, 3.0]]]])


def test_as_director_field_rejects_nonreal_object_array():
    director = np.array([[[[1.0, "bad", 0.0]]]], dtype=object)

    with pytest.raises(TypeError, match="real numbers"):
        as_director_field(director, log_mode="none")


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_as_director_field_rejects_nonfinite_values(invalid_value):
    director = np.ones((1, 1, 1, 3))
    director[0, 0, 0, 0] = invalid_value

    with pytest.raises(ValueError, match="finite"):
        as_director_field(director, log_mode="none")


def test_as_director_field_rejects_zero_when_disallowed():
    director = np.zeros((1, 1, 1, 3))

    with pytest.raises(ValueError, match="zero directors"):
        as_director_field(
            director,
            is_normalized=False,
            is_zero_allowed=False,
            log_mode="none",
        )
