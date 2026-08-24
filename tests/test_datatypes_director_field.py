import numpy as np

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
