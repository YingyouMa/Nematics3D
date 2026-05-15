import numpy as np

from nematics3d.datatypes import check_Sn


def test_check_sn_keeps_zero_director_zero_when_normalizing():
    director = np.array(
        [
            [[[0.0, 0.0, 0.0], [3.0, 0.0, 4.0]]],
        ]
    )

    normalized = check_Sn(director, "n")

    np.testing.assert_allclose(normalized[0, 0, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(normalized[0, 0, 1], [0.6, 0.0, 0.8])
    assert np.isfinite(normalized).all()
