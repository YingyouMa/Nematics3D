import numpy as np

from nematics3d.classes.visual.color import director_color_pareto_oklab_043
from nematics3d.field import n_color_immerse


def test_n_color_immerse_uses_selected_oklab_mapping():
    directors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    directors /= np.linalg.norm(directors, axis=1, keepdims=True)

    actual = np.asarray(n_color_immerse(directors))
    expected = director_color_pareto_oklab_043(directors)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        actual, n_color_immerse(-directors), rtol=0.0, atol=1e-14
    )
    assert np.all((actual >= 0.0) & (actual <= 1.0))
