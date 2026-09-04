"""Legacy director-color immersion retained before the OKLab replacement."""

from typing import List, Tuple

import numpy as np

from nematics3d.datatypes import as_director_field, nField


def n_color_immerse_legacy(n: nField) -> List[Tuple]:
    """Return colors from the original hand-tuned Boy-surface mapping."""
    n = as_director_field(n, name="n", is_normalized=True)

    rgb = np.zeros((*(np.shape(n)[:-1]), 3))
    x = n[..., 0]
    y = n[..., 1]
    z = n[..., 2]
    x2 = x**2
    y2 = y**2
    z2 = z**2

    rgb[..., 0] = (
        (2 * x2 - y2 - z2)
        + 2 * y * z * (y2 - z2)
        + z * x * (x2 - z2)
        + x * y * (y2 - x2)
    )
    rgb[..., 1] = (y2 - z2) + z * x * (z2 - x2) + x * y * (y2 - x2)
    rgb[..., 2] = (x + y + z) * ((x + y + z) ** 3 + 4 * (y - x) * (z - y) * (x - z))

    rgb[..., 0] = rgb[..., 0] / 2
    rgb[..., 1] = rgb[..., 1] * 7 / 8
    rgb[..., 2] = rgb[..., 2] / 8

    transform = np.array(
        [
            [1.01667, -0.3, -0.48333],
            [-1.01667, -1.5, -1.31667],
            [-0.18333, 0.3, 1.31667],
        ]
    )
    result = np.einsum("...i, ji -> ...j", rgb, transform)
    result = result / np.array([2.1, 4.2, 2.0]) + np.array([0.45, 0.51, 0.23])
    return [tuple(color) for color in result]
