"""Color helpers for visualization."""

import numpy as np


def blue_green_red_colors() -> np.ndarray:
    """Return L2-normalized RGB colors progressing from blue to green to red.

    The table contains 511 colors: 256 samples from blue to green followed by
    255 samples from green to red, with the shared green endpoint included only
    once. Each RGB vector is normalized to unit Euclidean norm so mixed colors
    remain visually strong on a white background.
    """
    t = np.linspace(0.0, 1.0, 256)

    blue_to_green = np.column_stack(
        (
            np.zeros_like(t),
            t,
            1.0 - t,
        )
    )
    green_to_red = np.column_stack(
        (
            t[1:],
            1.0 - t[1:],
            np.zeros_like(t[1:]),
        )
    )

    colors = np.vstack((blue_to_green, green_to_red))
    colors /= np.linalg.norm(colors, axis=1, keepdims=True)

    return colors
