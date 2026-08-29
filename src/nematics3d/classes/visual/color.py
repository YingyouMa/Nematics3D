"""Color helpers for visualization."""

import numpy as np


def blue_red_in_white_bg() -> np.ndarray:
    """Return the normalized blue-green-red colormap used for line colors."""
    colormap = np.zeros((511, 3))
    colormap[:256, 1] = np.arange(256)
    colormap[:256, 2] = 255 - np.arange(256)
    colormap[255:, 1] = 255 - np.arange(256)
    colormap[255:, 0] = np.arange(256)
    colormap = colormap / 255
    colormap = colormap / np.linalg.norm(colormap, axis=-1, keepdims=True)
    return colormap
