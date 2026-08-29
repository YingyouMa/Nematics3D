"""Axis-aligned box geometry helpers."""

import numpy as np

from ..datatypes import as_number


def get_box_corners(length_x, length_y, length_z) -> np.ndarray:
    """Return the eight corners of an axis-aligned box from the origin.

    The box spans from ``(0, 0, 0)`` to
    ``(length_x, length_y, length_z)``. Lengths must be finite,
    non-negative real numbers. Zero is allowed so lower-dimensional
    degenerate boxes can represent one-cell lattice dimensions.

    The returned corner order is fixed and is used by downstream box-edge and
    face topology:

    ``(0,0,0)``, ``(x,0,0)``, ``(0,y,0)``, ``(0,0,z)``,
    ``(x,y,0)``, ``(x,0,z)``, ``(0,y,z)``, ``(x,y,z)``.
    """

    length_x = as_number(
        length_x,
        name="length_x",
        value_range=(0.0, np.inf),
    )
    length_y = as_number(
        length_y,
        name="length_y",
        value_range=(0.0, np.inf),
    )
    length_z = as_number(
        length_z,
        name="length_z",
        value_range=(0.0, np.inf),
    )

    return np.array(
        [
            [0.0, 0.0, 0.0],
            [length_x, 0.0, 0.0],
            [0.0, length_y, 0.0],
            [0.0, 0.0, length_z],
            [length_x, length_y, 0.0],
            [length_x, 0.0, length_z],
            [0.0, length_y, length_z],
            [length_x, length_y, length_z],
        ],
        dtype=float,
    )
