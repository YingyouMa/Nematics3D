"""Box geometry helpers."""

import numpy as np

from ..datatypes import as_bool, as_number, as_points


__all__ = ["get_box_corners", "select_points_in_box"]


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


def select_points_in_box(
    points,
    corners,
    is_return_mask=False,
    *,
    atol=1e-9,
):
    """Select 3D points lying inside or on an oriented rectangular box.

    The box is defined by the first four corners using the same ordering as
    :func:`get_box_corners`: ``corners[0]`` is the reference corner and
    ``corners[1:4]`` terminate its three mutually perpendicular edges.
    Additional corners are accepted but are not needed for the calculation.

    Parameters
    ----------
    points : array-like
        Points with shape ``(n_points, 3)``. Empty input is allowed.
    corners : array-like or None
        Box corners with shape ``(n_corners, 3)`` and at least four rows. If
        None, all input points are selected.
    is_return_mask : bool, default=False
        If True, also return the boolean selection mask over the input points.
    atol : float, default=1e-9
        Absolute geometric tolerance used at the six box faces.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, numpy.ndarray]
        Selected floating-point points, optionally together with the boolean
        mask over the input collection.
    """
    is_return_mask = as_bool(is_return_mask, name="is_return_mask")
    atol = as_number(atol, name="atol", value_range=(0.0, np.inf))
    points = as_points(points, d=3, name="points", is_empty=True)

    if corners is None:
        mask = np.ones(len(points), dtype=bool)
        selected = points
        return (selected, mask) if is_return_mask else selected

    corners = as_points(corners, d=3, name="corners", min_num=4)
    edges = corners[1:4] - corners[0]
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths == 0.0):
        raise ValueError("The first three box edges must have positive length.")

    axes = edges / lengths[:, None]
    gram = axes @ axes.T
    if not np.allclose(gram, np.eye(3), rtol=0.0, atol=1e-8):
        raise ValueError(
            "The first three box edges must be mutually perpendicular."
        )

    local = (points - corners[0]) @ axes.T
    mask = np.all((local >= -atol) & (local <= lengths + atol), axis=1)
    selected = points[mask]

    return (selected, mask) if is_return_mask else selected
