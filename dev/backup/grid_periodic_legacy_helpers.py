"""Legacy periodic-grid helpers retained for possible future reuse.

These functions are intentionally kept outside the public ``nematics3d``
package. They were removed from the active API during public-beta cleanup
because no active internal callers were found, but the implementations may be
useful as references for future periodic-boundary workflows.
"""

from itertools import product

import numpy as np

from nematics3d.datatypes import (
    BoxSizePeriodic,
    Vect,
    as_box_size_periodic,
    as_vector,
)


def generate_mirror_point_periodic_boundary(
    point: Vect(3),
    box_size_periodic: BoxSizePeriodic = np.inf,
    is_self: bool = True,
):
    """Generate nearby mirror images of a point across periodic boundaries."""
    box_size = as_box_size_periodic(
        box_size_periodic,
        name="box_size_periodic",
    )
    point = as_vector(
        point,
        name="The position of point which needs to find mirror image",
    )

    point = np.where(box_size == np.inf, point, point % box_size)

    mirrors = [[value] for value in point]
    for i, mirror in enumerate(mirrors):
        size = box_size[i]
        value = point[i]
        if size != np.inf:
            if -1 <= value <= 0:
                mirror.append(value + size)
            elif size - 1 <= value <= size:
                mirror.append(value - size)

    mirror_points = np.array(list(product(*mirrors)))

    if not is_self:
        mirror_points = mirror_points[1:]

    return mirror_points


def unfold_cluster(
    points: np.ndarray,
    box_size_periodic: BoxSizePeriodic = np.inf,
):
    """Unfold a periodic cluster into one continuous region."""
    points = np.asarray(points, dtype=float)
    box_size_periodic = as_box_size_periodic(
        box_size_periodic,
        name="box_size_periodic",
    )
    if np.all(np.isinf(box_size_periodic)):
        return points

    unfolded = points.copy()
    ref = points[0]

    for i in range(len(points)):
        for dim, size in enumerate(box_size_periodic):
            if size != np.inf:
                delta = points[i, dim] - ref[dim]
                if delta > size / 2:
                    unfolded[i, dim] -= size
                elif delta < -size / 2:
                    unfolded[i, dim] += size

    return unfolded
