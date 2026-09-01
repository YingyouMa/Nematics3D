"""Surface triangulation helpers based on spherical projection."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ..classes.result_base import ResultBase
from ..datatypes import as_points


@dataclass(slots=True, frozen=True, repr=False)
class TriangulationQuality(ResultBase):
    """Quality metrics for a spherical-projection surface triangulation."""

    __result_name__: ClassVar[str] = "Triangulation quality from spherical projection"

    __field_docs__: ClassVar[dict[str, str]] = {
        "flip_ratio": (
            "Fraction of triangles whose outward normal points inward. "
            "Non-zero only when the star-shape assumption is violated. "
            "Values above ~0.05 suggest the input is poorly suited for this method."
        ),
        "radial_cv": (
            "Coefficient of variation of point distances from the reference center. "
            "Zero for a perfect sphere; larger values indicate elongated or irregular "
            "shapes where projection distortion is higher."
        ),
        "n_flipped": "Number of inward-facing triangles.",
        "n_triangles": "Total number of triangles in the mesh.",
    }

    flip_ratio: float
    radial_cv: float
    n_flipped: int
    n_triangles: int


def check_triangulation_quality(
    points: np.ndarray,
    mesh: "pv.PolyData",
) -> TriangulationQuality:
    """Assess a spherical-projection surface triangulation.

    The input surface is assumed to be approximately star-shaped with respect
    to the centroid of ``points``. The same centroid convention is used by
    :func:`triangulate_surface_points`.

    Parameters
    ----------
    points : array-like, shape (N, 3)
        The original point cloud passed to ``triangulate_surface_points``.
    mesh : pyvista.PolyData
        The triangulated mesh returned by ``triangulate_surface_points``.

    Returns
    -------
    TriangulationQuality
        Dataclass result with ``flip_ratio``, ``radial_cv``, ``n_flipped``,
        and ``n_triangles``.
    """
    points = as_points(points, name="points", d=3, min_num=4)
    center = points.mean(axis=0)

    faces = np.asarray(mesh.faces, dtype=int).reshape(-1, 4)[:, 1:]

    v0, v1, v2 = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    tri_centers = (v0 + v1 + v2) / 3
    normals = np.cross(v1 - v0, v2 - v0)
    outward = tri_centers - center
    n_flipped = int(np.sum(np.einsum("ij,ij->i", normals, outward) < 0))
    n_triangles = len(faces)

    radii = np.linalg.norm(points - center, axis=1)
    mean_radius = radii.mean()
    radial_cv = float(radii.std() / mean_radius) if mean_radius > 1e-14 else 0.0

    return TriangulationQuality(
        flip_ratio=n_flipped / n_triangles,
        radial_cv=radial_cv,
        n_flipped=n_flipped,
        n_triangles=n_triangles,
    )


def triangulate_surface_points(points: np.ndarray) -> "pv.PolyData":
    """Build a closed triangular surface mesh from sampled surface points.

    The input points are radially projected onto a unit sphere centered at the
    point-cloud centroid. A convex hull of the projected points determines the
    triangle connectivity, which is then transferred back to the original
    coordinates. The returned mesh therefore uses exactly the input points as
    its vertices.

    This method assumes that the sampled surface is approximately star-shaped
    with respect to the centroid, meaning that each ray from the centroid
    intersects the surface once. Strongly non-star-shaped surfaces are not
    supported by this reconstruction method.

    A known degenerate case occurs when the centroid coincides with one of the
    sampled surface points. The corresponding radial direction is then
    undefined. This case is detected explicitly but is not repaired
    automatically; a systematic fallback should be designed separately.

    Parameters
    ----------
    points : array-like, shape (N, 3)
        Point cloud lying on the target closed surface.

    Returns
    -------
    pyvista.PolyData
        Triangulated surface whose vertices are exactly ``points``.
    """
    # Defer the optional heavy visualization dependency until mesh construction.
    import pyvista as pv

    points = as_points(points, name="points", d=3, min_num=4)
    center = points.mean(axis=0)

    directions = points - center
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norms < 1e-14):
        raise ValueError(
            "A surface point coincides with the point-cloud centroid, so the "
            "radial projection is undefined."
        )

    sphere_points = directions / norms

    try:
        hull = ConvexHull(sphere_points)
    except QhullError as exc:
        raise ValueError(
            "ConvexHull failed on the spherically projected points. The "
            "surface may be degenerate or too few points were supplied."
        ) from exc

    faces = hull.simplices
    face_array = np.empty((len(faces), 4), dtype=faces.dtype)
    face_array[:, 0] = 3
    face_array[:, 1:] = faces

    return pv.PolyData(points, face_array.ravel())
