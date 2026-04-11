"""
Lightweight Q-tensor defect analysis.

This file is a standalone extraction of the small numerical path needed to:

- accept a Q tensor field,
- diagonalize Q into scalar order parameter S and director n,
- detect defect points from n,
- group defect points into disclination lines,
- report how many defects and lines were found.

It intentionally avoids the full Nematics3D object/visualization stack. Only NumPy is
required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


DEFECT_NEIGHBOR = np.array(
    [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, -0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.5, 0.0, -0.5),
        (-0.5, 0.5, 0.0),
        (-0.5, -0.5, 0.0),
        (-0.5, 0.0, 0.5),
        (-0.5, 0.0, -0.5),
    ],
    dtype=float,
)


@dataclass(frozen=True)
class QAnalysisResult:
    """Container returned by analyze_Q()."""

    S: np.ndarray
    n: np.ndarray
    defect_indices: np.ndarray
    defect_count: int
    defect_lines: list[np.ndarray]
    line_count: int


def as_dimension_info(value: bool | float | Sequence[float]) -> np.ndarray:
    """Convert scalar or 3-vector dimension metadata into a length-3 array."""
    if isinstance(value, (bool, np.bool_)):
        return np.array([value, value, value])
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.array([value, value, value])
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 3:
        return np.asarray(value)
    raise ValueError("Expected a scalar or a length-3 sequence.")


def as_QField9(Q: np.ndarray) -> np.ndarray:
    """
    Convert Q to full (..., 3, 3) representation.

    Accepted input shapes:
    - (..., 5): [Qxx, Qxy, Qxz, Qyy, Qyz], with Qzz = -Qxx - Qyy.
    - (..., 3, 3): full tensor form.
    """
    Q = np.asarray(Q)
    if not np.issubdtype(Q.dtype, np.floating):
        Q = Q.astype(float)

    if Q.ndim >= 2 and Q.shape[-1] == 5:
        Q9 = np.zeros((*Q.shape[:-1], 3, 3), dtype=Q.dtype)
        Q9[..., 0, 0] = Q[..., 0]
        Q9[..., 0, 1] = Q[..., 1]
        Q9[..., 0, 2] = Q[..., 2]
        Q9[..., 1, 0] = Q[..., 1]
        Q9[..., 1, 1] = Q[..., 3]
        Q9[..., 1, 2] = Q[..., 4]
        Q9[..., 2, 0] = Q[..., 2]
        Q9[..., 2, 1] = Q[..., 4]
        Q9[..., 2, 2] = -Q[..., 0] - Q[..., 3]
        return Q9

    if Q.ndim >= 3 and Q.shape[-2:] == (3, 3):
        return Q

    raise ValueError(f"Q must have shape (..., 5) or (..., 3, 3), got {Q.shape}.")


def build_Q_from_nS(n: np.ndarray, S: np.ndarray | float = 1.0) -> np.ndarray:
    """Build a uniaxial Q tensor from director n and scalar order parameter S."""
    n = np.asarray(n, dtype=float)
    if n.shape[-1] != 3:
        raise ValueError(f"n must end with shape (..., 3), got {n.shape}.")

    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("n contains zero-length vectors.")

    n = n / norm
    Q = np.einsum("...i,...j->...ij", n, n) - np.eye(3) / 3
    return np.asarray(S, dtype=float)[..., np.newaxis, np.newaxis] * Q


def diagonalize_Q(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize Q and return (S, n).

    This uses the same fast invariant-based path as the full package instead of
    calling np.linalg.eigh on every 3x3 tensor. S is 1.5 times the largest
    eigenvalue, matching the convention used by the full package. n is the
    eigenvector associated with that largest eigenvalue.
    """
    Q = as_QField9(Q)

    p = 0.5 * np.einsum("...ab,...ba->...", Q, Q)
    q = np.linalg.det(Q)
    r = 2 * np.sqrt(p / 3)

    cos_arg = 4 * q / r**3
    cos_arg = np.clip(cos_arg, -1.0, 1.0)
    lambda_max = r * np.cos((1 / 3) * np.arccos(cos_arg))

    n_raw = np.array(
        [
            Q[..., 0, 2] * (Q[..., 1, 1] - lambda_max) - Q[..., 0, 1] * Q[..., 1, 2],
            Q[..., 1, 2] * (Q[..., 0, 0] - lambda_max) - Q[..., 0, 1] * Q[..., 0, 2],
            Q[..., 0, 1] ** 2
            - (Q[..., 0, 0] - lambda_max) * (Q[..., 1, 1] - lambda_max),
        ]
    )
    n = np.moveaxis(n_raw / np.linalg.norm(n_raw, axis=0), 0, -1)
    S = 1.5 * lambda_max
    return S, n


def align_stack(stack: np.ndarray) -> np.ndarray:
    """Align a stack of directors under n == -n nematic symmetry."""
    dots = np.einsum("...i,...i->...", stack[:-1], stack[1:])
    flips = np.ones(stack.shape[:-1])
    flips[1:] = np.where(dots < 0, -1, 1)
    acc_flips = np.cumprod(flips, axis=0)
    return stack * acc_flips[..., np.newaxis]


def add_periodic_boundary(data: np.ndarray, is_boundary_periodic=0) -> np.ndarray:
    """Append one copied boundary slice for each periodic spatial axis."""
    is_boundary_periodic = as_dimension_info(is_boundary_periodic).astype(bool)
    if not np.any(is_boundary_periodic):
        return data

    Nx, Ny, Nz, *rest_shape = data.shape
    output = np.empty(
        (
            Nx + int(is_boundary_periodic[0]),
            Ny + int(is_boundary_periodic[1]),
            Nz + int(is_boundary_periodic[2]),
            *rest_shape,
        ),
        dtype=data.dtype,
    )
    output[:Nx, :Ny, :Nz] = data

    if is_boundary_periodic[0]:
        output[Nx] = output[0]
    if is_boundary_periodic[1]:
        output[:, Ny] = output[:, 0]
    if is_boundary_periodic[2]:
        output[:, :, Nz] = output[:, :, 0]

    return output


def detect_defects_xyplane(n: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Detect defects on xy plaquettes of a field whose normal axis is z."""
    a_orig = n[:-1, :-1]
    b_orig = n[1:, :-1]
    c_orig = n[1:, 1:]
    d_orig = n[:-1, 1:]

    a, _b, _c, d = align_stack(np.stack([a_orig, b_orig, c_orig, d_orig], axis=0))
    test = np.einsum("...i,...i->...", a, d)

    coords = np.array(np.where(test < threshold)).T.astype(float)
    if len(coords) == 0:
        return np.empty((0, 3), dtype=float)

    coords[:, [0, 1]] += 0.5
    return coords


def detect_defects(
    n: np.ndarray,
    threshold: float = 0.0,
    is_boundary_periodic=0,
    planes=1,
) -> np.ndarray:
    """
    Detect defect points in a 3D director field.

    Returned coordinates are lattice-index positions with one integer component
    and two half-integer components.
    """
    n = np.asarray(n, dtype=float)
    if n.ndim != 4 or n.shape[-1] != 3:
        raise ValueError(f"n must have shape (Nx, Ny, Nz, 3), got {n.shape}.")

    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("n contains zero-length vectors.")
    n_origin = n / norm

    is_boundary_periodic = as_dimension_info(is_boundary_periodic).astype(bool)
    planes = as_dimension_info(planes).astype(bool)
    n_periodic = add_periodic_boundary(n_origin, is_boundary_periodic)
    defect_indices = np.empty((0, 3), dtype=float)

    axis_permutations = {
        0: (2, 1, 0),
        1: (0, 2, 1),
        2: (0, 1, 2),
    }

    for axis in range(3):
        if not planes[axis]:
            continue

        perm = axis_permutations[axis]
        n_rot = np.moveaxis(n_periodic, [0, 1, 2], perm)
        coords = detect_defects_xyplane(n_rot, threshold)
        if len(coords) == 0:
            continue

        coords = coords[:, np.argsort(perm)]
        defect_indices = np.vstack((defect_indices, coords))

    for i, periodic in enumerate(is_boundary_periodic):
        if periodic and len(defect_indices) > 0:
            defect_indices[:, i] %= n_origin.shape[i]

    if len(defect_indices) == 0:
        return defect_indices

    defect_indices, _ = np.unique(defect_indices, axis=0, return_index=True)
    return defect_indices


def make_hash_table(items: Iterable[Iterable[float]]) -> dict[tuple[float, ...], int]:
    """Map vector-like rows to their row index."""
    return {tuple(item): idx for idx, item in enumerate(items)}


def generate_mirror_points(point: np.ndarray, box_size_periodic=np.inf) -> np.ndarray:
    """Generate periodic mirror images near periodic boundaries."""
    box_size = as_dimension_info(box_size_periodic).astype(float)
    point = np.asarray(point, dtype=float)
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

    return np.array(np.meshgrid(*mirrors, indexing="ij")).reshape(3, -1).T


def possible_defect_neighbors(defect_index: Sequence[float], box_size_periodic=np.inf):
    """Return all neighboring defect-index positions for line connectivity."""
    defect_index = np.asarray(defect_index, dtype=float)
    if defect_index.shape != (3,):
        raise ValueError(
            f"defect_index must have shape (3,), got {defect_index.shape}."
        )

    box_size_periodic = as_dimension_info(box_size_periodic).astype(float)
    neighbor = DEFECT_NEIGHBOR.copy()

    layer_index = np.where(np.isclose(defect_index % 1, 0))[0]
    if len(layer_index) != 1:
        raise ValueError(
            "Each defect index must have exactly one integer coordinate. "
            f"Got {defect_index}."
        )
    layer_index = layer_index[0]

    if layer_index != 0:
        neighbor[:, (0, layer_index)] = neighbor[:, (layer_index, 0)]

    result = np.tile(defect_index, (len(neighbor), 1)) + neighbor

    periodic_mask = box_size_periodic != np.inf
    if np.any(periodic_mask):
        coord_in_periodic = defect_index[periodic_mask]
        size_in_periodic = box_size_periodic[periodic_mask]
        near_boundary = np.min(coord_in_periodic) <= 1 or np.any(
            coord_in_periodic >= size_in_periodic - 2
        )
        if near_boundary:
            result = np.vstack(
                [
                    generate_mirror_points(point, box_size_periodic=box_size_periodic)
                    for point in result
                ]
            )

    return result


def unwrap_trajectory(points: np.ndarray, box_size_periodic=np.inf) -> np.ndarray:
    """Unwrap a line across periodic boundaries using the minimum-image convention."""
    points = np.asarray(points, dtype=float)
    if len(points) <= 1:
        return points.copy()

    box_size_periodic = as_dimension_info(box_size_periodic).astype(float)
    deltas = np.diff(points, axis=0)
    mask_periodic = np.isfinite(box_size_periodic)
    deltas[:, mask_periodic] -= (
        np.round(deltas[:, mask_periodic] / box_size_periodic[mask_periodic])
        * box_size_periodic[mask_periodic]
    )
    return np.vstack([points[0], points[0] + np.cumsum(deltas, axis=0)])


def classify_defect_lines(
    defect_indices: np.ndarray,
    box_size_periodic=np.inf,
    is_unwrap: bool = True,
) -> list[np.ndarray]:
    """Group defect points into connected disclination lines."""
    defect_indices = np.asarray(defect_indices, dtype=float)
    if defect_indices.size == 0:
        return []
    if defect_indices.ndim != 2 or defect_indices.shape[1] != 3:
        raise ValueError(
            f"defect_indices must have shape (N, 3), got {defect_indices.shape}."
        )

    box_size_periodic = as_dimension_info(box_size_periodic).astype(float)
    index_by_point = make_hash_table(defect_indices)
    adjacency = {idx: set() for idx in range(len(defect_indices))}

    for idx1, defect in enumerate(defect_indices):
        neighbors = possible_defect_neighbors(
            defect, box_size_periodic=box_size_periodic
        )
        for neighbor in neighbors:
            idx2 = index_by_point.get(tuple(neighbor))
            if idx2 is not None:
                adjacency[idx1].add(idx2)
                adjacency[idx2].add(idx1)

    lines = []
    visited = set()
    for start in range(len(defect_indices)):
        if start in visited:
            continue

        stack = [start]
        component = []
        visited.add(start)

        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        line = defect_indices[component]
        if is_unwrap:
            line = unwrap_trajectory(line, box_size_periodic=box_size_periodic)
        lines.append(line)

    lines.sort(key=len, reverse=True)
    return lines


def analyze_Q(
    Q: np.ndarray,
    threshold: float = 0.0,
    is_boundary_periodic=0,
    planes=1,
    box_size_periodic=None,
) -> QAnalysisResult:
    """
    Analyze Q and return n/S plus defect and line counts.

    Parameters
    ----------
    Q
        Q tensor field with shape (Nx, Ny, Nz, 5) or (Nx, Ny, Nz, 3, 3).
    threshold
        Defect detection threshold for the final director dot product around a
        plaquette. The package default is 0.
    is_boundary_periodic
        Boolean or length-3 boolean flags for periodic boundaries during defect
        detection.
    planes
        Boolean or length-3 boolean flags selecting plaquette normals to analyze.
    box_size_periodic
        Periodic box size for line classification. Defaults to the Q grid size
        on periodic axes and np.inf on non-periodic axes.
    """
    Q9 = as_QField9(Q)
    if Q9.ndim != 5:
        raise ValueError(f"Q must be a 3D tensor field, got shape {Q9.shape}.")

    S, n = diagonalize_Q(Q9)
    defect_indices = detect_defects(
        n,
        threshold=threshold,
        is_boundary_periodic=is_boundary_periodic,
        planes=planes,
    )

    if box_size_periodic is None:
        periodic_flag = as_dimension_info(is_boundary_periodic).astype(bool)
        box_size_periodic = np.where(periodic_flag, Q9.shape[:3], np.inf)

    defect_lines = classify_defect_lines(
        defect_indices, box_size_periodic=box_size_periodic
    )

    return QAnalysisResult(
        S=S,
        n=n,
        defect_indices=defect_indices,
        defect_count=len(defect_indices),
        defect_lines=defect_lines,
        line_count=len(defect_lines),
    )
