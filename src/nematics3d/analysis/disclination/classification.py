from time import perf_counter
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from ...datatypes import (
    DefectIndex,
    DimensionFlagInput,
    Vect,
    as_defect_index,
    as_dimension_info,
)
from ...grid import (
    GRID_TRANSFORM_IDENTITY,
    as_grid_transform,
    unwrap_trajectory,
)
from ...logging_decorator import logging_and_warning_decorator

if TYPE_CHECKING:
    from ...classes.disclination_line import DisclinationLine

_DEFECT_NEIGHBORS_2X = np.array(
    [
        [2, 0, 0],
        [-2, 0, 0],
        [1, 1, 0],
        [1, -1, 0],
        [1, 0, 1],
        [1, 0, -1],
        [-1, 1, 0],
        [-1, -1, 0],
        [-1, 0, 1],
        [-1, 0, -1],
    ],
    dtype=np.int64,
)

_DEFECT_NEIGHBORS_2X_BY_LAYER = np.empty((3, 10, 3), dtype=np.int64)
_DEFECT_NEIGHBORS_2X_BY_LAYER[0] = _DEFECT_NEIGHBORS_2X
_DEFECT_NEIGHBORS_2X_BY_LAYER[1] = _DEFECT_NEIGHBORS_2X[:, [1, 0, 2]]
_DEFECT_NEIGHBORS_2X_BY_LAYER[2] = _DEFECT_NEIGHBORS_2X[:, [2, 1, 0]]


def _canonicalize_defect_indices_2x(defect_indices, box_size_periodic):
    """Convert exact integer/half-integer defect coordinates to integers."""
    values = as_defect_index(
        defect_indices,
        name="defect indices to classify",
        tolerance=5e-10,
    )
    doubled = np.rint(2.0 * values).astype(np.int64, copy=False)

    periodic = np.isfinite(box_size_periodic)
    if np.any(periodic):
        periods_2x = np.rint(2.0 * box_size_periodic[periodic]).astype(np.int64)
        if np.any(periods_2x <= 0):
            raise ValueError("Finite periodic box sizes must be positive.")
        doubled[:, periodic] %= periods_2x

    return doubled


def _build_defect_edges(points_2x, box_size_periodic):
    """Build undirected defect-neighbor edges in vectorized batches."""
    node_count = len(points_2x)
    if node_count == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    periodic = np.isfinite(box_size_periodic)
    periods_2x = np.zeros(3, dtype=np.int64)
    periods_2x[periodic] = np.rint(2.0 * box_size_periodic[periodic]).astype(np.int64)

    mins = np.empty(3, dtype=np.int64)
    maxs = np.empty(3, dtype=np.int64)
    for axis in range(3):
        if periodic[axis]:
            mins[axis] = 0
            maxs[axis] = periods_2x[axis] - 1
        else:
            mins[axis] = int(points_2x[:, axis].min())
            maxs[axis] = int(points_2x[:, axis].max())

    extents = maxs - mins + 1
    key_space = int(extents[0]) * int(extents[1]) * int(extents[2])
    if key_space > np.iinfo(np.int64).max:
        raise ValueError("Defect-coordinate extent is too large for packed keys.")

    stride_y = int(extents[0])
    stride_z = int(extents[0]) * int(extents[1])
    shifted = points_2x - mins
    keys = (shifted[:, 0] + shifted[:, 1] * stride_y + shifted[:, 2] * stride_z).astype(
        np.int64, copy=False
    )

    if np.unique(keys).size != node_count:
        raise ValueError(
            "Duplicate defect indices remain after canonical periodic wrapping."
        )

    order = np.argsort(keys)
    sorted_keys = keys[order]
    layer_axes = np.argmax((points_2x & 1) == 0, axis=1)
    edge_sources = []
    edge_targets = []

    for layer_axis in range(3):
        source = np.flatnonzero(layer_axes == layer_axis)
        if source.size == 0:
            continue

        candidates = (
            points_2x[source, None, :]
            + _DEFECT_NEIGHBORS_2X_BY_LAYER[layer_axis][None, :, :]
        )
        for axis in range(3):
            if periodic[axis]:
                candidates[..., axis] %= periods_2x[axis]

        shifted_candidates = candidates - mins
        is_valid = np.ones(shifted_candidates.shape[:2], dtype=bool)
        for axis in range(3):
            is_valid &= (shifted_candidates[..., axis] >= 0) & (
                shifted_candidates[..., axis] < extents[axis]
            )

        candidate_keys = (
            shifted_candidates[..., 0]
            + shifted_candidates[..., 1] * stride_y
            + shifted_candidates[..., 2] * stride_z
        ).reshape(-1)
        positions = np.searchsorted(sorted_keys, candidate_keys)
        is_found = positions < node_count
        clipped_positions = np.minimum(positions, node_count - 1)
        is_found &= sorted_keys[clipped_positions] == candidate_keys

        neighbors = np.full(candidate_keys.shape, -1, dtype=np.int64)
        neighbors[is_found] = order[positions[is_found]]
        neighbors = neighbors.reshape(source.size, 10)
        source_grid = np.broadcast_to(source[:, None], neighbors.shape)
        is_kept = is_valid & (neighbors > source_grid)
        edge_sources.append(source_grid[is_kept])
        edge_targets.append(neighbors[is_kept])

    if not edge_sources:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    edge_u = np.concatenate(edge_sources)
    edge_v = np.concatenate(edge_targets)
    if np.any(periodic):
        edge_keys = edge_u * node_count + edge_v
        _, unique_indices = np.unique(edge_keys, return_index=True)
        edge_u = edge_u[unique_indices]
        edge_v = edge_v[unique_indices]

    return edge_u, edge_v


def _build_defect_adjacency(node_count, edge_u, edge_v):
    """Build adjacency entries containing both neighbor and edge ID."""
    adjacency = [[] for _ in range(node_count)]
    for edge_id, (u, v) in enumerate(zip(edge_u.tolist(), edge_v.tolist())):
        adjacency[u].append((v, edge_id))
        adjacency[v].append((u, edge_id))
    return adjacency


def _extract_defect_trails(adjacency, edge_count):
    """Consume every graph edge into ordered maximal trails."""
    if edge_count == 0:
        return []

    remaining_degree = np.fromiter(map(len, adjacency), dtype=np.int64)
    active_nodes = set(np.flatnonzero(remaining_degree).tolist())
    odd_nodes = {node for node in active_nodes if remaining_degree[node] % 2 == 1}
    used_edges = np.zeros(edge_count, dtype=bool)
    adjacency_cursor = np.zeros(len(adjacency), dtype=np.int64)
    remaining_edge_count = edge_count
    trails = []

    def update_node(node):
        if remaining_degree[node] == 0:
            active_nodes.discard(node)
            odd_nodes.discard(node)
        elif remaining_degree[node] % 2 == 1:
            odd_nodes.add(node)
        else:
            odd_nodes.discard(node)

    while remaining_edge_count:
        start = next(iter(odd_nodes or active_nodes))
        trail = [start]
        current = start

        while remaining_degree[current]:
            cursor = adjacency_cursor[current]
            entries = adjacency[current]
            while cursor < len(entries) and used_edges[entries[cursor][1]]:
                cursor += 1
            adjacency_cursor[current] = cursor + 1

            neighbor, edge_id = entries[cursor]
            used_edges[edge_id] = True
            remaining_edge_count -= 1
            remaining_degree[current] -= 1
            remaining_degree[neighbor] -= 1
            update_node(current)
            update_node(neighbor)

            trail.append(neighbor)
            current = neighbor

        trails.append(trail)

    return trails


@logging_and_warning_decorator()
def defect_classify_into_lines(
    defect_indices: DefectIndex,
    box_size_periodic: DimensionFlagInput = np.inf,
    grid_offset: Optional[Vect(3)] = None,
    grid_transform=GRID_TRANSFORM_IDENTITY,
    logger=None,
) -> List["DisclinationLine"]:
    """Group half-grid defect points into ordered disclination-line trails."""
    # Local import avoids the module cycle: DisclinationLine depends on helpers
    # defined in this module through higher-level line operations.
    from ...classes.disclination_line import DisclinationLine

    box_size_periodic = as_dimension_info(box_size_periodic)
    grid_transform = as_grid_transform(grid_transform)
    defect_indices = np.asarray(defect_indices, dtype=np.float64)
    if defect_indices.ndim != 2 or defect_indices.shape[1] != 3:
        raise ValueError(
            "'defect_indices' must have shape (N, 3). " f"Got {defect_indices.shape}."
        )
    if not np.isfinite(defect_indices).all():
        raise ValueError("'defect_indices' must contain only finite values.")
    if len(defect_indices) == 0:
        return []

    start = perf_counter()
    points_2x = _canonicalize_defect_indices_2x(
        defect_indices,
        box_size_periodic,
    )
    edge_u, edge_v = _build_defect_edges(points_2x, box_size_periodic)
    adjacency = _build_defect_adjacency(len(defect_indices), edge_u, edge_v)
    trails = _extract_defect_trails(adjacency, len(edge_u))
    logger.debug(
        f"Classified {len(defect_indices):,} defects through {len(edge_u):,} "
        f"edges into {len(trails):,} trails in "
        f"{perf_counter() - start:.3f} seconds."
    )

    paths = [
        unwrap_trajectory(
            defect_indices[np.asarray(trail, dtype=np.int64)],
            box_size_periodic=box_size_periodic,
        )
        for trail in trails
    ]
    return [
        DisclinationLine(
            defect_indices=path,
            box_size_periodic_index=box_size_periodic,
            grid_offset=grid_offset,
            grid_transform=grid_transform,
            is_sorted=True,
        )
        for path in paths
    ]
