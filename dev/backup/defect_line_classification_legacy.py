"""Legacy defect-line classifier and its private graph implementation."""

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import numpy as np

from nematics3d.datatypes import (
    DefectIndex,
    DimensionFlagInput,
    Vect,
    as_dimension_info,
)
from nematics3d.analysis.disclination import defect_neighbor_possible_get
from nematics3d.general import make_hash_table, search_in_reservoir
from nematics3d.grid import (
    GRID_TRANSFORM_IDENTITY,
    as_grid_transform,
    unwrap_trajectory,
)
from nematics3d.logging_decorator import logging_and_warning_decorator

if TYPE_CHECKING:
    from nematics3d.classes.disclination_line import DisclinationLine


class Graph:
    """Legacy undirected adjacency-set graph used by the classifier."""

    def __init__(self):
        self.graph: Dict[Any, Set[Any]] = defaultdict(set)

    def add_edge(self, u: Any, v: Any) -> None:
        self.graph[u].add(v)
        self.graph[v].add(u)

    def remove_edge(self, u: Any, v: Any) -> None:
        self.graph[u].discard(v)
        self.graph[v].discard(u)

    def find_start_node(self) -> Optional[Any]:
        start_node = next(
            (node for node in self.graph if len(self.graph[node]) % 2 == 1),
            None,
        )
        if start_node is None:
            start_node = next(
                (node for node in self.graph if len(self.graph[node]) > 0),
                None,
            )
        return start_node

    def hierholzer_algorithm(self, start_node: Any) -> List[Any]:
        path = [start_node]
        while True:
            u = path[-1]
            if not self.graph[u]:
                break
            v = next(iter(self.graph[u]))
            path.append(v)
            self.remove_edge(u, v)
        return path

    def find_path(self) -> List[List[Any]]:
        paths = []
        while any(self.graph.values()):
            start_node = self.find_start_node()
            if start_node is None:
                return paths
            paths.append(self.hierholzer_algorithm(start_node))
        return paths


@logging_and_warning_decorator()
def defect_classify_into_lines(
    defect_indices: DefectIndex,
    box_size_periodic: DimensionFlagInput = np.inf,
    grid_offset: Optional[Vect(3)] = None,
    grid_transform=GRID_TRANSFORM_IDENTITY,
    logger=None,
) -> List["DisclinationLine"]:
    """Legacy hash-table and adjacency-set line classifier."""
    from nematics3d.classes.disclination_line import DisclinationLine

    box_size_periodic = as_dimension_info(box_size_periodic)
    grid_transform = as_grid_transform(grid_transform)

    logger.debug(
        "Start line classfication.\n" f"box_size_periodic: {box_size_periodic}."
    )

    defect_indices_hash = make_hash_table(defect_indices)
    graph = Graph()

    for idx1, defect in enumerate(defect_indices):
        neighbors = defect_neighbor_possible_get(
            defect,
            box_size_periodic=box_size_periodic,
        )
        matches = search_in_reservoir(
            neighbors,
            defect_indices_hash,
            is_reservoir_hash=True,
        )
        matches = matches[~np.isnan(matches)].astype(int)
        for idx2 in matches:
            graph.add_edge(idx1, idx2)

    paths = graph.find_path()
    paths = [
        unwrap_trajectory(
            defect_indices[path],
            box_size_periodic=box_size_periodic,
        )
        for path in paths
    ]
    logger.debug("Done!")

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
