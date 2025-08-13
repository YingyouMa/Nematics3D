from collections import defaultdict
from typing import Any, Dict, List, Optional, Set


class Graph:
    """
    A simple undirected graph implementation using adjacency sets,
    with support for adding/removing edges and finding Eulerian paths
    using Hierholzer's algorithm.
    """

    def __init__(self):
        """
        Initialize an empty undirected graph using a defaultdict of sets.
        """
        self.graph: Dict[Any, Set[Any]] = defaultdict(set)

    def add_edge(self, u: Any, v: Any) -> None:
        """
        Add an undirected edge between nodes u and v.

        Parameters
        ----------
        u : hashable
            One endpoint of the edge.
        v : hashable
            The other endpoint of the edge.
        """
        self.graph[u].add(v)
        self.graph[v].add(u)

    def remove_edge(self, u: Any, v: Any) -> None:
        """
        Remove an undirected edge between nodes u and v, if it exists.

        Parameters
        ----------
        u : hashable
            One endpoint of the edge.
        v : hashable
            The other endpoint of the edge.
        """
        self.graph[u].discard(v)
        self.graph[v].discard(u)

    def find_start_node(self) -> Optional[Any]:
        """
        Find a valid starting node for Hierholzer's algorithm.

        Returns
        -------
        start_node : hashable or None
            A node with odd degree (if any), or the first non-isolated node.
            Returns None if the graph has no edges.
        """
        # Prefer node with odd degree (for Eulerian path)
        start_node = next(
            (node for node in self.graph if len(self.graph[node]) % 2 == 1), None
        )
        if start_node is None:
            # Fallback: find any node with at least one neighbor
            start_node = next(
                (node for node in self.graph if len(self.graph[node]) > 0), None
            )
        return start_node

    def hierholzer_algorithm(self, start_node: Any) -> List[Any]:
        """
        Perform Hierholzer's algorithm from a starting node
        to find one Eulerian path component.

        Parameters
        ----------
        start_node : hashable
            The node to start the path from.

        Returns
        -------
        path : list
            A list of nodes representing the Eulerian path segment.
        """
        path = [start_node]

        while True:
            u = path[-1]
            if self.graph[u]:
                v = next(iter(self.graph[u]))
                path.append(v)
                self.remove_edge(u, v)
            else:
                break

        return path

    def find_path(self) -> List[List[Any]]:
        """
        Decompose the graph into a collection of Eulerian path components.

        Returns
        -------
        paths : list of lists
            Each sublist represents one Eulerian path component found in the graph.
            If the graph is fully Eulerian, the list will contain one complete path.
        """
        paths = []

        while any(self.graph.values()):
            start_node = self.find_start_node()
            if start_node is None:
                return paths

            path = self.hierholzer_algorithm(start_node)
            paths.append(path)

        return paths
