"""Connections are used to indicate connections between particles at the same time point, for example because the
particles are close by, or they are part of some subsystem. This is different from links, which indicates that two
observations made at different time points refer to the same particle."""
import typing
from typing import Dict, List, Set, Tuple, Iterable
import networkx
from networkx import Graph

from organoid_tracker.core import TimePoint
from organoid_tracker.core.position import Position


def _lowest_first(position1: Position, position2: Position) -> Tuple[Position, Position]:
    """Returns both positions, but the position with the lowest spatial coords first."""
    if position1.x < position2.x:
        return position1, position2
    elif position1.x > position2.x:
        return position2, position1
    elif position1.y < position2.y:
        return position1, position2
    elif position1.y > position2.y:
        return position2, position1
    elif position1.z < position2.z:
        return position1, position2
    else:
        return position2, position1


class _ConnectionsByTimePoint:
    _graph: "networkx.Graph"

    def __init__(self):
        import networkx
        self._graph = networkx.Graph()

    def add(self, position1: Position, position2: Position):
        """Adds a connection between position 1 and position 2. Does nothing if that connection already exists."""
        self._graph.add_edge(position1, position2)

    def exists(self, position1: Position, position2: Position):
        """Checks if a connection exists between position1 and position2."""
        return self._graph.has_edge(position1, position2)

    def replace_position(self, position_old: Position, position_new: Position):
        """Reroutes all connections from position_old to position_new. Does nothing if position_old has no
         connections."""
        if not self._graph.has_node(position_old):
            return False

        # Remove old edges and node
        old_connections = list(self._graph.neighbors(position_old))
        self._graph.remove_node(position_old)

        # Add new edge and nodes
        self._graph.add_node(position_new)
        for old_connection in old_connections:
            self._graph.add_edge(position_new, old_connection)

    def remove(self, position1: Position, position2: Position) -> bool:
        """Removes a connection between two positions. Does nothing if that connection doesn't exist. Returns True if
        a connection was removed."""
        if not self._graph.has_edge(position1, position2):
            return False
        self._graph.remove_edge(position1, position2)
        return True

    def remove_connections_of_position(self, position: Position):
        if self._graph.has_node(position):
            self._graph.remove_node(position)

    def get_all(self) -> Iterable[Tuple[Position, Position]]:
        """Gets all connections of this time point."""
        return self._graph.edges

    def find_connections(self, position: Position) -> Iterable[Position]:
        """Finds all connections starting and going to the given position."""
        if not self._graph.has_node(position):
            return []
        return self._graph.neighbors(position)

    def __len__(self) -> int:
        """Returns the total number of connections (lines)."""
        return len(self._graph.edges)

    def is_empty(self) -> bool:
        """Returns True if there are no connections stored for this time point."""
        return len(self._graph.edges) == 0

    def copy(self) -> "_ConnectionsByTimePoint":
        """Gets a deep copy of this object. Changes to the returned object will not affect this object, and vice versa.
        """
        copy = _ConnectionsByTimePoint()
        copy._graph = self._graph.copy()
        return copy

    def calculate_distances(self, sources: Iterable[Position]) -> Dict[Position, int]:
        """Gets the distances of all positions to the nearest position in [sources]."""
        import networkx
        sources = [source for source in sources if self._graph.has_node(source)]
        return networkx.multi_source_dijkstra_path_length(self._graph, sources)

    def has_full_neighbors(self, position: Position) -> bool:
        """
        Parameters
        ----------
        position: The position to check the neighbors for.

        Returns
        -------
        True if we think they have full neighbors annotated for that position. This is the case
        if the neighbor graph is cyclic, or if the neighbor graph contains cycles.
        """
        if not self._graph.has_node(position):
            return False
        neighbors = self._graph.subgraph(self._graph.neighbors(position))
        number_of_neighbors = neighbors.number_of_nodes()
        cyclic_graph = networkx.cycle_graph(number_of_neighbors, create_using=None)
        if number_of_neighbors > 2 and \
                (len(networkx.cycle_basis(neighbors)) > 0 or networkx.is_isomorphic(neighbors, cyclic_graph)):
            return True

        return False

    def to_networkx_graph(self) -> Graph:
        """Gets a non-directional NetworkX graph that represents the connections of this time point. The graph is a copy, so any
        changes will not affect the connections in the experiment. This has been done so that we can still switch to
        another data storage method in the future."""
        return self._graph.copy()

    def _move_in_time(self, time_point_delta: int):
        """Must only be called from the Connections class, otherwise the time index is out of sync."""
        new_graph = networkx.Graph()
        for position_a, position_b in self._graph.edges:
            new_graph.add_edge(position_a.with_time_point_number(position_a.time_point_number() + time_point_delta),
                               position_b.with_time_point_number(position_b.time_point_number() + time_point_delta))
        self._graph.clear()  # Helps garbage collector
        self._graph = new_graph


class Connections:
    """Holds the connections of an experiment."""

    _by_time_point: Dict[int, _ConnectionsByTimePoint]

    def __init__(self):
        self._by_time_point = dict()

    def add_connection(self, position1: Position, position2: Position):
        """Adds a connection between the two positions. They must be in the same time point."""
        position1.check_time_point(position2.time_point())
        if position1 == position2:
            raise ValueError(f"Both provided positions are equal: {position1}")
        time_point_number = position1.time_point_number()
        if time_point_number is None:
            raise ValueError(f"Please specify a time point number for {position1} and {position2}")

        connections = self._by_time_point.get(time_point_number)
        if connections is None:
            connections = _ConnectionsByTimePoint()
            self._by_time_point[time_point_number] = connections
        connections.add(position1, position2)

    def remove_connection(self, position1: Position, position2: Position) -> bool:
        """Removes a connection between the given positions. Does nothing if no such connection exists. Returns True if
        a connection was removed."""
        if position1.time_point_number() != position2.time_point_number():
            return False
        time_point_number = position1.time_point_number()
        if time_point_number is None:
            return False

        connections = self._by_time_point.get(time_point_number)
        if connections is None:
            return False
        if not connections.remove(position1, position2):
            return False
        if connections.is_empty():
            del self._by_time_point[time_point_number]

    def replace_position(self, position_old: Position, position_new: Position):
        """Reroutes all connections from position_old to position_new. Does nothing if position_old has no
        connections. Raises ValueError if the time point is unspecified, or if the time points of both positions are
        not equal."""
        position_old.check_time_point(position_new.time_point())
        time_point_number = position_old.time_point_number()
        if time_point_number is None:
            raise ValueError(f"Please specify a time point number for {position_old} and {position_new}")

        connections = self._by_time_point.get(time_point_number)
        if connections is None:
            return
        connections.replace_position(position_old, position_new)

    def contains_connection(self, position1: Position, position2: Position) -> bool:
        """Returns True if a connection between the two positions exists."""
        if position1.time_point_number() != position2.time_point_number():
            return False
        time_point_number = position1.time_point_number()
        if time_point_number is None:
            return False

        connections = self._by_time_point.get(time_point_number)
        if connections is None:
            return False
        return connections.exists(position1, position2)

    def of_time_point(self, time_point: TimePoint) -> Iterable[Tuple[Position, Position]]:
        """Gets all connections of a time point."""
        connections = self._by_time_point.get(time_point.time_point_number())
        if connections is None:
            return []
        return connections.get_all()

    def time_points(self) -> Iterable[TimePoint]:
        """Gets all time points that have at least one connection present."""
        for time_point_number in self._by_time_point.keys():
            yield TimePoint(time_point_number)

    def __len__(self) -> int:
        """Gets the total number of connections over all time points."""
        sum = 0
        for connections in self._by_time_point.values():
            sum += len(connections)
        return sum

    def has_connections(self) -> bool:
        """Returns True if there is at least one connection stored."""
        return len(self._by_time_point) > 0

    def is_connected(self, position: Position) -> bool:
        """Checks if the given position has a connection to anywhere."""
        for _ in self.find_connections(position):
            return True
        return False

    def find_connections(self, position: Position) -> Iterable[Position]:
        """Finds connections starting from and going to the given position. See find_connections_starting_at for
        details. This method is slower than find_connections_starting_at, as it has to do more lookups.

        Note: if you are looping over all positions in a time point, and then finding their connections, every
        connection will be found twice if you use this method (the connection from A to B will be returned, but also the
        connection from B to A). If you use find_connections_starting_at, you won't have this problem: only the
        connection from A to B will be returned."""
        time_point_number = position.time_point_number()
        connections = self._by_time_point.get(time_point_number)
        if connections is None:
            return []
        return connections.find_connections(position)

    def add_connections(self, other: "Connections"):
        """Merges all connections in the other collection with this collection."""
        for time_point_number, other_connections in other._by_time_point.items():
            if time_point_number in self._by_time_point:
                # Merge connections
                self_connections = self._by_time_point[time_point_number]
                for position1, position2 in other_connections.get_all():
                    self_connections.add(position1, position2)
            else:
                # Just copy in
                self._by_time_point[time_point_number] = other_connections.copy()

    def remove_connections_of_position(self, position: Position):
        """Removes all connections to or from the position."""
        time_point_number = position.time_point_number()
        if time_point_number is None:
            raise ValueError(f"Please specify a time point number for {position}")
        connections = self._by_time_point.get(time_point_number)
        if connections is None:
            return
        connections.remove_connections_of_position(position)

    def calculate_distances(self, sources: List[Position]) -> Dict[Position, int]:
        """Gets the distances of all positions to the nearest position in [sources].
        All sources must be in the same time point, otherwise ValueError is raised.
        Returns an empty dictionary if sources is empty, or if there are no connections."""
        if len(sources) == 0:
            return dict()

        time_point = None
        for source in sources:
            if time_point is None:
                time_point = source.time_point()
            elif time_point.time_point_number() != source.time_point_number():
                raise ValueError("All positions must be in the same time point; " + str(sources))

        if time_point.time_point_number() in self._by_time_point:
            return self._by_time_point[time_point.time_point_number()].calculate_distances(sources)
        return dict()

    def calculate_distances_over_time(self, sources: Iterable[Position]) -> Dict[Position, int]:
        """Like calculate_distances, but supports sources for multiple time points. Note: connection distances are still
        only calculated within a time point. So if you're 10 connections from a Paneth cell, but in the next time point
        1, then for the original time point the distance is still 10."""
        by_time_point = dict()
        for position in sources:
            time_point = position.time_point()
            if time_point in by_time_point:
                by_time_point[time_point].append(position)
            else:
                by_time_point[time_point] = [position]

        results = dict()
        for time_point, positions in by_time_point.items():
            results.update(self.calculate_distances(positions))
        return results

    def contains_time_point(self, time_point: TimePoint) -> bool:
        """Returns whether there are connections for the given time point."""
        return time_point.time_point_number() in self._by_time_point

    def has_full_neighbors(self, position: Position) -> bool:
        """
        Parameters
        ----------
        position: The position to check the neighbors for.

        Returns
        -------
        True if we think they have full neighbors annotated for that position. This is the case
        if the neighbor graph is cyclic, or if the neighbor graph contains cycles.
        """
        if position.time_point_number() in self._by_time_point:
            return self._by_time_point[position.time_point_number()].has_full_neighbors(position)
        return False

    def to_networkx_graph(self, *, time_point: TimePoint) -> Graph:
        """Gets a non-directional NetworkX graph that represents the connections of the given time point. The graph is
        a copy, so any changes will not affect the connections in the experiment. This has been done so that we can
        still switch to another data storage method in the future."""
        if time_point.time_point_number() not in self._by_time_point:
            return networkx.Graph()  # Return an empty graph
        return self._by_time_point[time_point.time_point_number()].to_networkx_graph()

    def copy(self) -> "Connections":
        """Returns a copy of this object. Changes made to the copy will not affect this object."""
        copy = Connections()
        for time_point, connections in self._by_time_point.items():
            copy._by_time_point[time_point] = connections.copy()
        return copy

    def move_in_time(self, time_point_delta: int):
        """Moves all data with the given time point delta."""
        new_connections_dict = dict()
        for time_point_number, values in self._by_time_point.items():
            values._move_in_time(time_point_delta)
            new_connections_dict[time_point_number + time_point_delta] = values
        self._by_time_point = new_connections_dict
