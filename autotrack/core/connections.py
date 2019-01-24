"""Connections are used to indicate connections between particles at the same time point, for example because the
particles are close by, or they are part of some subsystem. This is different from links, which indicates that two
observations made at different time points refer to the same particle."""
from typing import Dict, List, Set, Tuple, Iterable

from autotrack.core import TimePoint
from autotrack.core.position import Position


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

    _connections: Dict[Position, Set[Position]]

    def __init__(self):
        self._connections = dict()

    def add(self, position1: Position, position2: Position):
        """Adds a connection between position 1 and position 2. Does nothing if that connection already exists."""
        position1, position2 = _lowest_first(position1, position2)
        connections_position1 = self._connections.get(position1)
        if connections_position1 is None:
            self._connections[position1] = {position2}
        else:
            connections_position1.add(position2)

    def exists(self, position1: Position, position2: Position):
        """Checks if a connection exists between position1 and position2."""
        position1, position2 = _lowest_first(position1, position2)
        connections_position1 = self._connections.get(position1)
        if connections_position1 is None:
            return False

        return position2 in connections_position1

    def replace_position(self, position_old: Position, position_new: Position):
        """Reroutes all connections from position_old to position_new. Does nothing if position_old has no
         connections."""

        # Replace as key
        old_connections = self._connections.get(position_old)
        if old_connections is not None:
            del self._connections[position_old]
            self._connections[position_new] = old_connections

        # Replace as value
        for position1, positions2 in self._connections.items():
            if position_old in positions2:
                positions2.remove(position_old)
                positions2.add(position_new)

    def remove(self, position1: Position, position2: Position) -> bool:
        """Removes a connection between two positions. Does nothing if that connection doesn't exist. Returns True if
        a connection was removed."""
        position1, position2 = _lowest_first(position1, position2)
        connections_position1 = self._connections.get(position1)
        if connections_position1 is None or position2 not in connections_position1:
            return False  # This connection doesn't exist
        if len(connections_position1) == 1:
            del self._connections[position1]  # Removed the last connection of this position
        else:
            connections_position1.remove(position2)  # Remove this connection, but more remain
        return True

    def remove_connections_of_position(self, position: Position):
        # Delete as key
        try:
            del self._connections[position]
        except KeyError:
            pass

        # Delete as value
        for position1, positions2 in self._connections.items():
            positions2.discard(position)

    def get_all(self) -> Iterable[Tuple[Position, Position]]:
        """Gets all connections of this time point."""
        for position1, positions2 in self._connections.items():
            for position2 in positions2:
                yield position1, position2

    def find_connections_starting_at(self, position: Position) -> List[Position]:
        """Returns connections starting from the given position. Note: connections are stored internally as going from
        the lowest position (x, then y, then z) to the highest position. This method only returns connections going from
        a position, not connections going to a position."""
        if position in self._connections:
            return list(self._connections[position])
        return []

    def __len__(self) -> int:
        """Returns the total number of connections (lines)."""
        sum = 0
        for value in self._connections.values():
            sum += len(value)
        return sum

    def is_empty(self) -> bool:
        """Returns True if there are no connections stored for this time point."""
        return len(self._connections) == 0

    def copy(self) -> "_ConnectionsByTimePoint":
        """Gets a deep copy of this object. Changes to the returned object will not affect this object, and vice versa.
        """
        copy = _ConnectionsByTimePoint()
        copy._connections = self._connections.copy()
        return copy


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

    def find_connections_starting_at(self, position: Position) -> List[Position]:
        """Returns connections starting from the given position. Note: connections are stored internally as going from
        the lowest position (x, then y, then z) to the highest position. This method only returns connections going from
        a position, not connections going to a position."""
        time_point_number = position.time_point_number()
        connections = self._by_time_point.get(time_point_number)
        if connections is None:
            return []
        return connections.find_connections_starting_at(position)

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
