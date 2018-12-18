"""Contains function that allows you to find the nearest few positions"""

import operator
from typing import Iterable, List, Optional, Set

from autotrack.core import TimePoint
from autotrack.core.positions import PositionCollection, Position


class _NearestPositions:
    """Internal class for bookkeeping of what the nearest few positions are"""

    def __init__(self, tolerance: float):
        self._tolerance_squared = tolerance ** 2
        self._nearest = {}
        self._shorted_distance_squared = float("inf")

    def add_candidate(self, new_position: Position, distance_squared: float) -> None:
        if distance_squared > self._tolerance_squared * self._shorted_distance_squared:
            return # Position is too far away compared to nearest

        if distance_squared < self._shorted_distance_squared:
            # New shortest distance, remove positions that do not conform
            self._shorted_distance_squared = distance_squared
            self._prune()

        self._nearest[new_position] = distance_squared

    def _prune(self):
        """Removes all cells with distances greater than shortest_distance * tolerance. Useful when the shortest
        distance just changed."""
        max_allowed_distance_squared = self._shorted_distance_squared * self._tolerance_squared
        for position in list(self._nearest.keys()): # Iterating over copy of keys to avoid a RuntimeError
            its_distance_squared = self._nearest[position]
            if its_distance_squared > max_allowed_distance_squared:
                del self._nearest[position]

    def get_positions(self, max_amount: int) -> List[Position]:
        """Gets the found positions."""
        items = sorted(self._nearest.items(), key=operator.itemgetter(1))
        positions = [item[0] for item in items]
        if len(positions) > max_amount:
            return positions[0:max_amount]
        return positions


def find_close_positions(positions: Iterable[Position], around: Position, tolerance: float,
                         max_amount: int = 1000) -> List[Position]:
    """Finds the positions nearest to the given position.

    - search_in is the time_point to search in
    - around is the position to search around
    - tolerance is a number that influences how much positions other than the nearest are included. A tolerance of 1.05
      makes positions that are 5% further than the nearest also included.
    - max_amount if the maximum amount of returned positions. If there are more positions within the tolerance, then
      only the nearest positions are returned.

    Returns a list of the nearest positions, ordered from closest to furthest
    """
    if tolerance < 1:
        raise ValueError()
    nearest_positions = _NearestPositions(tolerance)
    for position in positions:
        nearest_positions.add_candidate(position, position.distance_squared(around, z_factor=3))
    return nearest_positions.get_positions(max_amount)


def find_closest_position(positions: Iterable[Position], around: Position, ignore_z: bool = False,
                          max_distance: int = 100000) -> Optional[Position]:
    """Gets the position closest ot the given position."""
    closest_position = None
    closest_distance_squared = max_distance ** 2

    for position in positions:
        if ignore_z:
            around.z = position.z  # Make search ignore z
        distance = position.distance_squared(around)
        if distance < closest_distance_squared:
            closest_distance_squared = distance
            closest_position = position

    return closest_position


def find_closest_n_positions(positions: Iterable[Position], around: Position, max_amount: int,
                             max_distance: int = 100000) -> Set[Position]:
    max_distance_squared = max_distance ** 2
    closest_positions = []

    for position in positions:
        distance_squared = position.distance_squared(around)
        if distance_squared > max_distance_squared:
            continue
        if len(closest_positions) < max_amount or closest_positions[-1][0] > distance_squared:
            # Found closer position
            closest_positions.append((distance_squared, position))
            closest_positions.sort(key=operator.itemgetter(0))
            if len(closest_positions) > max_amount:
                # List too long, remove furthest
                del closest_positions[-1]

    return_value = set()
    for distance_squared, position in closest_positions:
        return_value.add(position)
    return return_value
