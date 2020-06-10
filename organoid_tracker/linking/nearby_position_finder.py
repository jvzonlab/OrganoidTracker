"""Contains function that allows you to find the nearest few positions"""

import operator
from typing import Iterable, List, Optional, Set, Dict

from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution


class _NearestPositions:
    """Internal class for bookkeeping of what the nearest few positions are"""

    _tolerance_squared: float
    _all_distances: Dict[Position, float]
    _shortest_distance_squared: float
    _max_distance_squared_um2: float  # Used to set a maximum distance above which we don't even need to look anymore

    def __init__(self, tolerance: float, max_distance_um: float):
        self._tolerance_squared = tolerance ** 2
        self._max_distance_squared_um2 = max_distance_um ** 2
        self._all_distances = {}
        self._shortest_distance_squared = float("inf")

    def add_candidate(self, new_position: Position, distance_squared: float) -> None:
        if distance_squared > self._max_distance_squared_um2:
            return  # Too far according to global limit
        if distance_squared > self._tolerance_squared * self._shortest_distance_squared:
            return  # Position is too far away compared to nearest

        if distance_squared < self._shortest_distance_squared:
            # New shortest distance
            self._shortest_distance_squared = distance_squared

        self._all_distances[new_position] = distance_squared

    def get_positions(self, max_amount: int) -> List[Position]:
        """Gets the found positions."""
        max_allowed_distance_squared = self._shortest_distance_squared * self._tolerance_squared
        return_list = []
        for position, distance_squared in self._all_distances.items():
            if distance_squared <= max_allowed_distance_squared:
                return_list.append(position)
        if len(return_list) > max_amount:
            # Need to return only the closest
            def get_distance_squared(pos: Position):
                return self._all_distances[pos]
            return_list.sort(key=get_distance_squared)
            return return_list[0:max_amount]
        return return_list


def find_close_positions(positions: Iterable[Position], *, around: Position, tolerance: float, resolution: ImageResolution,
                         max_amount: int = 1000, max_distance_um: float = float("inf")) -> List[Position]:
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
    nearest_positions = _NearestPositions(tolerance, max_distance_um)
    for position in positions:
        nearest_positions.add_candidate(position, position.distance_squared(around, resolution))
    return nearest_positions.get_positions(max_amount)


def find_closest_position(positions: Iterable[Position], *, around: Position, resolution: ImageResolution,
                          ignore_z: bool = False, max_distance_um: int = 100000) -> Optional[Position]:
    """Gets the position closest ot the given position."""
    closest_position = None
    closest_distance_squared = max_distance_um ** 2

    for position in positions:
        if ignore_z:
            around.z = position.z  # Make search ignore z
        distance = position.distance_squared(around, resolution)

        around_time_point_number = around.time_point_number()
        if around_time_point_number is not None:  # Make positions in same time point closer
            distance += (around_time_point_number - position.time_point_number()) ** 2

        if distance < closest_distance_squared:
            closest_distance_squared = distance
            closest_position = position

    return closest_position


def find_closest_n_positions(positions: Iterable[Position], *, around: Position, max_amount: int,
                             resolution: ImageResolution, max_distance_um: float = 100000, ignore_self: bool = True
                             ) -> Set[Position]:
    max_distance_squared = max_distance_um ** 2
    closest_positions = []

    for position in positions:
        if ignore_self and position == around:
            continue
        distance_squared = position.distance_squared(around, resolution)
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
