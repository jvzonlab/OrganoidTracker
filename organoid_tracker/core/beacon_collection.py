import math
from typing import List, Dict, Optional, Iterable

from organoid_tracker.core import TimePoint
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.core.vector import Vector3


class ClosestBeacon:
    """Used to represent the distance towards, the identity and the position of the closest beacon."""
    distance_um: float
    beacon_position: Position
    search_position: Position
    beacon_index: int
    resolution: ImageResolution

    def __init__(self, search_position: Position, beacon_position: Position, beacon_index: int, distance_um: float,
                 resolution: ImageResolution):
        self.search_position = search_position
        self.beacon_position = beacon_position
        self.beacon_index = beacon_index
        self.distance_um = distance_um
        self.resolution = resolution

    def difference_um(self) -> Vector3:
        """Gets the difference between the search position and the beacon position in micrometers."""
        dx = (self.search_position.x - self.beacon_position.x) * self.resolution.pixel_size_x_um
        dy = (self.search_position.y - self.beacon_position.y) * self.resolution.pixel_size_y_um
        dz = (self.search_position.z - self.beacon_position.z) * self.resolution.pixel_size_z_um
        return Vector3(dx, dy, dz)


class BeaconCollection:
    """Ordered list of beacons per time point."""

    _beacons: Dict[TimePoint, List[Position]]

    def __init__(self):
        self._beacons = dict()

    def add(self, position: Position):
        """Adds a new beacon. Duplicate beacons are allowed."""
        time_point = position.time_point()
        if time_point is None:
            raise ValueError(f"No time point specified for {position}.")

        beacons_at_time_point = self._beacons.get(time_point)
        if beacons_at_time_point is None:
            beacons_at_time_point = list()
            self._beacons[time_point] = beacons_at_time_point
        beacons_at_time_point.append(position)

    def remove(self, position: Position) -> bool:
        """Removes the beacon at the given position. Returns True if succesful, returns False if there was no beacon at
        that position."""
        time_point = position.time_point()
        if time_point is None:
            raise ValueError(f"No time point specified for {position}.")

        beacons_at_time_point = self._beacons.get(time_point)
        if beacons_at_time_point is None:
            return False

        try:
            beacons_at_time_point.remove(position)
            if len(beacons_at_time_point) == 0:
                del self._beacons[time_point]  # Remove the now-empty list
            return True
        except ValueError:
            return False  # Nothing was deleted

    def move(self, old_position: Position, new_position: Position) -> bool:
        """Moves a beacon from its old to its new position. The old and new positions must have the same time point.
        Does nothing and returns False if there was no beacon at old_position. Returns True if successful."""
        time_point = old_position.time_point()
        if time_point is None:
            raise ValueError(f"No time point specified for {old_position}.")
        if new_position.time_point_number() != time_point.time_point_number():
            raise ValueError(f"Different time points for old and new position: old={old_position} vs new={new_position}")

        beacons_at_time_point = self._beacons.get(time_point)
        if beacons_at_time_point is None:
            return False  # No beacons at that time point, nothing to move

        for i in range(len(beacons_at_time_point)):
            if beacons_at_time_point[i] == old_position:
                beacons_at_time_point[i] = new_position
                return True
        return False

    def contains_position(self, beacon: Position) -> bool:
        """Gets whether the given beacon exists in this collection."""
        time_point = beacon.time_point()
        beacons_at_time_point = self._beacons.get(time_point)
        if beacons_at_time_point is None:
            return False
        return beacon in beacons_at_time_point

    def get_next_index(self, time_point: TimePoint) -> int:
        """Beacons have indices: 1, 2, 3, etc. for every time point. This method returns the index that the next beacon
        placed at the given time point will get."""
        beacons_at_time_point = self._beacons.get(time_point)
        if beacons_at_time_point is None:
            return 1
        return len(beacons_at_time_point) + 1

    def get_beacon_by_index(self, time_point: TimePoint, index: int) -> Optional[Position]:
        """Gets the beacon with the given index at the given time point. Index 1 is the first beacon."""
        beacons_at_time_point = self._beacons.get(time_point)
        if beacons_at_time_point is None:
            return None
        if index <= 0 or index > len(beacons_at_time_point):
            return None
        return beacons_at_time_point[index - 1]

    def of_time_point(self, time_point: TimePoint) -> Iterable[Position]:
        """Gets all beacons at the given time point."""
        beacons_at_time_point = self._beacons.get(time_point)
        if beacons_at_time_point is None:
            return
        yield from beacons_at_time_point

    def find_closest_beacon(self, position: Position, resolution: ImageResolution) -> Optional[ClosestBeacon]:
        """Finds the closest beacon at the same time point as the position. Returns None if there are no beacons at that
        time point."""
        time_point = position.time_point()
        if time_point is None:
            raise ValueError(f"No time point specified for {position}.")
        beacons_at_time_point = self._beacons.get(time_point)
        if beacons_at_time_point is None:
            return None

        shortest_distance_squared = float("inf")
        closest_beacon = None
        closest_beacon_index = None
        for i, beacon in enumerate(beacons_at_time_point):
            distance_squared = beacon.distance_squared(position, resolution)
            if distance_squared < shortest_distance_squared:
                shortest_distance_squared = distance_squared
                closest_beacon = beacon
                closest_beacon_index = i + 1
        return ClosestBeacon(position, closest_beacon, closest_beacon_index, math.sqrt(shortest_distance_squared),
                             resolution)

    def time_points(self) -> Iterable[TimePoint]:
        """Gets all used time points."""
        return set(self._beacons.keys())

    def count_beacons_at_time_point(self, time_point: TimePoint) -> int:
        """Gets the number of beacons at the given time point."""
        beacons = self._beacons.get(time_point)
        if beacons is None:
            return 0
        return len(beacons)

    def has_beacons(self) -> bool:
        """Checks whether there are any beacons stored."""
        return len(self._beacons) > 0

    def add_beacons(self, beacons: "BeaconCollection"):
        """Adds all beacons from the given collection to this collection. Like for add(..), duplicate beacons are
         allowed."""
        for time_point, beacons_of_time_point in beacons._beacons.items():
            if time_point in self._beacons:
                self._beacons[time_point] += beacons_of_time_point
            else:
                self._beacons[time_point] = beacons_of_time_point

    def find_single_beacon(self) -> Optional[Position]:
        """If there is only one beacon in the entire experiment, return it. Otherwise, it returns None."""
        if len(self._beacons) != 1:
            return None  # There cannot be exactly one beacon
        beacons = next(iter(self._beacons.values()))
        if len(beacons) == 1:
            return beacons[0]
        return None