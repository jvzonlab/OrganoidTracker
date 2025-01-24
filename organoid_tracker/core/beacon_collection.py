import math
from typing import List, Dict, Optional, Iterable, NamedTuple, Tuple

from organoid_tracker.core import TimePoint
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.core.vector import Vector3


_DEFAULT_BEACON_TYPE = "__DEFAULT__"


class Beacon(NamedTuple):
    """A beacon is a named point in space."""
    position: Position
    beacon_type: Optional[str]


class ClosestBeacon:
    """Used to represent the distance towards, the identity and the position of the closest beacon."""
    distance_um: float
    beacon_position: Position
    beacon_type: str
    search_position: Position
    beacon_index: int
    resolution: ImageResolution

    def __init__(self, search_position: Position, beacon_position: Position, beacon_type: Optional[str], distance_um: float,
                 resolution: ImageResolution):
        self.search_position = search_position
        self.beacon_position = beacon_position
        self.beacon_type = beacon_type
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

    _beacons: Dict[TimePoint, Dict[Position, str]]

    def __init__(self):
        self._beacons = dict()

    def add(self, position: Position, beacon_type: Optional[str] = None):
        """Adds a new beacon. If there is already a beacon at the given position, it is overwritten."""
        time_point = position.time_point()
        if time_point is None:
            raise ValueError(f"No time point specified for {position}.")
        if beacon_type is None:
            beacon_type = _DEFAULT_BEACON_TYPE  # We don't store None

        beacons_at_time_point = self._beacons.get(time_point)
        if beacons_at_time_point is None:
            beacons_at_time_point = dict()
            self._beacons[time_point] = beacons_at_time_point
        beacons_at_time_point[position] = beacon_type

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
            del beacons_at_time_point[position]
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

        old_beacon_type = beacons_at_time_point.get(old_position)
        if old_beacon_type is None:
            return False  # No beacon at old position

        del beacons_at_time_point[old_position]
        beacons_at_time_point[new_position] = old_beacon_type
        return True

    def contains_position(self, beacon: Position) -> bool:
        """Gets whether the given beacon exists in this collection."""
        time_point = beacon.time_point()
        beacons_at_time_point = self._beacons.get(time_point)
        if beacons_at_time_point is None:
            return False
        return beacon in beacons_at_time_point

    def of_time_point(self, time_point: TimePoint) -> Iterable[Position]:
        """Gets all beacons at the given time point."""
        beacons_at_time_point = self._beacons.get(time_point)
        if beacons_at_time_point is None:
            return
        yield from beacons_at_time_point

    def get_beacon_type(self, beacon_position: Position) -> Optional[str]:
        """Gets the name of the beacon at the given position."""
        time_point = beacon_position.time_point()
        if time_point is None:
            raise ValueError(f"No time point specified for {beacon_position}.")
        beacons_at_time_point = self._beacons.get(time_point)
        if beacons_at_time_point is None:
            return None
        beacon_type = beacons_at_time_point.get(beacon_position)
        if beacon_type == _DEFAULT_BEACON_TYPE:
            return None
        return beacon_type

    def set_beacon_type(self, beacon_position: Position, name: Optional[str]):
        """Sets the name of the beacon at the given position. Does nothing if there is no beacon at that position."""
        time_point = beacon_position.time_point()
        if time_point is None:
            raise ValueError(f"No time point specified for {beacon_position}.")
        beacons_at_time_point = self._beacons.get(time_point)
        if beacons_at_time_point is None:
            return

        if beacon_position not in beacons_at_time_point:
            return

        beacons_at_time_point[beacon_position] = name if name is not None else _DEFAULT_BEACON_TYPE

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
        closest_beacon_type = None
        for beacon, name in beacons_at_time_point.items():
            distance_squared = beacon.distance_squared(position, resolution)
            if distance_squared < shortest_distance_squared:
                shortest_distance_squared = distance_squared
                closest_beacon = beacon
                closest_beacon_type = name
        if closest_beacon_type == _DEFAULT_BEACON_TYPE:
            closest_beacon_type = None
        return ClosestBeacon(position, closest_beacon, closest_beacon_type, math.sqrt(shortest_distance_squared),
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
        """Adds all beacons from the given collection to this collection."""
        for time_point, beacons_of_time_point in beacons._beacons.items():
            if time_point in self._beacons:
                self._beacons[time_point].update(beacons_of_time_point)
            else:
                self._beacons[time_point] = beacons_of_time_point

    def find_single_beacon(self) -> Optional[Position]:
        """If there is only one beacon in the entire experiment, return it. Otherwise, it returns None."""
        if len(self._beacons) != 1:
            return None  # There cannot be exactly one beacon
        beacons = next(iter(self._beacons.values()))
        if len(beacons) == 1:
            return next(iter(beacons.keys()))
        return None

    def find_single_beacon_of_time_point(self, time_point: TimePoint) -> Optional[Position]:
        """If there is only one beacon in the given time point, return it. Otherwise, it returns None."""
        beacons = self._beacons.get(time_point)
        if beacons is None or len(beacons) != 1:
            return None
        return next(iter(beacons.keys()))

    def move_in_time(self, time_point_delta: int):
        """Moves all data with the given time point delta."""
        new_beacons_dict = dict()
        for time_point, values in self._beacons.items():
            new_beacons_dict[time_point + time_point_delta] = values
        self._beacons = new_beacons_dict

    def of_time_point_with_type(self, time_point: TimePoint) -> Iterable[Beacon]:
        """Gets all beacons at the given time point, including their beacon types."""
        beacons_at_time_point = self._beacons.get(time_point)
        if beacons_at_time_point is None:
            return
        for position, name in beacons_at_time_point.items():
            if name == _DEFAULT_BEACON_TYPE:
                name = None
            yield Beacon(position, name)
