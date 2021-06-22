from typing import Optional

from organoid_tracker.core import TimePoint
from organoid_tracker.core.connections import Connections
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution


class ConnectorByDistance:
    """Used to create connections between positions if they are close enough."""

    _max_distance_um: float
    _max_number: Optional[int]

    def __init__(self, max_distance_um: float, max_number: Optional[int] = None):
        """Creates a connector that adds connections between all positions that are within the specified distance of
        each other."""
        self._max_distance_um = max_distance_um
        self._max_number = max_number

    def create_connections(self, experiment: Experiment) -> Connections:
        """Adds connections for all time points in the experiment. Doesn't modify the experiment; instead this method
        returns the new connections. (This is useful for implementing Undo functionality.)"""
        connections = Connections()
        for time_point in experiment.time_points():
            self._add_connections_for_time_point(connections, experiment, time_point)
        return connections

    def _add_connections_for_time_point(self, connections: Connections, experiment: Experiment, time_point: TimePoint):
        """Adds connections for a single time point to the specified collection."""
        positions = experiment.positions.of_time_point(time_point)
        resolution = experiment.images.resolution()

        for position1 in positions:
            for position2 in positions:
                if position1 is position2:
                    continue   # Don't connect to self
                distance_um = position1.distance_um(position2, resolution)
                if distance_um > self._max_distance_um:
                    continue

                connections.add_connection(position1, position2)
            self._prune_to_max_number(connections, resolution, position1)

    def _prune_to_max_number(self, connections: Connections, resolution: ImageResolution, position: Position):
        if self._max_number is None:
            return
        all_connections = list(connections.find_connections(position))
        if len(all_connections) <= self._max_number:
            return  # No need to remove anything
        all_distances_squared = [connection.distance_squared(position, resolution) for connection in all_connections]

        # Remove largest distances
        srt = sorted(zip(all_distances_squared, all_connections),key=lambda item: item[0])
        for distance_squared, connection in srt[self._max_number:]:
            connections.remove_connection(connection, position)
