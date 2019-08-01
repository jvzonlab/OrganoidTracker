from ai_track.core import TimePoint
from ai_track.core.connections import Connections
from ai_track.core.experiment import Experiment


class ConnectorByDistance:
    """Used to create connections between positions if they are close enough."""

    _max_distance_um: float

    def __init__(self, max_distance_um: float):
        """Creates a connector that adds connections between all positions that are within the specified distance of
        each other."""
        self._max_distance_um = max_distance_um

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
                if distance_um <= self._max_distance_um:
                    connections.add_connection(position1, position2)
