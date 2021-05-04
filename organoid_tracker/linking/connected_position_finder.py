from typing import Callable, Optional

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position


def has_neighbor_that(experiment: Experiment, position: Position, condition: Callable[[Experiment, Position], bool], max_dt: int = 9) -> Optional[bool]:
    """Gets whether any neighbor cell (determined using experiment.connections, at time points up to abs(dt)
    away) conforms to the given condition.."""
    links = experiment.links
    connections = experiment.connections

    # We don't map out the links for every time point, that would be too much work
    # So we need to go back and forward a bit
    for dt_abs in range(max_dt):
        for dt in [dt_abs, -dt_abs]:
            if dt == 0:
                position_at_time = position
            else:
                time_point = TimePoint(position.time_point_number() + dt)
                position_at_time = links.get_position_near_time_point(position, time_point)
                if position_at_time is None:
                    continue
            connections_at_time = list(
                connections.find_connections(position_at_time)) if position_at_time is not None else []
            if len(connections_at_time) == 0:
                continue  # No connections at that time - only part of the connections are specified

            # There are connections, find if there's a Paneth cell
            for connection in connections_at_time:
                if condition(experiment, connection):
                    return True
            return False
    return None  # We don't know
