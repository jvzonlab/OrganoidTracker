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
    position_at_time =find_position_in_connected_time_point(experiment, position)
    if position_at_time is None:
        return None  # We don't know
    connections_at_time = list(
        connections.find_connections(position_at_time)) if position_at_time is not None else []
    if len(connections_at_time) == 0:
        return None  # No connections at that time - only part of the connections are specified. So we don't know.

    # There are connections, find if there's a Paneth cell
    for connection in connections_at_time:
        if condition(experiment, connection):
            return True
    return False


def find_position_in_connected_time_point(experiment: Experiment, position: Position, max_dt: int = 9) -> Optional[Position]:
    """When the links are established by hand, we don't annotate every time point. That would be too much work.
    This method runs a bit forward and backward in time until it finds a time point that does have connections defined.
    It then returns the position at that time point. Returns None if we couldn't find the same cell in a time point
    with links (within dt)."""
    links = experiment.links

    for dt_abs in range(max_dt):
        for dt in [dt_abs, -dt_abs]:
            if dt == 0:
                position_at_time = position
            else:
                time_point = TimePoint(position.time_point_number() + dt)
                position_at_time = links.get_position_near_time_point(position, time_point)
                if position_at_time is None or position_at_time.time_point() != time_point:
                    continue

            if experiment.connections.contains_time_point(position_at_time.time_point()):
                return position_at_time
    return None
