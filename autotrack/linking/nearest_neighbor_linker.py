"""Ultra-simple linker. Used as a starting point for more complex links."""

from autotrack.core.links import PositionLinks
from autotrack.core.positions import PositionCollection
from autotrack.core import TimePoint
from autotrack.core.experiment import Experiment
from autotrack.linking.nearby_position_finder import find_close_positions


def nearest_neighbor(experiment: Experiment, tolerance: float = 1.0) -> PositionLinks:
    """Simple nearest neighbour linking, keeping a list of potential candidates based on a given tolerance.

    A tolerance of 1.05 also links positions 5% from the closest position. Note that if a tolerance higher than 1 is
    given, some pruning is needed on the final result.

    max_time_point is the last time point that will still be included.
    """
    links = PositionLinks()

    time_point_previous = None
    for time_point_current in experiment.time_points():
        if time_point_previous is not None:
            _add_nearest_edges(links, experiment.positions, time_point_previous, time_point_current, tolerance)
            _add_nearest_edges_extra(links, experiment.positions, time_point_previous, time_point_current, tolerance)

        time_point_previous = time_point_current

    print("Done creating nearest-neighbor links!")
    return links


def _add_nearest_edges(links: PositionLinks, positions: PositionCollection, time_point_previous: TimePoint, time_point_current: TimePoint, tolerance: float):
    """Adds edges pointing towards previous time point, making the shortest one the preferred."""
    for position in positions.of_time_point(time_point_current):
        nearby_list = find_close_positions(positions.of_time_point(time_point_previous), position, tolerance, max_amount=5)
        for nearby_position in nearby_list:
            links.add_link(position, nearby_position)


def _add_nearest_edges_extra(links: PositionLinks, positions: PositionCollection, time_point_current: TimePoint, time_point_next: TimePoint, tolerance: float):
    """Adds edges to the next time point, which is useful if _add_edges missed some possible links."""
    for position in positions.of_time_point(time_point_current):
        nearby_list = find_close_positions(positions.of_time_point(time_point_next), position, tolerance, max_amount=5)
        for nearby_position in nearby_list:
            links.add_link(position, nearby_position)
