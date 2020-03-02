"""Ultra-simple linker. Used as a starting point for more complex links."""
from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.images import Images
from organoid_tracker.core.links import Links
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.linking.nearby_position_finder import find_close_positions


def nearest_neighbor(experiment: Experiment, *, tolerance: float = 1.0, back: bool = True, forward: bool = True
                     ) -> Links:
    """Simple nearest neighbour linking, keeping a list of potential candidates based on a given tolerance.

    A tolerance of 1.05 also links positions 5% from the closest position, so you end up with more links than you have
    positions.

    If the experiment has images loaded, then no links outside the images will be created.

    Nearest neighbor-linking can happen both forwards (every position is linked to the nearest in the next time point)
    and backwards (every position is linked to the nearest in the previous time point). If you do both, note that the
    tolerance is calculated independently for both directions: with a tolerance of for example 2, you'll get all forward
    links that are at most twice as long as the shortest forward link, and you'll get all backward links that are at
    most twice as long as the shortest backward link.
    """
    if not back and not forward:
        raise ValueError("Cannot create links if back and forward are both False.")
    links = Links()

    time_point_previous = None
    for time_point_current in experiment.time_points():

        if time_point_previous is not None:
            if back:
                _add_nearest_edges(links, experiment.positions, experiment.images,
                                   time_point_previous, time_point_current, tolerance)
            if forward:
                _add_nearest_edges_extra(links, experiment.positions, experiment.images,
                                         time_point_previous, time_point_current, tolerance)

        if time_point_current.time_point_number() % 50 == 0:
            print("    completed up to time point", time_point_current.time_point_number())
        time_point_previous = time_point_current

    print("Done creating nearest-neighbor links!")
    return links


def _add_nearest_edges(links: Links, positions: PositionCollection, images: Images, time_point_previous: TimePoint,
                       time_point_current: TimePoint, tolerance: float):
    """Adds edges pointing towards previous time point, making the shortest one the preferred."""
    resolution = images.resolution()
    for position in positions.of_time_point(time_point_current):
        # Check if position was inside the image in the previous time point
        previous_position = position.with_time_point(time_point_previous)
        if images.is_inside_image(previous_position) is False:
            # ^ Using "is False" because the method can also return None
            continue  # Skip, position will go out of view

        # If yes, make links to previous time point
        nearby_list = find_close_positions(positions.of_time_point(time_point_previous), around=position,
                                           tolerance=tolerance, max_amount=5, resolution=resolution)
        for nearby_position in nearby_list:
            links.add_link(position, nearby_position)


def _add_nearest_edges_extra(links: Links, positions: PositionCollection, images: Images, time_point_current: TimePoint, time_point_next: TimePoint, tolerance: float):
    """Adds edges to the next time point, which is useful if _add_edges missed some possible links."""
    resolution = images.resolution()
    for position in positions.of_time_point(time_point_current):
        # Check if position is still inside the image in the next time point
        next_position = position.with_time_point(time_point_next)
        if images.is_inside_image(next_position) is False:
            # ^ Using "is False" because the method can also return None
            continue  # Skip, position will go out of view

        # If yes, make links to next time point
        nearby_list = find_close_positions(positions.of_time_point(time_point_next), around=position,
                                           tolerance=tolerance, max_amount=5, resolution=resolution)
        for nearby_position in nearby_list:
            links.add_link(position, nearby_position)
