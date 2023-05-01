from typing import Optional, Iterable, Callable, Tuple

from organoid_tracker.core.link_data import LinkData
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import Links
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.warning_limits import WarningLimits
from organoid_tracker.linking_analysis import linking_markers, particle_age_finder
from organoid_tracker.linking_analysis.errors import Error


def find_errors_in_experiment(experiment: Experiment) -> Tuple[int, int]:
    """Adds errors for all logical inconsistencies in the graph, like cells that spawn out of nowhere, cells that
    merge together and cells that have three or more daughters.
    Returns the amount of errors (excluding positions without links) and the number of positions without links."""
    position_data = experiment.position_data
    links = experiment.links

    warning_count = 0
    no_links_count = 0
    for position in experiment.positions:
        error = get_error(experiment, position)
        linking_markers.set_error_marker(position_data, position, error)
        if error is not None:
            if links.contains_position(position):
                warning_count += 1
            else:  # It's just a position without links
                no_links_count += 1
    return warning_count, no_links_count


def get_error(experiment: Experiment, position: Position) -> Optional[Error]:
    links = experiment.links
    position_data = experiment.position_data
    link_data = experiment.link_data
    positions = experiment.positions
    resolution = experiment.images.resolution()
    warning_limits = experiment.warning_limits

    if linking_markers.is_uncertain(position_data, position):
        return Error.UNCERTAIN_POSITION

    if not links.has_links():
        return  # Don't attempt to find other errors

    future_positions = links.find_futures(position)
    if len(future_positions) > 2:
        return Error.TOO_MANY_DAUGHTER_CELLS
    elif len(future_positions) == 0 \
            and position.time_point_number() < positions.last_time_point_number() \
            and linking_markers.get_track_end_marker(position_data, position) is None:
        return Error.NO_FUTURE_POSITION
    elif len(future_positions) == 2:
        # Found a putative mother
        division_probability = position_data.get_position_data(position, 'division_probability')
        if division_probability is not None and division_probability < warning_limits.min_probability:
            return Error.LOW_MOTHER_SCORE

        age = particle_age_finder.get_age(links, position)
        if age is not None and age * resolution.time_point_interval_h < warning_limits.min_time_between_divisions_h:
            return Error.YOUNG_MOTHER
    elif len(future_positions) == 1:
        division_probability = position_data.get_position_data(position, 'division_probability')
        if division_probability is not None\
                and division_probability > 1 - warning_limits.min_probability:
            # Likely missed a division
            if not _has_high_division_probability_hereafter(links, position_data, warning_limits,
                                                            next(iter(future_positions))):
                return Error.POTENTIALLY_SHOULD_BE_A_MOTHER

    past_positions = links.find_pasts(position)
    if len(past_positions) == 0:
        if position.time_point_number() > positions.first_time_point_number() \
                and linking_markers.get_track_start_marker(position_data, position) is None:
            return Error.NO_PAST_POSITION
    elif len(past_positions) >= 2:
        return Error.CELL_MERGE
    else:  # len(past_positions) == 1
        past_position = past_positions.pop()

        # Check movement distance (fast movement is only allowed when a cell is launched into its death)
        distance_moved_um_per_m = past_position.distance_um(position, resolution) / resolution.time_point_interval_m
        if distance_moved_um_per_m > warning_limits.max_distance_moved_um_per_min:
            if linking_markers.is_live(position_data, position):
                return Error.MOVED_TOO_FAST

        # Check link probability
        link_probability = link_data.get_link_data(position, past_position, data_name="link_probability")
        if link_probability is not None and link_probability < warning_limits.min_probability\
                and linking_markers.is_live(position_data, position):
            return Error.LOW_LINK_SCORE

    return None


def _has_high_division_probability_hereafter(links: Links, position_data: PositionData, warning_limits: WarningLimits,
                                             future_position: Position) -> bool:
    """Returns True if the cell has a high division probability in one or two time points. If the tracking data actually
    included a division after future_position, this method always returns True.
    """

    # Check division probability in next time point, to avoid showing warning multiple times in a row
    future_future_positions = links.find_futures(future_position)
    division_probability_next = position_data.get_position_data(future_position, 'division_probability')
    if division_probability_next is None:
        division_probability_next = 0

    if division_probability_next > 1 - warning_limits.min_probability:
        return True  # Already seen

    # Check division probability after this
    division_probability_next_next = 0
    if len(future_future_positions) == 1:
        division_probability_next_next = position_data.get_position_data(next(iter(future_future_positions)),
                                                                         'division_probability')
        if division_probability_next_next is None:
            division_probability_next_next = 0
        return division_probability_next_next > 1 - warning_limits.min_probability
    elif len(future_future_positions) >= 2:
        # We have seen a division! Assume probability of 1
        return True  # Will divide
    else:
        # Cell disappeared
        return False


def _get_volumes(position: Position, volume_lookup: PositionData,
                 next_position_getter: Callable[[Position], Optional[Position]], max_amount: int) -> Optional[float]:
    """Gets the mean volume over time, based on the given number of recorded volumes. If there aren't that many
    recorded volumes, then the it uses less. However, if there are less than 2 volumes recorded, None is returned, as in
    that case we don't have enough data to say anything useful."""
    volumes = list()
    while len(volumes) < max_amount:
        shape = linking_markers.get_shape(volume_lookup, position)
        if shape.is_unknown():
            break
        volumes.append(shape.volume())

        position = next_position_getter(position)
        if position is None:
            break
    if len(volumes) < 2:
        return None  # Too few data points for an average
    return sum(volumes) / len(volumes)


def find_errors_in_positions_links_and_all_dividing_cells(experiment: Experiment, *iterable: Position):
    """Checks all of the given positions and all dividing cells in the experiment for logical errors, like cell merges,
    cell dividing into three daughters, cells moving too fast, ect. The reason dividing cells are also checked is that
    otherwise it's not possible to detect when a young mother cell is no longer a young mother cell because far away in
    time some link changed."""
    positions = set()

    # Add given positions and links
    for position in iterable:
        positions.add(position)
        positions |= experiment.links.find_links_of(position)

    _find_errors_in_just_the_iterable(experiment, positions)
    find_errors_in_all_dividing_cells(experiment)


def find_errors_in_all_dividing_cells(experiment: Experiment):
    """Rechecks all mother and daughter cells to verify that the mothers aren't too young. Checking this accross the
    entire experiment is useful, as changes in links far away might affect the measured cell cycle length."""
    resolution = experiment.images.resolution()
    warning_limits = experiment.warning_limits

    for track in experiment.links.find_all_tracks():
        if len(track.get_next_tracks()) != 2:
            continue
        # Found a mother track!
        age = particle_age_finder.get_age_at_end_of_track(track)
        if age is not None and age * resolution.time_point_interval_h < warning_limits.min_time_between_divisions_h:
            return Error.YOUNG_MOTHER


def _find_errors_in_just_the_iterable(experiment: Experiment, iterable: Iterable[Position]):
    """Checks all positions in the given iterable for logical errors, like cell merges, cell dividing into three
    daughters, cells moving too fast, ect."""
    links = experiment.links
    position_data = experiment.position_data
    for position in iterable:
        error = get_error(experiment, position)
        linking_markers.set_error_marker(position_data, position, error)


def find_errors_in_just_these_positions(experiment: Experiment, *iterable: Position):
    """Checks all positions in the given iterable for logical errors, like cell merges, cell dividing into three
    daughters, cells moving too fast, ect."""
    _find_errors_in_just_the_iterable(experiment, iterable)

