from typing import Optional, Iterable, Callable, Tuple

from organoid_tracker.core.link_data import LinkData
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.score import Score, ScoreCollection, Family
from organoid_tracker.linking import cell_division_finder
from organoid_tracker.linking_analysis import linking_markers, particle_age_finder
from organoid_tracker.linking_analysis.errors import Error


def find_errors_in_experiment(experiment: Experiment, marginalization = False) -> Tuple[int, int]:
    """Adds errors for all logical inconsistencies in the graph, like cells that spawn out of nowhere, cells that
    merge together and cells that have three or more daughters.
    Returns the amount of errors (excluding positions without links) and the number of positions without links."""
    position_data = experiment.position_data
    links = experiment.links

    warning_count = 0
    no_links_count = 0
    for position in experiment.positions:
        error = get_error(experiment, position, marginalization=marginalization)
        linking_markers.set_error_marker(position_data, position, error)
        if error is not None:
            if links.contains_position(position):
                warning_count += 1
            else:  # It's just a position without links
                no_links_count += 1
    return warning_count, no_links_count


def get_error(experiment: Experiment, position: Position, marginalization = False) -> Optional[Error]:
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
        if position_data.get_position_data(position, 'division_probability') is None:
            return Error.LOW_MOTHER_SCORE
        elif position_data.get_position_data(position, 'division_probability') < warning_limits.min_probability:
            return Error.LOW_MOTHER_SCORE

        age = particle_age_finder.get_age(links, position)
        if age is not None and age * resolution.time_point_interval_h < warning_limits.min_time_between_divisions_h:
            return Error.YOUNG_MOTHER

    past_positions = links.find_pasts(position)
    if len(past_positions) == 0:
        if position.time_point_number() > positions.first_time_point_number() \
                and linking_markers.get_track_start_marker(position_data, position) is None:
            return Error.NO_PAST_POSITION
    elif len(past_positions) >= 2:
        return Error.CELL_MERGE

    elif marginalization:  # len(past_positions) == 1
        past_position = past_positions.pop()
        # Check marginalized link probability
        link_probability = link_data.get_link_data(past_position, position, data_name="marginal_probability")
        if link_probability is not None and link_probability < warning_limits.min_marginal_probability\
                and linking_markers.is_live(position_data, position):
            print('low_link_score')
            return Error.LOW_LINK_SCORE
    else:
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


def _get_highest_mother_score(scores: ScoreCollection, position: Position) -> Optional[Score]:
    highest_score = None
    highest_score_num = -999
    for scored_family in scores.of_time_point(position.time_point()):
        score = scored_family.score
        score_num = score.total()
        if score_num > highest_score_num:
            highest_score = score
            highest_score_num = score_num
    return highest_score


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

