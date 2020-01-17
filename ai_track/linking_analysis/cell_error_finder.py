from typing import Optional, Iterable, Callable

import numpy

from ai_track.core import TimePoint
from ai_track.core.experiment import Experiment
from ai_track.core.links import Links
from ai_track.core.position_collection import PositionCollection
from ai_track.core.position import Position
from ai_track.core.position_data import PositionData
from ai_track.core.resolution import ImageResolution
from ai_track.core.score import Score, ScoreCollection, Family
from ai_track.linking import cell_division_finder
from ai_track.linking_analysis import linking_markers, particle_age_finder
from ai_track.linking_analysis.errors import Error
from ai_track.linking_analysis.linking_markers import EndMarker


def find_errors_in_experiment(experiment: Experiment) -> int:
    """Adds errors for all logical inconsistencies in the graph, like cells that spawn out of nowhere, cells that
    merge together and cells that have three or more daughters. Returns the amount of errors."""
    links = experiment.links
    scores = experiment.scores
    positions = experiment.positions
    resolution = experiment.images.resolution()
    position_data = experiment.position_data

    count = 0
    for position in experiment.positions:
        error = get_error(links, position, scores, positions, position_data, resolution)
        linking_markers.set_error_marker(links, position, error)
        if error is not None:
            count += 1
    return count


def get_error(links: Links, position: Position, scores: ScoreCollection, positions: PositionCollection,
              position_data: PositionData, resolution: ImageResolution) -> Optional[Error]:
    if linking_markers.is_uncertain(position_data, position):
        return Error.UNCERTAIN_POSITION

    if not links.has_links():
        return  # Don't attempt to find other errors

    future_positions = links.find_futures(position)
    if len(future_positions) > 2:
        return Error.TOO_MANY_DAUGHTER_CELLS
    elif len(future_positions) == 0 \
            and position.time_point_number() < positions.last_time_point_number() \
            and linking_markers.get_track_end_marker(links, position) is None:
        return Error.NO_FUTURE_POSITION
    elif len(future_positions) == 2:
        if scores.has_family_scores():  # Use family scores
            score = scores.of_family(Family(position, *future_positions))
            if score is None or score.is_unlikely_mother():
                return Error.LOW_MOTHER_SCORE
        else:  # Use mother scores
            score = linking_markers.get_mother_score(links, position)
            if score <= 0:
                return Error.LOW_MOTHER_SCORE
        age = particle_age_finder.get_age(links, position)
        if age is not None and age * resolution.time_point_interval_h <= 10:
            return Error.YOUNG_MOTHER

    past_positions = links.find_pasts(position)
    if len(past_positions) == 0:
        if position.time_point_number() > positions.first_time_point_number() \
                and linking_markers.get_track_start_marker(links, position) is None:
            return Error.NO_PAST_POSITION
    elif len(past_positions) >= 2:
        return Error.CELL_MERGE
    else:  # len(past_positions) == 1
        # Check cell size
        past_position = past_positions.pop()
        future_positions_of_past_position = links.find_futures(past_position)
        shape = positions.get_shape(position)
        past_shape = positions.get_shape(past_position)
        if shape.is_failed() and len(future_positions) != 2:
            return Error.FAILED_SHAPE  # Gaussian fit failed, can happen for dividing cells, but should not happen otherwise
        elif not shape.is_unknown() and len(future_positions_of_past_position) == 1:
            if not past_shape.is_unknown() and past_shape.volume() / (shape.volume() + 0.0001) > 2:
                # Found a sudden decrease in volume. Check averages to see if it is an outlier, or something real

                # Compare volumes of last 5 and next 5 positions
                volume_last_five = _get_volumes(past_position, positions, links.find_single_past, 5)
                volume_next_five = _get_volumes(position, positions, links.find_single_future, 5)
                if volume_last_five is not None and volume_next_five is not None \
                        and volume_last_five / (volume_next_five + 0.0001) > 2:
                    return Error.SHRUNK_A_LOT

        # Check movement distance (fast movement is only allowed when a cell is launched into its death)
        if past_position.distance_um(position, resolution) > 10:
            end_marker = linking_markers.get_track_end_marker(links, position)
            if end_marker != EndMarker.DEAD and end_marker != EndMarker.SHED:
                return Error.MOVED_TOO_FAST
    return None


def _get_volumes(position: Position, volume_lookup: PositionCollection,
                 next_position_getter: Callable[[Position], Optional[Position]], max_amount: int) -> Optional[float]:
    """Gets the mean volume over time, based on the given number of recorded volumes. If there aren't that many
    recorded volumes, then the it uses less. However, if there are less than 2 volumes recorded, None is returned, as in
    that case we don't have enough data to say anything useful."""
    volumes = list()
    while len(volumes) < max_amount:
        shape = volume_lookup.get_shape(position)
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

    # Add mother and daughter cells
    for family in cell_division_finder.find_families(experiment.links):
        positions.add(family.mother)
        positions |= family.daughters

    _find_errors_in_just_the_iterable(experiment, positions)


def find_errors_in_all_dividing_cells(experiment: Experiment):
    """Rechecks all mother and daughter cells for logical errors. Rechecking this is useful, because the errors in those
    positions can be influenced by changes far away."""
    positions = set()
    for family in cell_division_finder.find_families(experiment.links):
        positions.add(family.mother)
        positions |= family.daughters

    _find_errors_in_just_the_iterable(experiment, positions)


def _find_errors_in_just_the_iterable(experiment: Experiment, iterable: Iterable[Position]):
    """Checks all positions in the given iterable for logical errors, like cell merges, cell dividing into three
    daughters, cells moving too fast, ect."""
    links = experiment.links
    for position in iterable:
        error = get_error(links, position, experiment.scores, experiment.positions, experiment.position_data,
                          experiment.images.resolution())
        linking_markers.set_error_marker(links, position, error)


def find_errors_in_just_these_positions(experiment: Experiment, *iterable: Position):
    """Checks all positions in the given iterable for logical errors, like cell merges, cell dividing into three
    daughters, cells moving too fast, ect."""
    _find_errors_in_just_the_iterable(experiment, iterable)

