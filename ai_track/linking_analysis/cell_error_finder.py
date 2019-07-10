from typing import Optional, Iterable

from ai_track.core import TimePoint
from ai_track.core.experiment import Experiment
from ai_track.core.links import Links
from ai_track.core.position_collection import PositionCollection
from ai_track.core.position import Position
from ai_track.core.resolution import ImageResolution
from ai_track.core.score import Score, ScoreCollection, Family
from ai_track.linking_analysis import linking_markers, particle_age_finder
from ai_track.linking_analysis.errors import Error
from ai_track.linking_analysis.linking_markers import EndMarker


def apply(experiment: Experiment):
    """Adds errors for all logical inconsistencies in the graph, like cells that spawn out of nowhere, cells that
    merge together and cells that have three or more daughters."""
    links = experiment.links
    scores = experiment.scores
    positions = experiment.positions
    resolution = experiment.images.resolution()
    for position in links.find_all_positions():
        error = get_error(links, position, scores, positions, resolution)
        linking_markers.set_error_marker(links, position, error)


def get_error(links: Links, position: Position, scores: ScoreCollection, positions: PositionCollection,
              resolution: ImageResolution) -> Optional[Error]:
    future_positions = links.find_futures(position)
    if len(future_positions) > 2:
        return Error.TOO_MANY_DAUGHTER_CELLS
    elif len(future_positions) == 0 \
            and position.time_point_number() < positions.last_time_point_number() \
            and linking_markers.get_track_end_marker(links, position) is None:
        return Error.NO_FUTURE_POSITION
    elif len(future_positions) == 2:
        if scores.has_scores():
            score = scores.of_family(Family(position, *future_positions))
            if score is None or score.is_unlikely_mother():
                return Error.LOW_MOTHER_SCORE
        age = particle_age_finder.get_age(links, position)
        if age is not None and age < 5:
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
        if not shape.is_unknown() and len(future_positions_of_past_position) == 1:
            past_shape = positions.get_shape(past_position)
            if not past_shape.is_unknown() and past_shape.volume() / (shape.volume() + 0.0001) > 3:
                return Error.SHRUNK_A_LOT

        # Check movement distance (fast movement is only allowed when a cell is launched into its death)
        if past_position.distance_um(position, resolution) > 10:
            end_marker = linking_markers.get_track_end_marker(links, position)
            if end_marker != EndMarker.DEAD and end_marker != EndMarker.SHED:
                return Error.MOVED_TOO_FAST
    return None


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


def apply_on_time_point(experiment: Experiment, time_point: TimePoint):
    """Checks all positions in the given time point for logical errors, like cell merges, cell dividing into three
    daughters, cells moving too fast, ect."""
    apply_on_iterable(experiment, experiment.positions.of_time_point(time_point))


def apply_on(experiment: Experiment, *iterable: Position):
    """Checks all of the given positions for logical errors, like cell merges, cell dividing into three
    daughters, cells moving too fast, ect."""
    apply_on_iterable(experiment, iterable)


def apply_on_iterable(experiment: Experiment, iterable: Iterable[Position]):
    """Checks all positions in the given iterable for logical errors, like cell merges, cell dividing into three
    daughters, cells moving too fast, ect."""
    links = experiment.links
    for position in iterable:
        error = get_error(links, position, experiment.scores, experiment.positions, experiment.images.resolution())
        linking_markers.set_error_marker(links, position, error)
