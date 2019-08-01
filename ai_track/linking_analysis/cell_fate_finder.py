from enum import Enum
from typing import Optional

from ai_track.core.experiment import Experiment
from ai_track.core.links import Links
from ai_track.core.position import Position
from ai_track.linking_analysis import linking_markers
from ai_track.linking_analysis.linking_markers import EndMarker


class CellFateType(Enum):
    """Cells with either divide another time, will continue moving for a long time, or will die."""
    UNKNOWN = 0  # Used if we cannot look ahead enough time points to conclude that the cell is just moving.
    JUST_MOVING = 1
    WILL_DIVIDE = 2
    WILL_DIE = 3
    WILL_SHED = 4


class CellFate:
    type: CellFateType  # What happened to the cell
    time_points_remaining: Optional[int]  # In how many time points the cell will die/divide. Only set if WILL_DIVIDE or WILL_DIE.

    def __init__(self, type: CellFateType, time_points_remaining: Optional[int]):
        self.type = type
        self.time_points_remaining = time_points_remaining

    def __repr__(self) -> str:
        return "CellFate(" + str(self.type) + ", " + repr(self.time_points_remaining) + ")"


def get_fate(experiment: Experiment, position: Position) -> CellFate:
    """Checks if a cell will undergo a division later in the experiment. Returns None if not sure, because we are near
    the end of the experiment. max_time_point_number is the number of the last time point in the experiment."""
    max_time_point_number = position.time_point_number() + experiment.division_lookahead_time_points
    links = experiment.links
    track = links.get_track(position)
    if track is None:
        # No links found for this position
        return CellFate(CellFateType.UNKNOWN, None)

    next_tracks = track.get_next_tracks()
    if len(next_tracks) == 0:
        marker = linking_markers.get_track_end_marker(links, track.find_last_position())
        if marker == EndMarker.DEAD:
            # Actual cell death
            time_points_remaining = track.max_time_point_number() - position.time_point_number()
            return CellFate(CellFateType.WILL_DIE, time_points_remaining)
        elif marker == EndMarker.SHED:
            # Cell shedding
            time_points_remaining = track.max_time_point_number() - position.time_point_number()
            return CellFate(CellFateType.WILL_SHED, time_points_remaining)
        elif track.max_time_point_number() > max_time_point_number:
            # No idea what happened, but we followed the cell for a long enough time, conclude that cell just moved
            return CellFate(CellFateType.JUST_MOVING, None)

        # We didn't follow the cell for a long enough time, we cannot say anything about the fate of this cell
        return CellFate(CellFateType.UNKNOWN, None)
    elif len(next_tracks) >= 2:
        # Found the next division
        time_points_remaining = track.max_time_point_number() - position.time_point_number()
        return CellFate(CellFateType.WILL_DIVIDE, time_points_remaining)
    else:
        raise ValueError("len(next_tracks) == 1, this should be impossible")

