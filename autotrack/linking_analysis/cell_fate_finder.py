from enum import Enum
from typing import Optional

from autotrack.core.experiment import Experiment
from autotrack.core.position import Position
from autotrack.linking_analysis import linking_markers
from autotrack.linking_analysis.linking_markers import EndMarker


class CellFateType(Enum):
    """Cells with either divide another time, will continue moving for a long time, or will die."""
    UNKNOWN = 0  # Used if we cannot look ahead enough time points to conclude that the cell is just moving.
    JUST_MOVING = 1
    WILL_DIVIDE = 2
    WILL_DIE = 3


class CellFate:
    type: CellFateType  # What happened to the cell
    time_points_remaining: Optional[int]  # In how many time points the cell will die/divide. Only set if WILL_DIVIDE or WILL_DIE.

    def __init__(self, type: CellFateType, time_points_remaining: Optional[int]):
        self.type = type
        self.time_points_remaining = time_points_remaining


def get_fate(experiment: Experiment, position: Position) -> CellFate:
    """Checks if a cell will undergo a division later in the experiment. Returns None if not sure, because we are near
    the end of the experiment. max_time_point_number is the number of the last time point in the experiment."""
    starting_position = position
    max_time_point_number = position.time_point_number() + experiment.division_lookahead_time_points
    links = experiment.links

    while True:
        next_positions = links.find_futures(position)
        if len(next_positions) == 0 and linking_markers.get_track_end_marker(links, position) == EndMarker.DEAD:
            # Actual cell death
            time_points_remaining = position.time_point_number() - starting_position.time_point_number()
            return CellFate(CellFateType.WILL_DIE, time_points_remaining)

        if len(next_positions) == 0 or linking_markers.get_error_marker(links, position) is not None:
            # Stop following the cell: it has an error, or it moved out of view

            # Not an actual cell death, but cell moved out of view or the experiment ended
            if position.time_point_number() > max_time_point_number:
                # We followed the cell for a long enough time, conclude that cell just moved
                return CellFate(CellFateType.JUST_MOVING, None)

            # We didn't follow the cell for a long enough time, we cannot say anything about the fate of this cell
            return CellFate(CellFateType.UNKNOWN, None)
        if len(next_positions) == 1:
            # Simple movement, go to next time point
            position = next_positions.pop()
            continue
        if len(next_positions) >= 2:
            # Found the next division
            time_points_remaining = position.time_point_number() - starting_position.time_point_number()
            return CellFate(CellFateType.WILL_DIVIDE, time_points_remaining)
