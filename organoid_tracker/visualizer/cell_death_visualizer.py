from typing import List, Optional

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.gui.window import Window
from organoid_tracker.linking_analysis import linking_markers
from organoid_tracker.linking_analysis.errors import Error
from organoid_tracker.visualizer.position_list_visualizer import PositionListVisualizer


def _get_end_of_tracks(experiment: Experiment) -> List[Position]:
    return list(linking_markers.find_death_and_shed_positions(experiment.links))


class CellTrackEndVisualizer(PositionListVisualizer):
    """Shows cells that are about to divide.
    Use the left/right arrow keys to move to the next cell division.
    Type /exit to exit this view."""

    def __init__(self, window: Window, focus_position: Optional[Position]):
        super().__init__(window, chosen_position=focus_position,
                         all_positions=_get_end_of_tracks(window.get_experiment()))

    def get_message_no_positions(self):
        return "No ending cell tracks found. Is the linking data missing?"

    def get_message_press_right(self):
        return "No end of cell track found at mouse position." \
               "\nPress the right arrow key to view the first end of a cell track in the sample."

    def get_title(self, all_cells: List[Position], cell_index: int):
        cell = all_cells[cell_index]
        end_reason = self._get_end_cause(cell)

        return f"Track end {self._current_position_index + 1}/{len(self._position_list)}    ({end_reason})" \
               f"\n{cell}"

    def _get_end_cause(self, position: Position) -> str:
        links = self._experiment.links

        end_reason = linking_markers.get_track_end_marker(links, position)
        if end_reason is None:
            if linking_markers.is_error_suppressed(links, position, Error.NO_FUTURE_POSITION):
                return "analyzed, but no conclusion"
            return "not analyzed"
        return end_reason.get_display_name()
