from typing import List

from matplotlib.backend_bases import KeyEvent

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.gui.window import Window, DisplaySettings
from organoid_tracker.linking import cell_division_finder
from organoid_tracker.visualizer.position_list_visualizer import PositionListVisualizer


def _get_mothers(experiment: Experiment) -> List[Position]:
    return list(cell_division_finder.find_mothers(experiment.links))


class CellDivisionVisualizer(PositionListVisualizer):
    """Shows cells that are about to divide.
    Use the left/right arrow keys to move to the next cell division.
    Press M to exit this view."""

    def __init__(self, window: Window):
        super().__init__(window, all_positions=_get_mothers(window.get_experiment()))

    def get_message_no_positions(self):
        return "No mothers found. Is the linking data missing?"

    def get_message_press_right(self):
        return "No mother found at mouse position.\nPress the right arrow key to view the first mother in the sample."

    def get_title(self, all_cells: List[Position], cell_index: int):
        mother = all_cells[cell_index]
        recognized_str = ""
        return "Mother " + str(self._current_position_index + 1) + "/" + str(len(self._position_list))\
               + recognized_str + "\n" + str(mother)

    def _on_key_press(self, event: KeyEvent):
        if event.key == "m":
            self._exit_view()
        else:
            super()._on_key_press(event)
