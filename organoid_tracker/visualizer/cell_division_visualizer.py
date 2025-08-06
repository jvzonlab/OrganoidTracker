from typing import List, Dict, Any, Tuple, Set

import numpy
from matplotlib.backend_bases import KeyEvent

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.link_data import LinkData
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.core.typing import DataType
from organoid_tracker.gui.undo_redo import UndoableAction
from organoid_tracker.gui.window import Window, DisplaySettings
from organoid_tracker.linking import cell_division_finder
from organoid_tracker.linking_analysis import cell_error_finder
from organoid_tracker.visualizer.position_list_visualizer import PositionListVisualizer


class _DeleteDivisionLinksAction(UndoableAction):
    """Inserts multiple links. Will not interpolate any positions."""
    position_pairs: List[Tuple[Position, Position, Dict[str, DataType]]]

    def __init__(self, link_data: LinkData, position_pairs: List[Tuple[Position, Position]]):
        self.position_pairs = list()
        for position_a, position_b in position_pairs:
            self.position_pairs.append((position_a, position_b,
                                        dict(link_data.find_all_data_of_link(position_a, position_b))))

    def do(self, experiment: Experiment) -> str:
        for position1, position2, data in self.position_pairs:
            experiment.links.remove_link(position1, position2)

        # Redo the error checking for the involved positions
        cell_error_finder.find_errors_in_positions_links_and_all_dividing_cells(experiment, self._get_involved_positions())

        return f"Removed the division by removing {len(self.position_pairs)} link(s)"

    def _get_involved_positions(self) -> Set[Position]:
        involved_positions = set()
        for position1, position2, _ in self.position_pairs:
            involved_positions.add(position1)
            involved_positions.add(position2)
        return involved_positions

    def undo(self, experiment: Experiment) -> str:
        for position1, position2, data in self.position_pairs:
            experiment.links.add_link(position1, position2)
            for data_key, data_value in data.items():
                experiment.link_data.set_link_data(position1, position2, data_key, data_value)

        # Redo the error checking for the involved positions
        cell_error_finder.find_errors_in_positions_links_and_all_dividing_cells(experiment, self._get_involved_positions())

        return f"Restored the division by inserting {len(self.position_pairs)} link(s)"


def _get_mothers(experiment: Experiment) -> List[Position]:
    return list(cell_division_finder.find_mothers(experiment.links))


class CellDivisionVisualizer(PositionListVisualizer):
    """Shows cells that are about to divide.
    Use the left/right arrow keys to move to the next cell division.
    Press Delete or Backspace to delete the division. This will remove the links to all daughter cells except the closest one.
    Press M to exit this view."""

    def __init__(self, window: Window):
        super().__init__(window, all_positions=_get_mothers(window.get_experiment()))

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "Edit//Delete-Delete this cell division [Delete]": self._delete_division,
        }

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
        elif event.key == "backspace":
            self._delete_division()
        else:
            super()._on_key_press(event)

    def _delete_division(self):
        position_list = self._position_list
        position_index = self._current_position_index
        if position_index < 0 or position_index >= len(position_list):
            return
        position = position_list[position_index]

        # Get the resolution, with a fallback to pixels if not available
        try:
            resolution = self._experiment.images.resolution()
        except UserError:
            resolution = ImageResolution.PIXELS

        # Find the closest daughter to the mother
        daughters = list(self._experiment.links.find_futures(position))
        if len(daughters) < 2:
            return
        distances = [position.distance_squared(daughter, resolution) for daughter in daughters]
        closest_daughter = daughters[numpy.argmin(distances)]

        # Remove the division by deleting the links to all daughters except the closest one
        position_pairs_to_remove = list()
        for daughter in daughters:
            if daughter != closest_daughter:
                position_pairs_to_remove.append((position, daughter))
        action = _DeleteDivisionLinksAction(self._experiment.link_data, position_pairs_to_remove)
        self._window.get_undo_redo().do(action, self._experiment)

        # Update the position list to remove the mother position
        del self._position_list[position_index]
        if position_index >= len(self._position_list):
            self._current_position_index = len(self._position_list) - 1
        self.draw_view()
        self.update_status("Deleted division of " + str(position) + ". Press M to exit this view.")
