from typing import List, Optional, Dict, Any, Tuple

from matplotlib.backend_bases import KeyEvent

from organoid_tracker.core import TimePoint
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog, option_choose_dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.linking_analysis import linking_markers, lineage_error_finder, cell_error_finder
from organoid_tracker.linking_analysis.errors import Error
from organoid_tracker.linking_analysis.lineage_error_finder import LineageWithErrors
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.position_list_visualizer import PositionListVisualizer


class ErrorsVisualizer(PositionListVisualizer):
    """Shows all errors and warnings in the experiment.
    Press Up/Down to go to the next/previous lineage with errors.
    Press Left/Right to go to the next/previous error in same lineage.
    Press Delete or Backspace to delete (suppress) the shown error.
    Press C to change (fix) the data and press E or Escape to exit this view.
    """

    _problematic_lineages: List[LineageWithErrors]
    _current_lineage_index: int = -1
    _total_number_of_warnings: int

    def __init__(self, window: Window, start_position: Optional[Position]):
        experiment = window.get_experiment()

        crumb_positions = set()
        if start_position is not None:
            crumb_positions.add(start_position)
        if self._get_last_position() is not None:
            crumb_positions.add(self._get_last_position())
        self._problematic_lineages = lineage_error_finder.get_problematic_lineages(experiment, crumb_positions)
        self._total_number_of_warnings = sum((len(lineage.errored_positions) for lineage in self._problematic_lineages))

        super().__init__(window, chosen_position=start_position, all_positions=[])

    def _show_closest_or_stored_position(self, position: Optional[Position]):
        if position is None:
            position = self._get_last_position()

        lineage_index = lineage_error_finder.find_lineage_index_with_crumb(self._problematic_lineages, position)
        if lineage_index is None:
            # Try again, now with last position
            position = self._get_last_position()
            lineage_index = lineage_error_finder.find_lineage_index_with_crumb(self._problematic_lineages, position)
            if lineage_index is None:
                return

        self._current_lineage_index = lineage_index  # Found the lineage the cell is in
        self._position_list = self._problematic_lineages[self._current_lineage_index].errored_positions
        try:
            # We even found the cell itself
            self._current_position_index = self._position_list.index(position)
        except ValueError:
            self._current_position_index = -1

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "Edit//Errors-Suppress this error [Delete]": self._suppress_error,
            "Edit//Errors-Recheck all errors": self._recheck_errors,
            "Navigate//Lineage-Next lineage [Up]": self.__goto_next_lineage,
            "Navigate//Lineage-Previous lineage [Down]": self.__goto_previous_lineage
        }

    def get_message_no_positions(self):
        if len(self._problematic_lineages) > 0:
            return "No warnings or errors found at position.\n" \
                   "Press the up arrow key to view the first lineage tree with warnings."
        return "No warnings or errors found. Hurray?"

    def get_message_press_right(self):
        return "No warnings or errors found in at position." \
               "\nPress the right arrow key to view the first warning in the lineage."

    def get_title(self, position_list: List[Position], current_position_index: int):
        position = position_list[current_position_index]
        error = linking_markers.get_error_marker(self._experiment.position_data, position)
        type = error.get_severity().name if error is not None else "Position"
        message = error.get_message() if error is not None else "Error was suppressed"

        return f"{type} {current_position_index + 1} / {len(position_list)} "\
               f" of lineage {self._current_lineage_index + 1} / {len(self._problematic_lineages)} " \
               f"  ({self._total_number_of_warnings} warnings in total) - time point {position.time_point_number()}" +\
               "\n" + message

    def _on_command(self, command: str) -> bool:
        if command == "recheck":
            self._recheck_errors()
            return True
        return super()._on_command(command)

    def _recheck_errors(self):
        cell_error_finder.find_errors_in_experiment(self._experiment)
        # Recalculate everything
        selected_position = None
        if 0 <= self._current_position_index < len(self._position_list):
            selected_position = self._position_list[self._current_position_index]
        activate(ErrorsVisualizer(self.get_window(), selected_position))
        self.update_status("Rechecked all cells in the experiment. "
                           "Please note that suppressed warnings remain suppressed.")

    def _recalculate_errors(self):
        selected_position = None
        if 0 <= self._current_position_index < len(self._position_list):
            selected_position = self._position_list[self._current_position_index]
        activate(ErrorsVisualizer(self.get_window(), selected_position))

    def _on_key_press(self, event: KeyEvent):
        if event.key == "e":
            self._exit_view()
        elif event.key == "backspace":
            self._suppress_error()  # The Delete key also works, but that key is registered through the menu
        else:
            super()._on_key_press(event)

    def __goto_previous_lineage(self):
        if len(self._problematic_lineages) < 1:
            return
        self._current_lineage_index -= 1
        if self._current_lineage_index < 0:
            self._current_lineage_index = len(self._problematic_lineages) - 1
        self._position_list = self._problematic_lineages[self._current_lineage_index].errored_positions
        self._current_position_index = 0
        self.draw_view()

    def __goto_next_lineage(self):
        if len(self._problematic_lineages) < 1:
            return
        self._current_lineage_index += 1
        if self._current_lineage_index >= len(self._problematic_lineages):
            self._current_lineage_index = 0
        self._position_list = self._problematic_lineages[self._current_lineage_index].errored_positions
        self._current_position_index = 0
        self.draw_view()

    def _exit_view(self):
        from organoid_tracker.visualizer.link_and_position_editor import LinkAndPositionEditor

        if self._current_position_index < 0 or self._current_position_index >= len(self._position_list):
            # Don't know where to go
            data_editor = LinkAndPositionEditor(self._window)
        else:
            viewed_position = self._position_list[self._current_position_index]
            data_editor = LinkAndPositionEditor(self._window, selected_positions=[viewed_position])
        activate(data_editor)

    def _suppress_error(self):
        if self._current_position_index < 0 or self._current_position_index >= len(self._position_list):
            return
        position = self._position_list[self._current_position_index]
        position_data = self._experiment.position_data
        error = linking_markers.get_error_marker(position_data, position)
        if error is None:
            self.update_status(f"Warning for {position} was already suppressed")
            return
        linking_markers.suppress_error_marker(position_data, position, error)
        self._total_number_of_warnings -= 1
        self.draw_view()
        self.update_status(f"Suppressed warning for {position}")

