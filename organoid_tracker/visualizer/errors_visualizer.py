from typing import List, Optional, Dict, Any, Tuple

from matplotlib.backend_bases import KeyEvent

from organoid_tracker.core import TimePoint
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.linking_analysis import linking_markers, lineage_error_finder, cell_error_finder
from organoid_tracker.linking_analysis.lineage_error_finder import LineageWithErrors
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.position_list_visualizer import PositionListVisualizer


class ErrorsVisualizer(PositionListVisualizer):
    """Shows all errors and warnings in the experiment.
    Press Left/Right to view the previous/next error.
    Press Delete to delete (suppress) the shown error.
    Press C to change (fix) the data and press E to exit this view.
    """

    _problematic_lineages: List[LineageWithErrors]
    _current_lineage_index: int = -1
    _total_number_of_warnings: int

    def __init__(self, window: Window, start_position: Optional[Position]):
        experiment = window.get_experiment()
        links = experiment.links
        position_data = experiment.position_data

        crumb_positions = set()
        if start_position is not None:
            crumb_positions.add(start_position)
        if self._get_last_position() is not None:
            crumb_positions.add(self._get_last_position())
        display_settings = window.display_settings
        self._problematic_lineages = lineage_error_finder.get_problematic_lineages(links, position_data, crumb_positions,
                                             min_time_point=display_settings.error_correction_min_time_point,
                                             max_time_point=display_settings.error_correction_max_time_point)
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
            "Edit//Error settings-Change minimum allowed time in between divisions...": self._change_min_division_time,
            "Edit//Error settings-Change maximum allowed movement per minute...": self._change_max_distance,
            "Edit//Correction settings-Change minimum time point for correction...": self._change_min_time_point,
            "Edit//Correction settings-Change maximum time point for correction...": self._change_max_time_point,
            "Navigate//Lineage-Next lineage [Up]": self.__goto_next_lineage,
            "Navigate//Lineage-Previous lineage [Down]": self.__goto_previous_lineage
        }

    def _change_min_division_time(self):
        old_time = self._experiment.warning_limits.min_time_between_divisions_h
        new_time = dialog.prompt_float("Minimum allowed time in between divisions",
                                       "What is the minimum amount of time that should pass before a cell divides"
                                       " again?\nCells this violate this will be flagged. Please specify the amount in"
                                       " hours.", minimum=0, default=old_time)
        if new_time is None:
            return
        if new_time != old_time:
            self._experiment.warning_limits.min_time_between_divisions_h = new_time
            self._recheck_errors()

    def _change_max_distance(self):
        old_distance = self._experiment.warning_limits.max_distance_moved_um_per_min
        new_distance = dialog.prompt_float("Maximum allowed distance per time point",
                                       "What is the maximum distance in micrometers that cells can travel per minute?\n"
                                       "Cells that go faster will be flagged.",
                                       minimum=0, default=old_distance)
        if new_distance is None:
            return
        if new_distance != old_distance:
            self._experiment.warning_limits.max_distance_moved_um_per_min = new_distance
            self._recheck_errors()

    def _change_min_time_point(self):
        # Find out the bounds
        first_time_point_number, last_time_point_number = self._get_error_checking_time_points()

        # Find the current value
        current_min_time_point = self.get_window().display_settings.error_correction_min_time_point
        if current_min_time_point is None:
            current_min_time_point = TimePoint(first_time_point_number)

        # Update new value
        answer = dialog.prompt_int("Error checking", "At which time point should error checking start?",
                                   default=current_min_time_point.time_point_number(), minimum=first_time_point_number,
                                   maximum=last_time_point_number)
        if answer is None:
            return
        new_min_time_point = None if answer <= first_time_point_number else TimePoint(answer)
        self.get_window().display_settings.error_correction_min_time_point = new_min_time_point
        self._recalculate_errors()
        self.update_status("Now checking errors starting at time point " + str(answer) + ".")

    def _change_max_time_point(self):
        # Find out the bounds
        first_time_point_number, last_time_point_number = self._get_error_checking_time_points()

        # Find the current value
        current_max_time_point = self.get_window().display_settings.error_correction_max_time_point
        if current_max_time_point is None:
            current_max_time_point = TimePoint(last_time_point_number)

        # Update new value
        answer = dialog.prompt_int("Error checking", "Up to and including which time point should error checking"
                                   " continue?", default=current_max_time_point.time_point_number(),
                                   minimum=first_time_point_number, maximum=last_time_point_number)
        if answer is None:
            return
        new_max_time_point = None if answer >= last_time_point_number else TimePoint(answer)
        self.get_window().display_settings.error_correction_max_time_point = new_max_time_point
        self._recalculate_errors()
        self.update_status("Now checking errors up too and including time point " + str(answer) + ".")

    def _get_error_checking_time_points(self) -> Tuple[int, int]:
        """Gets the min and max time point used for error checking. If no positions are loaded, arbitrary values are
        returned."""
        first_time_point_number = self._experiment.positions.first_time_point_number()
        if first_time_point_number is None:
            first_time_point_number = 0
        last_time_point_number = self._experiment.positions.last_time_point_number()
        if last_time_point_number is None:
            last_time_point_number = 9999
        return first_time_point_number, last_time_point_number

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
               f"  ({self._total_number_of_warnings} warnings in total)" +\
               "\n" + message + "\n" + str(position)

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
            data_editor = LinkAndPositionEditor(self._window, selected_position=viewed_position)
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

