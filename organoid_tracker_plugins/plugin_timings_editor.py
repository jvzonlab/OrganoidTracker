from typing import Dict, Any, Optional, List, Tuple

import numpy
from matplotlib.backend_bases import MouseEvent, MouseButton

from organoid_tracker.core import TimePoint, UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.resolution import ImageTimings
from organoid_tracker.gui import dialog
from organoid_tracker.gui.dialog import DefaultOption
from organoid_tracker.gui.undo_redo import UndoableAction
from organoid_tracker.gui.window import Window
from organoid_tracker.visualizer import Visualizer


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Edit//Experiment-Set image timings...": lambda: _set_image_timings(window)
    }


def _set_image_timings(window: Window):
    # First, check if timings are available (ask the user for one if not)
    if not window.get_experiment().images.has_timings():
        time_resolution_m = dialog.prompt_float("Time resolution", "No time resolution has been set yet. "
                                                                   "Please provide a time resolution in minutes:",
                                                minimum=0.0001, maximum=100000, decimals=3)
        if time_resolution_m is None:
            return
        window.perform_data_action(_ReplaceImageTimingsAction(window, None,
                                                              ImageTimings.contant_timing(time_resolution_m)))
        answer = dialog.prompt_options("Time resolution set", f"A constant time resolution has now been set of"
                                                              f" {time_resolution_m:.2f} minutes.",
                                       option_1="Set variable time resolution...", option_default=DefaultOption.OK)
        if answer != 1:
            return  # The user has chosen to not set a variable time resolution

    # Then, open the visualizer
    dialog.popup_visualizer(window, _TimingsVisualizer, size_cm=(30, 5))


class _ReplaceImageTimingsAction(UndoableAction):
    _old_timings: Optional[ImageTimings]
    _new_timings: ImageTimings

    def __init__(self, window: Window, old_timings: Optional[ImageTimings], new_timings: ImageTimings):
        self._old_timings = old_timings
        self._new_timings = new_timings

    def do(self, experiment: Experiment):
        experiment.images.set_timings(self._new_timings)
        return "Changed the time resolution of the selected range."

    def undo(self, experiment: Experiment):
        experiment.images.set_timings(self._old_timings)
        return "Reverted the time resolution of the selected range."


class _TimingsVisualizer(Visualizer):
    """Shows a timeline, visualizing the time points of the experiment. The user can select a range of time points by
    selecting two points on the timeline (using clicking). The selected range can then be used to set a new time
    resolution."""

    _min_selected_time_h: Optional[float] = None
    _max_selected_time_h: Optional[float] = None

    _Y_LIMITS = (-2, 5)  # For the y-axis of the timeline

    def __init__(self, window: Window):
        super().__init__(window)

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            "Edit//Set overall time resolution...": self._set_overall_time_resolution,
            "Edit//Set time resolution of selection...": self._set_selection_time_resolution
        }

    def _set_overall_time_resolution(self):
        # Get the old timings
        old_timings = self._experiment.images.timings()

        # Prompt the user for the new time resolution
        new_resolution_m = dialog.prompt_float("Time resolution", "What should the new time resolution be"
                                                                  " (in minutes)?\n\nNote: this will overwrite any"
                                                                  " existing time resolution.",
                                               minimum=0.0001, maximum=100000, decimals=3)
        if new_resolution_m is None:
            return

        # Calculate the new timings
        new_timings = ImageTimings.contant_timing(new_resolution_m)

        # Select all time points for visual feedback
        first_time_point = self._experiment.first_time_point()
        last_time_point = self._experiment.last_time_point()
        if first_time_point is not None and last_time_point is not None:
            self._min_selected_time_h = new_timings.get_time_h_since_start(
                first_time_point) + new_timings.get_time_h_since_start(first_time_point - 1) / 2
            self._max_selected_time_h = new_timings.get_time_h_since_start(
                last_time_point) + new_timings.get_time_h_since_start(last_time_point + 1) / 2

        # Perform the action
        self.get_window().perform_data_action(_ReplaceImageTimingsAction(self.get_window(), old_timings, new_timings))

    def _set_selection_time_resolution(self):
        # Get the selected time points
        min_max_time_points = self._get_selected_time_points()
        if min_max_time_points is None:
            raise UserError("No time points", "No time points are selected. Use double-clicking to select"
                                              " a time range of at least two time points.")
        min_time_point, max_time_point = min_max_time_points

        # Get the old timings
        old_timings = self._experiment.images.timings()
        time_points = list(self._experiment.time_points())
        times_since_previous_m = [old_timings.get_time_m_since_previous(time_point)
                                  for time_point in time_points]

        # Prompt the user for the new time resolution
        new_resolution_m = dialog.prompt_float("Time resolution", "Enter the new time resolution (in minutes):",
                                               minimum=0.0001, maximum=100000, decimals=3)
        if new_resolution_m is None:
            return

        # Calculate the new timings
        for i, time_point in enumerate(time_points):
            if min_time_point < time_point < max_time_point + 1:
                times_since_previous_m[i] = new_resolution_m
        cumulative_timings_m = numpy.cumsum(times_since_previous_m)
        cumulative_timings_m -= cumulative_timings_m[0]
        new_timings = ImageTimings(time_points[0].time_point_number(), cumulative_timings_m)

        # Update location of selection for increased/decreased time resolution
        self._max_selected_time_h = (new_timings.get_time_h_since_start(
            max_time_point) + new_timings.get_time_h_since_start(max_time_point + 1)) / 2

        # Perform the action
        self.get_window().perform_data_action(_ReplaceImageTimingsAction(self.get_window(), old_timings, new_timings))

    def _is_selected(self, time_h: float) -> bool:
        if self._min_selected_time_h is None or self._max_selected_time_h is None:
            return False
        return self._min_selected_time_h <= time_h <= self._max_selected_time_h

    def draw_view(self):
        self._clear_axis()
        if not self._experiment.images.has_timings():
            self._fig.canvas.draw()
            return
        timings = self._experiment.images.timings()

        times_h_unselected = list()
        times_h_selected = list()
        for time_point in self._experiment.time_points():
            time_h = timings.get_time_h_since_start(time_point)
            if self._is_selected(time_h):
                times_h_selected.append(time_h)
            else:
                times_h_unselected.append(time_h)
            if time_point.time_point_number() % 20 == 0:
                self._ax.text(time_h, 0.8, str(time_point.time_point_number()), color="#000000", fontsize=8,
                              ha="center")

        if len(times_h_unselected) > 0:
            self._ax.scatter(times_h_unselected, [0] * len(times_h_unselected), color="#000000", marker="|")
        if len(times_h_selected) > 0:
            self._ax.scatter(times_h_selected, [0] * len(times_h_selected), color="#00ff00", marker="|")
        self._ax.set_yticks([])
        self._ax.set_xlabel("Time (h)")

        if self._min_selected_time_h is not None:
            self._ax.axvline(self._min_selected_time_h, color="#00ff00", linestyle="--")
        if self._max_selected_time_h is not None:
            self._ax.axvline(self._max_selected_time_h, color="#00ff00", linestyle="--")

        self._ax.set_ylim(*self._Y_LIMITS)
        self._fig.tight_layout()
        self._fig.canvas.draw()

    def _on_scroll(self, event: MouseEvent):
        old_xlim = self._ax.get_xlim()
        old_width = old_xlim[1] - old_xlim[0]
        shift_x = old_width / 10
        if event.button == "up":
            shift_x *= -1  # Inverse direction
        self._ax.set_xlim(old_xlim[0] + shift_x, old_xlim[1] + shift_x)
        self._ax.set_ylim(*self._Y_LIMITS)
        self._fig.canvas.draw()

    def _on_mouse_single_click(self, event: MouseEvent):
        if event.button == MouseButton.RIGHT:
            # Right click
            self._min_selected_time_h = None
            self._max_selected_time_h = None
            self.draw_view()
            self.update_status("Selection cleared.")
            return

        if event.button == MouseButton.LEFT and event.xdata is not None:
            # Selecting a time range
            if self._min_selected_time_h is None or \
                    (self._min_selected_time_h is not None and self._max_selected_time_h is not None):
                self._min_selected_time_h = event.xdata
                self._max_selected_time_h = None
                self.draw_view()
                self.update_status(
                    f"Clicked at {self._min_selected_time_h:.2f}h. Click a second point to complete"
                    f" the selection.")
            elif self._max_selected_time_h is None:
                old_x = self._min_selected_time_h
                new_x = event.xdata
                self._min_selected_time_h = min(old_x, new_x)
                self._max_selected_time_h = max(old_x, new_x)
                self.draw_view()
                self.update_status(f"Selected time range: {self._min_selected_time_h:.2f}h to"
                                   f" {self._max_selected_time_h:.2f}h.")
            return

        super()._on_mouse_single_click(event)

    def _get_selected_time_points(self) -> Optional[Tuple[TimePoint, TimePoint]]:
        """Gets the time points that are selected in the current view. Returns None if one or zero time points are
        selected."""
        time_point_numbers = list()

        timings = self._experiment.images.timings()
        for time_point in self._experiment.time_points():
            time_h = timings.get_time_h_since_start(time_point)
            if self._is_selected(time_h):
                time_point_numbers.append(time_point.time_point_number())

        if len(time_point_numbers) < 2:
            return None
        return TimePoint(time_point_numbers[0]), TimePoint(time_point_numbers[-1])
