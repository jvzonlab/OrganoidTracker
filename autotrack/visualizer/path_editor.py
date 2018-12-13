from typing import Optional

from matplotlib.backend_bases import KeyEvent, MouseEvent

from autotrack.core import TimePoint
from autotrack.core.experiment import Experiment
from autotrack.core.particles import Particle
from autotrack.core.data_axis import DataAxis
from autotrack.gui.undo_redo import UndoableAction, ReversedAction
from autotrack.gui.window import Window
from autotrack.visualizer import activate, DisplaySettings
from autotrack.visualizer.abstract_editor import AbstractEditor


class _AddPathAction(UndoableAction):

    _path: DataAxis
    _time_point: TimePoint

    def __init__(self, path: DataAxis, time_point: TimePoint):
        self._path = path
        self._time_point = time_point

    def do(self, experiment: Experiment) -> str:
        experiment.data_axes.add_data_axis(self._time_point, self._path)
        self._path.update_offset_for_particles(experiment.particles.of_time_point(self._time_point))
        return "Added path to time point " + str(self._time_point.time_point_number())

    def undo(self, experiment: Experiment):
        experiment.data_axes.remove_data_axis(self._time_point, self._path)
        return "Removed path in time point " + str(self._time_point.time_point_number())


class _AddPointAction(UndoableAction):
    _path: DataAxis
    _new_point: Particle

    def __init__(self, path: DataAxis, new_point: Particle):
        self._path = path
        self._new_point = new_point

    def do(self, experiment: Experiment) -> str:
        self._path.add_point(self._new_point.x, self._new_point.y, self._new_point.z)
        self._path.update_offset_for_particles(experiment.particles.of_time_point(self._new_point.time_point()))
        return f"Added point at ({self._new_point.x:.0f}, {self._new_point.y:.0f}) to selected path"

    def undo(self, experiment: Experiment) -> str:
        self._path.remove_point(self._new_point.x, self._new_point.y)
        return f"Removed point at ({self._new_point.x:.0f}, {self._new_point.y:.0f}) from path"


class PathEditor(AbstractEditor):
    """Editor for paths. Double-click to (de)select a path.
    Press Insert to start a new path if no path is selected.
    Press Delete to delete the whole selected path."""

    _selected_path: Optional[DataAxis]
    _selected_path_time_point: Optional[TimePoint]

    def __init__(self, window: Window, *, time_point_number: Optional[int] = None, z: int = 14,
                 display_settings: DisplaySettings = None):
        super().__init__(window, time_point_number=time_point_number, z=z, display_settings=display_settings)
        self._selected_path = None
        self._selected_path_time_point = None

    def _select_path(self, path: Optional[DataAxis]):
        """(De)selects a path for the current time point. Make sure to redraw after calling this method."""
        if path is None:
            self._selected_path = None
            self._selected_path_time_point = None
        else:
            self._selected_path = path
            self._selected_path_time_point = self._time_point

    def _on_mouse_click(self, event: MouseEvent):
        if event.dblclick:
            # Select path
            position = Particle(event.xdata, event.ydata, self._z).with_time_point(self._time_point)
            path_position = self._experiment.data_axes.to_position_on_axis(position)
            if path_position is None or path_position.distance > 10 or path_position.axis == self._selected_path:
                self._select_path(None)
            else:
                self._select_path(path_position.axis)
            self.draw_view()

    def _get_figure_title(self) -> str:
        return "Editing axes in time point " + str(self._time_point.time_point_number()) + "    (z=" + str(self._z) + ")"

    def _get_window_title(self) -> str:
        return "Manual data editing"

    def _get_selected_path_of_current_time_point(self) -> Optional[DataAxis]:
        if self._selected_path is None:
            return None
        if not self._experiment.data_axes.exists(self._selected_path, self._selected_path_time_point):
            self._selected_path = None  # Path was deleted, remove selection
            return None
        if self._selected_path_time_point != self._time_point:
            return None  # Don't draw paths of other time points
        return self._selected_path

    def _draw_extra(self):
        # Highlight the selected path
        path = self._get_selected_path_of_current_time_point()
        if path is None:
            return
        self._ax.plot(*path.get_points_2d(), marker=path.get_direction_marker(), color="white", linewidth=4,
                      markerfacecolor="white", markeredgecolor="black", markersize=12)

    def _on_key_press(self, event: KeyEvent):
        if event.key == "insert":
            selected_path = self._get_selected_path_of_current_time_point()
            if selected_path is None:
                # Time for a new path
                path = DataAxis()
                path.add_point(event.xdata, event.ydata, self._z)
                self._select_path(path)
                self._perform_action(_AddPathAction(path, self._time_point))
            else:
                # Can modify existing path
                point = Particle(event.xdata, event.ydata, self._z).with_time_point(self._time_point)
                self._perform_action(_AddPointAction(selected_path, point))
        elif event.key == "delete":
            selected_path = self._get_selected_path_of_current_time_point()
            if selected_path is None:
                self.update_status("No path selected - cannot delete anything.")
                return
            else:
                self._select_path(None)
                self._perform_action(ReversedAction(_AddPathAction(selected_path, self._time_point)))
        elif event.key == "c":
            if self._selected_path is None:
                self.update_status("No path selected, cannot copy anything")
                return
            if self._selected_path_time_point == self._time_point:
                self.update_status("Cannot copy a path to the same time point")
                return
            copied_path = self._selected_path.copy()
            self._select_path(copied_path)
            self._perform_action(_AddPathAction(copied_path, self._time_point))
        else:
            super()._on_key_press(event)

    def _exit_view(self):
        from autotrack.visualizer.link_and_position_editor import LinkAndPositionEditor
        data_editor = LinkAndPositionEditor(self._window,
                                            time_point_number=self._time_point.time_point_number(),
                                            z=self._z, display_settings=self._display_settings)
        activate(data_editor)
