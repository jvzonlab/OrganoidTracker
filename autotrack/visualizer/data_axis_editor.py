from typing import Optional, Dict, Any

from matplotlib.backend_bases import KeyEvent, MouseEvent

from autotrack.core import TimePoint, UserError
from autotrack.core.experiment import Experiment
from autotrack.core.position import Position
from autotrack.core.marker import Marker
from autotrack.core.spline import Spline
from autotrack.gui import dialog
from autotrack.gui.undo_redo import UndoableAction, ReversedAction
from autotrack.gui.window import Window
from autotrack.visualizer import activate, DisplaySettings
from autotrack.visualizer.abstract_editor import AbstractEditor


class _AddPathAction(UndoableAction):

    _spline: Spline
    _time_point: TimePoint
    spline_id: Optional[int]

    def __init__(self, path: Spline, time_point: TimePoint, spline_id: Optional[int] = None):
        self._spline = path
        self._time_point = time_point
        self.spline_id = spline_id

    def do(self, experiment: Experiment) -> str:
        self.spline_id = experiment.splines.add_spline(self._time_point, self._spline, self.spline_id)
        self._spline.update_offset_for_positions(experiment.positions.of_time_point(self._time_point))
        return "Added spline to time point " + str(self._time_point.time_point_number())

    def undo(self, experiment: Experiment):
        experiment.splines.remove_spline(self._time_point, self._spline)
        return "Removed spline from time point " + str(self._time_point.time_point_number())


class _AddPointAction(UndoableAction):
    _path: Spline
    _new_point: Position

    def __init__(self, path: Spline, new_point: Position):
        self._path = path
        self._new_point = new_point

    def do(self, experiment: Experiment) -> str:
        self._path.add_point(self._new_point.x, self._new_point.y, self._new_point.z)
        self._path.update_offset_for_positions(experiment.positions.of_time_point(self._new_point.time_point()))
        return f"Added point at ({self._new_point.x:.0f}, {self._new_point.y:.0f}) to selected path"

    def undo(self, experiment: Experiment) -> str:
        self._path.remove_point(self._new_point.x, self._new_point.y)
        return f"Removed point at ({self._new_point.x:.0f}, {self._new_point.y:.0f}) from path"


class _SetMarkerAction(UndoableAction):
    _axis_id: int
    _new_marker: Marker
    _old_marker: Optional[Marker]

    def __init__(self, axis_id: int, new_marker: Marker, old_marker: Optional[Marker]):
        self._axis_id = axis_id
        self._new_marker = new_marker
        self._old_marker = old_marker

    def do(self, experiment: Experiment) -> str:
        experiment.splines.set_marker(self._axis_id, self._new_marker)
        return f"Marked axis {self._axis_id} as {self._new_marker.display_name}"

    def undo(self, experiment: Experiment) -> str:
        experiment.splines.set_marker(self._axis_id, self._old_marker)
        if self._old_marker is None:
            return f"Removed marker for axis {self._axis_id}"
        return f"Marked axis {self._axis_id} again as {self._old_marker.display_name}"


class _SetCheckpointAction(UndoableAction):
    _data_axis: Spline
    _new_checkpoint: float
    _old_checkpoint: Optional[float]

    def __init__(self, data_axis: Spline, new_checkpoint: float):
        self._data_axis = data_axis
        self._new_checkpoint = new_checkpoint
        self._old_checkpoint = data_axis.get_checkpoint()

    def do(self, experiment: Experiment) -> str:
        self._data_axis.set_checkpoint(self._new_checkpoint)
        return "Inserted checkpoint"

    def undo(self, experiment: Experiment) -> str:
        self._data_axis.set_checkpoint(self._old_checkpoint)
        if self._old_checkpoint is None:
            return "Deleted checkpoint"
        return "Restored original checkpoint"


class _SetReferenceTimePointAction(UndoableAction):
    _old_reference_time_point_number: int
    _new_reference_time_point_number: int

    def __init__(self, old_reference_time_point_number: int, new_reference_time_point_number: int):
        self._old_reference_time_point_number = old_reference_time_point_number
        self._new_reference_time_point_number = new_reference_time_point_number

    def do(self, experiment: Experiment) -> str:
        experiment.splines.reference_time_point_number(self._new_reference_time_point_number)
        return f"Changed the reference time point number to {self._new_reference_time_point_number}"

    def undo(self, experiment: Experiment) -> str:
        experiment.splines.reference_time_point_number(self._old_reference_time_point_number)
        return f"Changed the reference time point number back to {self._old_reference_time_point_number}"


class DataAxisEditor(AbstractEditor):
    """Editor for data axes. Double-click to (de)select a path.
    Draw the data axis by adding points using the Insert key. Use Delete to delete a data axis.
    Add a checkpoint to a path using the X key.
    Press C to copy a selected path from another time point to this time point.
    Press P to view the position of each cell on the nearest data axis. Positions after the checkpoint are marked with a star."""

    _selected_spline_id: Optional[int]
    _draw_axis_positions: bool

    def __init__(self, window: Window, *, time_point: Optional[TimePoint] = None, z: int = 14,
                 display_settings: DisplaySettings = None):
        super().__init__(window, time_point=time_point, z=z, display_settings=display_settings)
        self._selected_spline_id = None
        self._draw_axis_positions = False

    def get_extra_menu_options(self) -> Dict[str, Any]:
        options = {
            **super().get_extra_menu_options(),
            "View//Toggle-Toggle showing data axis positions [P]": self._toggle_viewing_axis_positions,
            "Edit//Axes-Change reference time point...": self._set_reference_time_point,
            "Edit//Axes-Copy axis to this time point [C]": self._copy_axis_to_current_time_point,
        }

        # Add options for changing axis types
        for position_type in self.get_window().get_gui_experiment().get_registered_markers(Spline):
            # Create copy of position_type variable to avoid it changing in loop iteration
            action = lambda bound_position_type=position_type: self._mark_axis(bound_position_type)

            options["Edit//Type-Set type of axis//" + position_type.display_name] = action
        return options

    def _toggle_viewing_axis_positions(self):
        self._draw_axis_positions = not self._draw_axis_positions
        self.draw_view()

    def _on_mouse_click(self, event: MouseEvent):
        if event.dblclick:
            # Select path
            links = self._experiment.links
            position = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
            spline_position = self._experiment.splines.to_position_on_spline(position)
            if spline_position is None or spline_position.distance > 10 or spline_position.axis_id == self._selected_spline_id:
                self._selected_spline_id = None
            else:
                self._selected_spline_id = spline_position.axis_id
            self.draw_view()

    def _get_figure_title(self) -> str:
        return "Editing axes in time point " + str(self._time_point.time_point_number()) + "    (z=" + str(self._z) + ")"

    def _get_window_title(self) -> str:
        return "Manual data editing"

    def _get_selected_path_of_current_time_point(self) -> Optional[Spline]:
        if self._selected_spline_id is None:
            return None
        selected_spline = self._experiment.splines.get_spline(self._time_point, self._selected_spline_id)
        if selected_spline is None:
            return None
        return selected_spline

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int):
        if not self._draw_axis_positions or dt != 0 or abs(dz) > 3:
            super()._on_position_draw(position, color, dz, dt)
            return

        axis_position = self._experiment.splines.to_position_on_original_axis(self._experiment.links, position)
        if axis_position is None:
            super()._on_position_draw(position, color, dz, dt)
            return

        background_color = (1, 1, 1, 0.8) if axis_position.axis_id == self._selected_spline_id else (0, 1, 0, 0.8)
        star = "*" if axis_position.is_after_checkpoint() else ""
        self._draw_annotation(position, f"{star}{axis_position.pos:.1f}", background_color=background_color)

    def _draw_data_axis(self, data_axis: Spline, id: int, color: str, marker_size_max: int):
        if data_axis == self._get_selected_path_of_current_time_point():
            color = "white"  # Highlight the selected path
            marker_size_max = int(marker_size_max * 1.5)

        super()._draw_data_axis(data_axis, id, color, marker_size_max)

        pos_x, pos_y = data_axis.get_points_2d()
        self._ax.annotate(self._get_axis_label(id), (pos_x[0], pos_y[0] + 10), fontsize=12, fontweight="bold", color=color)

    def _get_axis_label(self, axis_id: int) -> str:
        marker_name = self._experiment.splines.get_marker_name(axis_id)
        marker = self._window.get_gui_experiment().get_marker_by_save_name(marker_name)
        if marker is None:
            return f"Axis {axis_id}"
        return f"Axis {axis_id}: {marker.display_name}"

    def _on_key_press(self, event: KeyEvent):
        if event.key == "insert":
            selected_path = self._get_selected_path_of_current_time_point()
            if selected_path is None:
                # Time for a new path
                path = Spline()
                path.add_point(event.xdata, event.ydata, self._z)
                add_action = _AddPathAction(path, self._time_point)
                self._perform_action(add_action)
                self._selected_spline_id = add_action.spline_id
            else:
                # Can modify existing path
                point = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
                self._perform_action(_AddPointAction(selected_path, point))
        elif event.key == "delete":
            selected_path = self._get_selected_path_of_current_time_point()
            if selected_path is None:
                self.update_status("No path selected - cannot delete anything.")
                return
            else:
                self._selected_spline_id = None
                self._perform_action(ReversedAction(_AddPathAction(selected_path, self._time_point)))
        elif event.key == "x":
            if event.xdata is None or event.ydata is None:
                return
            self._set_checkpoint(event.xdata, event.ydata)
        else:
            super()._on_key_press(event)

    def _copy_axis_to_current_time_point(self):
        """Copies the axis of the previous time point to the current time point."""
        if self._selected_spline_id is None:
            self.update_status("No spline selected, cannot copy anything")
            return
        if self._experiment.splines.get_spline(self._time_point, self._selected_spline_id) is not None:
            self.update_status(f"Cannot copy spline with id {self._selected_spline_id} to current time point: there is"
                               f" already a spline with that id here.")
            return

        previous_spline = None
        next_spline = None
        try:
            previous_spline = self._experiment.splines.get_spline(self._experiment.get_previous_time_point(self._time_point), self._selected_spline_id)
        except ValueError:
            pass
        try:
            next_spline = self._experiment.splines.get_spline(self._experiment.get_next_time_point(self._time_point), self._selected_spline_id)
        except ValueError:
            pass

        if previous_spline is not None:
            copied_spline = previous_spline.copy()
            self._perform_action(_AddPathAction(copied_spline, self._time_point, self._selected_spline_id))
            return
        if next_spline is not None:
            copied_spline = next_spline.copy()
            self._perform_action(_AddPathAction(copied_spline, self._time_point, self._selected_spline_id))
            return
        self.update_status(f"Neither the previous nor the next time point has a spline with id"
                           f" {self._selected_spline_id}; cannot copy anything.")

    def _exit_view(self):
        from autotrack.visualizer.link_and_position_editor import LinkAndPositionEditor
        data_editor = LinkAndPositionEditor(self._window, time_point=self._time_point, z=self._z,
                                            display_settings=self._display_settings)
        activate(data_editor)

    def _set_checkpoint(self, x: float, y: float):
        selected_spline = self._get_selected_path_of_current_time_point()
        if selected_spline is None:
            self.update_status(f"No spline selected in the current time point. Double-click a spline to select it.")
            return
        on_axis = selected_spline.to_position_on_axis(Position(x, y, self._z))
        if on_axis is None:
            self.update_status("Cannot set a checkpoint here on this spline - add more points to the spline first")
            return
        if on_axis.distance > 20:
            self.update_status("Cannot set a checkpoint here - mouse is not near selected spline")
            return
        self._perform_action(_SetCheckpointAction(selected_spline, on_axis.pos))

    def _mark_axis(self, axis_marker: Marker):
        selected_spline = self._get_selected_path_of_current_time_point()

        if selected_spline is None:
            self.update_status("No spline selected - cannot set type")
            return

        old_marker_name = self._experiment.splines.get_marker_name(self._selected_spline_id)
        old_marker = self._window.get_gui_experiment().get_marker_by_save_name(old_marker_name)
        self._perform_action(_SetMarkerAction(self._selected_spline_id, axis_marker, old_marker))

    def _set_reference_time_point(self):
        """Asks the user for a new reference time point."""
        min_time_point_number = self._experiment.first_time_point_number()
        max_time_point_number = self._experiment.last_time_point_number()
        if min_time_point_number is None or max_time_point_number is None:
            raise UserError("Reference time point", "No data is loaded - cannot change reference time point")
        reference_time_point_number = self._experiment.splines.reference_time_point_number()
        explanation = "Data axes are used to follow positions over time accross a trajectory. If you have multiple\n" \
                      " of such trajectories, then all positions need to be assigned to one of these data axes.\n\n" \
                      f"Currently, each cell belongs to the data axis that was the closest by in time point" \
                      f" {reference_time_point_number}. Which\ntime point should it be instead?" \
                      f" ({min_time_point_number}-{max_time_point_number}, inclusive)"
        answer = dialog.prompt_int("Reference time point", explanation, minimum=min_time_point_number,
                                   maximum=max_time_point_number, default=self._time_point.time_point_number())
        if answer is None:
            return
        self._perform_action(_SetReferenceTimePointAction(reference_time_point_number, answer))
