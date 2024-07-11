from typing import Optional, Dict, Any

from matplotlib.backend_bases import KeyEvent, MouseEvent

from organoid_tracker.core import TimePoint, UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.marker import Marker
from organoid_tracker.core.spline import Spline
from organoid_tracker.gui import dialog
from organoid_tracker.gui.undo_redo import UndoableAction, ReversedAction
from organoid_tracker.gui.window import Window, DisplaySettings
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.abstract_editor import AbstractEditor


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


class _SetReferenceTimePointAction(UndoableAction):
    _old_reference_time_point: TimePoint
    _new_reference_time_point: TimePoint

    def __init__(self, old_reference_time_point: TimePoint, new_reference_time_point: TimePoint):
        self._old_reference_time_point = old_reference_time_point
        self._new_reference_time_point = new_reference_time_point

    def do(self, experiment: Experiment) -> str:
        experiment.splines.reference_time_point(self._new_reference_time_point)
        return f"Changed the reference time point number to {self._new_reference_time_point.time_point_number()}"

    def undo(self, experiment: Experiment) -> str:
        experiment.splines.reference_time_point(self._old_reference_time_point)
        return f"Changed the reference time point number back to {self._old_reference_time_point.time_point_number()}"


class SplineEditor(AbstractEditor):
    """Editor for splines. Double-click to (de)select a spline.
    Draw the spline by adding points using the Insert key. Use Delete to delete a spline.
    Press C to copy a selected spline from another time point to this time point.
    Press P to view the position of each cell on the nearest spline."""

    _selected_spline_id: Optional[int]
    _draw_spline_positions: bool

    def __init__(self, window: Window):
        window.display_settings.show_splines = True
        super().__init__(window)
        self._selected_spline_id = None
        self._draw_spline_positions = False

    def get_extra_menu_options(self) -> Dict[str, Any]:
        options = {
            **super().get_extra_menu_options(),
            "View//Toggle-Toggle showing spline positions [S]": self._toggle_viewing_axis_positions,
            "Edit//Axes-Change reference time point...": self._set_reference_time_point,
            "Edit//Axes-Copy axis to this time point [C]": self._copy_spline_to_current_time_point,
        }

        # Add options for changing axis types
        for position_type in self.get_window().registry.get_registered_markers(Spline):
            # Create copy of position_type variable to avoid it changing in loop iteration
            action = lambda bound_position_type=position_type: self._mark_spline_as_type(bound_position_type)

            options["Edit//Type-Set type of axis//" + position_type.display_name] = action
        return options

    def _toggle_viewing_axis_positions(self):
        self._draw_spline_positions = not self._draw_spline_positions
        self.draw_view()

    def _on_mouse_single_click(self, event: MouseEvent):
        # Select path
        links = self._experiment.links
        position = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
        spline_position = self._experiment.splines.to_position_on_spline(position)
        if spline_position is None or spline_position.distance > 10 or spline_position.spline_id == self._selected_spline_id:
            self._selected_spline_id = None
        else:
            self._selected_spline_id = spline_position.spline_id
        self.draw_view()

    def _get_figure_title(self) -> str:
        return ("Editing splines in time point " + str(self._time_point.time_point_number())
                + "    (z=" + self._get_figure_title_z_str() + ")")

    def _get_window_title(self) -> str:
        return "Manual data editing"

    def _get_selected_spline_of_current_time_point(self) -> Optional[Spline]:
        if self._selected_spline_id is None:
            return None
        selected_spline = self._experiment.splines.get_spline(self._time_point, self._selected_spline_id)
        if selected_spline is None:
            return None
        return selected_spline

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int) -> bool:
        if not self._draw_spline_positions or dt != 0 or abs(dz) > self.MAX_Z_DISTANCE:
            return super()._on_position_draw(position, color, dz, dt)

        spline_position = self._experiment.splines.to_position_on_original_axis(self._experiment.links, position)
        if spline_position is None:
            return super()._on_position_draw(position, color, dz, dt)

        background_color = (1, 1, 1, 0.8) if spline_position.spline_id == self._selected_spline_id else (0, 1, 0, 0.8)
        self._draw_annotation(position, f"{spline_position.pos:.1f}", background_color=background_color)

    def _draw_spline(self, data_axis: Spline, id: int, color: str, marker_size_max: int):
        if data_axis == self._get_selected_spline_of_current_time_point():
            color = "white"  # Highlight the selected path
            marker_size_max = int(marker_size_max * 1.5)

        super()._draw_spline(data_axis, id, color, marker_size_max)

        pos_x, pos_y = data_axis.get_points_2d()
        self._ax.annotate(self._get_axis_label(id), (pos_x[0], pos_y[0] + 10), fontsize=12, fontweight="bold", color=color)

    def _get_axis_label(self, axis_id: int) -> str:
        marker_name = self._experiment.splines.get_marker_name(axis_id)
        marker = self._window.registry.get_marker_by_save_name(marker_name)
        if marker is None:
            return f"Axis {axis_id}"
        return f"Axis {axis_id}: {marker.display_name}"

    def _on_key_press(self, event: KeyEvent):
        if event.key == "insert" or event.key == "enter":
            selected_path = self._get_selected_spline_of_current_time_point()
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
        elif event.key == "delete" or event.key == "backspace":
            selected_path = self._get_selected_spline_of_current_time_point()
            if selected_path is None:
                self.update_status("No path selected - cannot delete anything.")
                return
            else:
                self._selected_spline_id = None
                self._perform_action(ReversedAction(_AddPathAction(selected_path, self._time_point)))
        else:
            super()._on_key_press(event)

    def _copy_spline_to_current_time_point(self):
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
        from organoid_tracker.visualizer.link_and_position_editor import LinkAndPositionEditor
        data_editor = LinkAndPositionEditor(self._window)
        activate(data_editor)

    def _mark_spline_as_type(self, axis_marker: Marker):
        selected_spline = self._get_selected_spline_of_current_time_point()

        if selected_spline is None:
            self.update_status("No spline selected - cannot set type")
            return

        old_marker_name = self._experiment.splines.get_marker_name(self._selected_spline_id)
        old_marker = self._window.registry.get_marker_by_save_name(old_marker_name)
        self._perform_action(_SetMarkerAction(self._selected_spline_id, axis_marker, old_marker))

    def _set_reference_time_point(self):
        """Asks the user for a new reference time point."""
        min_time_point_number = self._experiment.first_time_point_number()
        max_time_point_number = self._experiment.last_time_point_number()
        if min_time_point_number is None or max_time_point_number is None:
            raise UserError("Reference time point", "No data is loaded - cannot change reference time point")
        reference_time_point = self._experiment.splines.reference_time_point()
        if reference_time_point is None:
            reference_time_point = TimePoint(min_time_point_number)
        explanation = "Splines are used to follow positions over time accross a trajectory. If you have multiple\n" \
                      " of such trajectories, then all positions need to be assigned to one of these splines.\n\n" \
                      f"Currently, each cell belongs to the spline that was the closest by in time point" \
                      f" {reference_time_point.time_point_number()}. Which\ntime point should it be instead?" \
                      f" ({min_time_point_number}-{max_time_point_number}, inclusive)"
        answer = dialog.prompt_int("Reference time point", explanation, minimum=min_time_point_number,
                                   maximum=max_time_point_number, default=self._time_point.time_point_number())
        if answer is None:
            return
        self._perform_action(_SetReferenceTimePointAction(reference_time_point, TimePoint(answer)))
