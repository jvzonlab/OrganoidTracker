from functools import partial
from typing import Optional, Dict, Any, NamedTuple

from matplotlib.backend_bases import KeyEvent, MouseEvent

from organoid_tracker import core
from organoid_tracker.core import TimePoint, UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.marker import Marker
from organoid_tracker.core.position import Position
from organoid_tracker.core.spline import Spline, SplineCheckpoint, SplinePosition
from organoid_tracker.gui import dialog, option_choose_dialog
from organoid_tracker.gui.undo_redo import UndoableAction, ReversedAction
from organoid_tracker.gui.window import Window
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.abstract_editor import AbstractEditor

_NO_PREVIOUS_CHECKPOINT_MARKER = "UNDECIDED"

class _SplineRef(NamedTuple):
    spline_id: int
    time_point: TimePoint

    @staticmethod
    def from_spline_position(spline_position: SplinePosition) -> "_SplineRef":
        return _SplineRef(spline_id=spline_position.spline_id, time_point=spline_position.time_point)

    def with_time_point(self, time_point: TimePoint) -> "_SplineRef":
        """Creates a new spline ref with the same id, but a different time point."""
        return _SplineRef(spline_id=self.spline_id, time_point=time_point)


def _spline_matches(spline_ref: Optional[_SplineRef], spline_position: SplinePosition) -> bool:
    if spline_ref is None:
        return False
    return spline_ref.spline_id == spline_position.spline_id and spline_ref.time_point == spline_position.time_point


class _AddSplineAction(UndoableAction):

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
        return "Removed spline from time point " + str(self._time_point.time_point_number()) + "."


class _AddPointAction(UndoableAction):
    _spline: Spline
    _new_point: Position

    def __init__(self, spline: Spline, new_point: Position):
        self._spline = spline
        self._new_point = new_point

    def do(self, experiment: Experiment) -> str:
        self._spline.add_point(self._new_point.x, self._new_point.y, self._new_point.z)
        self._spline.update_offset_for_positions(experiment.positions.of_time_point(self._new_point.time_point()))
        return f"Added point at ({self._new_point.x:.0f}, {self._new_point.y:.0f}) to selected spline."

    def undo(self, experiment: Experiment) -> str:
        self._spline.remove_point(self._new_point.x, self._new_point.y)
        return f"Removed point at ({self._new_point.x:.0f}, {self._new_point.y:.0f}) from spline."


class _AddCheckpointAction(UndoableAction):
    _spline: Spline
    _checkpoint: SplineCheckpoint

    def __init__(self, spline: Spline, checkpoint: SplineCheckpoint):
        self._spline = spline
        self._checkpoint = checkpoint

    def do(self, experiment: Experiment) -> str:
        self._spline.add_checkpoint(self._checkpoint)
        type_str = f"of type '{self._checkpoint.save_name}'" if self._checkpoint.save_name is not None else "without a specific type"
        return f"Added checkpoint {type_str} at height {self._checkpoint.pos:.1f} to selected spline"

    def undo(self, experiment: Experiment) -> str:
        self._spline.remove_checkpoint(self._checkpoint)
        return f"Removed checkpoint at height {self._checkpoint.pos:.1f} from spline."


class _SetSplineMarkerAction(UndoableAction):
    _axis_id: int
    _new_marker: Marker
    _old_marker: Optional[Marker]

    def __init__(self, axis_id: int, new_marker: Marker, old_marker: Optional[Marker]):
        self._axis_id = axis_id
        self._new_marker = new_marker
        self._old_marker = old_marker

    def do(self, experiment: Experiment) -> str:
        experiment.splines.set_marker(self._axis_id, self._new_marker)
        return f"Marked axis {self._axis_id} as {self._new_marker.display_name}."

    def undo(self, experiment: Experiment) -> str:
        experiment.splines.set_marker(self._axis_id, self._old_marker)
        if self._old_marker is None:
            return f"Removed marker for axis {self._axis_id}."
        return f"Marked axis {self._axis_id} again as {self._old_marker.display_name}."


class _SetCheckpointMarkerAction(UndoableAction):
    _spline: Spline
    _old_checkpoint: SplineCheckpoint
    _new_checkpoint: SplineCheckpoint
    _new_marker: Optional[Marker]

    def __init__(self, *, spline: Spline, checkpoint: SplineCheckpoint, new_marker: Optional[Marker]):
        self._spline = spline
        self._old_checkpoint = checkpoint
        self._new_checkpoint = checkpoint.with_save_name(None if new_marker is None else new_marker.save_name)
        self._new_marker = new_marker

    def do(self, experiment: Experiment) -> str:
        self._spline.remove_checkpoint(self._old_checkpoint)
        self._spline.add_checkpoint(self._new_checkpoint)
        if self._new_marker is None:
            return f"Updated checkpoint at {self._new_checkpoint.pos:.1f} to have no specific type."
        return f"Updated checkpoint at {self._new_checkpoint.pos:.1f} to be of type \"{self._new_marker.display_name}\"."

    def undo(self, experiment: Experiment) -> str:
        self._spline.remove_checkpoint(self._new_checkpoint)
        self._spline.add_checkpoint(self._new_checkpoint)

        if self._old_checkpoint.save_name is None:
            return f"Removed type from checkpoint at {self._old_checkpoint.pos:.1f}."
        else:
            return f"Updated checkpoint at {self._old_checkpoint.pos:.1f} to be of type \"{self._old_checkpoint.save_name}\"."


class _MoveCheckpointAction(UndoableAction):
    _spline: Spline
    _old_checkpoint: SplineCheckpoint
    _new_checkpoint: SplineCheckpoint

    def __init__(self, spline: Spline, checkpoint: SplineCheckpoint, new_position: float):
        self._spline = spline
        self._old_checkpoint = checkpoint
        self._new_checkpoint = checkpoint.with_position(new_position)

    def do(self, experiment: Experiment) -> str:
        self._spline.remove_checkpoint(self._old_checkpoint)
        self._spline.add_checkpoint(self._new_checkpoint)
        return f"Moved the checkpoint to position {self._new_checkpoint.pos:.1f}."

    def undo(self, experiment: Experiment) -> str:
        self._spline.remove_checkpoint(self._new_checkpoint)
        self._spline.add_checkpoint(self._old_checkpoint)
        return f"Moved the checkpoint back to position {self._new_checkpoint.pos:.1f}."



class _SetReferenceTimePointAction(UndoableAction):
    _old_reference_time_point: TimePoint
    _new_reference_time_point: TimePoint

    def __init__(self, old_reference_time_point: TimePoint, new_reference_time_point: TimePoint):
        self._old_reference_time_point = old_reference_time_point
        self._new_reference_time_point = new_reference_time_point

    def do(self, experiment: Experiment) -> str:
        experiment.splines.reference_time_point(self._new_reference_time_point)
        return f"Changed the reference time point number to {self._new_reference_time_point.time_point_number()}."

    def undo(self, experiment: Experiment) -> str:
        experiment.splines.reference_time_point(self._old_reference_time_point)
        return f"Changed the reference time point number back to {self._old_reference_time_point.time_point_number()}."


class _ReverseSplineAction(UndoableAction):

    _spline_id: int

    def __init__(self, spline_id: int):
        self._spline_id = spline_id

    def do(self, experiment: Experiment) -> str:
        for time_point in experiment.splines.time_points():
            spline = experiment.splines.get_spline(time_point, self._spline_id)
            if spline is None:
                continue
            spline.reverse()
            spline.update_offset_for_positions(experiment.positions.of_time_point(time_point))
        return f"Reversed the direction of spline {self._spline_id} across all time points."

    def undo(self, experiment: Experiment) -> str:
        return self.do(experiment)  # Do and undo are the same in this case



class SplineEditor(AbstractEditor):
    """Editor for splines. Draw the spline by adding points using the Insert (or Enter) key. Use Delete to delete a spline.
    If you click somewhere on a spline, the first click selects the spline, and the second a point on that spline.
    You can then use the Insert key to mark that point on the spline.
    If you have selected a spline at a different time point, the Insert key copies the spline over to this time point.
    Press P to view the position of each cell on the nearest spline."""

    _selected_spline_ref: Optional[_SplineRef] = None
    _selected_checkpoint_location: Optional[Position] = None
    _selected_checkpoint: Optional[SplineCheckpoint] = None  # Can only be set if _selected_checkpoint_location is not None

    # save_name of the previously used checkpoint type. None if the last used checkpoint had no specific type.
    # Set to _NO_PREVIOUS_CHECKPOINT_MARKER if no checkpoint type has been used yet.
    _last_used_checkpoint_type: Optional[str] = _NO_PREVIOUS_CHECKPOINT_MARKER

    _draw_spline_positions: bool = False

    def __init__(self, window: Window):
        window.display_settings.show_splines = True
        super().__init__(window)

    def _draw_extra(self):
        # Draws the clicked position
        if self._selected_checkpoint_location is not None and self._selected_checkpoint_location.time_point_number() == self._time_point.time_point_number():
            alpha = 0.6 if self._selected_checkpoint is None else 1.0  # Draw transparently if checkpoint doesn't exist yet
            self._ax.scatter(self._selected_checkpoint_location.x, self._selected_checkpoint_location.y, color="white",
                             edgecolor="black", marker="D", s=13 ** 2, zorder=2, alpha=alpha)

        # Draws the selected spline if in another time point
        if self._selected_spline_ref is not None and self._selected_spline_ref.time_point != self._time_point:
            selected_spline = self._experiment.splines.get_spline(self._selected_spline_ref.time_point, self._selected_spline_ref.spline_id)
            spline_here = self._experiment.splines.get_spline(self._time_point, self._selected_spline_ref.spline_id)
            if selected_spline is not None and spline_here is None:
                color = core.COLOR_CELL_PREVIOUS if self._selected_spline_ref.time_point < self._time_point else core.COLOR_CELL_NEXT
                self._ax.plot(*selected_spline.get_interpolation_2d(), color=color, linewidth=1.5, linestyle="dotted")


    def get_extra_menu_options(self) -> Dict[str, Any]:
        options = {
            **super().get_extra_menu_options(),
            "View//Toggle-Toggle showing spline positions [S]": self._toggle_viewing_axis_positions,
            "Edit//Axes-Change reference time point...": self._set_reference_time_point,
            "Edit//Axes-Reverse direction of spline": self._reverse_spline,
        }

        # Add options for changing spline types
        for spline_type in self.get_window().registry.get_registered_markers(Spline):
            options["Edit//Type-Set type of spline//" + spline_type.display_name] = partial(self._mark_spline_as_type, spline_type)

        # Add options for changing spline checkpoint types
        checkpoint_types = list(self.get_window().registry.get_registered_markers(SplineCheckpoint))
        for checkpoint_type in checkpoint_types:
            options["Edit//Type-Set type of checkpoint//Type-" + checkpoint_type.display_name] = partial(self._mark_spline_checkpoint_as_type,checkpoint_type)
        if checkpoint_types:
            options["Edit//Type-Set type of checkpoint//Untyped-Remove checkpoint type"] = partial(self._mark_spline_checkpoint_as_type,None)
        return options

    def _reverse_spline(self):
        spline = self._get_selected_spline_of_current_time_point()
        if spline is None or self._selected_spline_ref is None:
            self.update_status("No spline selected. Cannot reverse anything.")
            return

        self._perform_action(_ReverseSplineAction(self._selected_spline_ref.spline_id))

    def _toggle_viewing_axis_positions(self):
        self._draw_spline_positions = not self._draw_spline_positions
        self.draw_view()

    def _on_mouse_single_click(self, event: MouseEvent):
        if event.xdata is None or event.ydata is None:
            return

        position = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
        spline_position = self._experiment.splines.to_position_on_spline(position)
        if spline_position is None or spline_position.distance > 10:
            # Clicked outside any spline
            self._deselect_spline()
            self.update_status("Clicked outside any spline. Removed any existing selection.")
        elif _spline_matches(self._selected_spline_ref, spline_position):
            # Clicked on an already selected spline
            clicked_location = spline_position.spline.from_position_on_axis(spline_position.pos)
            if clicked_location is None:
                self._deselect_spline()
            else:
                closest_checkpoint = spline_position.spline.get_closest_checkpoint(spline_position.pos, max_distance=8)
                if closest_checkpoint is not None:
                    clicked_location = spline_position.spline.from_position_on_axis(closest_checkpoint.pos)
                    if clicked_location is None:
                        raise ValueError("Location of existing checkpoint was None")
                    self.update_status(f"Clicked on existing checkpoint at height {spline_position.pos:.1f}."
                                       f" Press Delete or Backspace to delete the checkpoint here."
                                       f" Press Shift to move the checkpoint to a new location along the spline.")
                else:
                    self.update_status(f"Clicked at height {spline_position.pos:.1f} on spline"
                                       f" {spline_position.spline_id}. Press Insert or Enter to insert a new checkpoint here.")
                self._selected_checkpoint_location = Position(*clicked_location, time_point=self._time_point)
                self._selected_checkpoint = closest_checkpoint
        else:
            # Select spline that was clicked on
            self._deselect_spline()
            self._selected_spline_ref = _SplineRef.from_spline_position(spline_position)

            # Check for clicks on an existing checkpoint
            closest_checkpoint = spline_position.spline.get_closest_checkpoint(spline_position.pos, max_distance=8)
            if closest_checkpoint is not None:
                clicked_location = spline_position.spline.from_position_on_axis(closest_checkpoint.pos)
                if clicked_location is None:
                    raise ValueError("Location of existing checkpoint was None")
                self.update_status(f"Selected spline {spline_position.spline_id} with checkpoint at height {spline_position.pos:.1f}."
                                   f" Press Delete or Backspace to delete the checkpoint here."
                                   f" Press Shift to move the checkpoint to a new location along the spline.")
                self._selected_checkpoint = closest_checkpoint
                self._selected_checkpoint_location = Position(*clicked_location, time_point=self._time_point)
            else:
                self.update_status(f"Selected spline {spline_position.spline_id}. Press Delete or Backspace to delete the spline here.")
        self.draw_view()

    def _get_figure_title(self) -> str:
        return ("Editing splines in time point " + str(self._time_point.time_point_number())
                + "    (z=" + self._get_figure_title_z_str() + ")")

    def _get_window_title(self) -> str:
        return "Manual data editing"

    def _get_selected_spline_of_current_time_point(self) -> Optional[Spline]:
        if self._selected_spline_ref is None or self._selected_spline_ref.time_point != self._time_point:
            return None
        selected_spline = self._experiment.splines.get_spline(self._time_point, self._selected_spline_ref.spline_id)
        if selected_spline is None:
            return None
        return selected_spline

    def _get_selected_checkpoint_of_current_time_point(self) -> Optional[SplineCheckpoint]:
        if (self._selected_checkpoint_location is None or
                self._selected_checkpoint_location.time_point_number() != self._time_point.time_point_number()):
            return None  # No checkpoint selected
        selected_spline = self._get_selected_spline_of_current_time_point()
        if selected_spline is None:
            return None  # No spline selected

        return self._selected_checkpoint

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int) -> bool:
        if not self._draw_spline_positions or dt != 0 or abs(dz) > self.MAX_Z_DISTANCE:
            return super()._on_position_draw(position, color, dz, dt)

        spline_position = self._experiment.splines.to_position_on_original_axis(self._experiment.links, position)
        if spline_position is None:
            return super()._on_position_draw(position, color, dz, dt)

        background_color = (1, 1, 1, 0.8) if _spline_matches(self._selected_spline_ref, spline_position) else (0, 1, 0, 0.8)
        self._draw_annotation(position, f"{spline_position.pos:.1f}", background_color=background_color)
        return False

    def _draw_spline(self, spline: Spline, id: int, color: str, marker_size_max: int):
        if spline == self._get_selected_spline_of_current_time_point():
            color = "white"  # Highlight the selected path
            marker_size_max = int(marker_size_max * 1.5)

        super()._draw_spline(spline, id, color, marker_size_max)

        pos_x, pos_y = spline.get_points_2d()
        self._ax.annotate(self._get_axis_label(id), (pos_x[0], pos_y[0] + 10), fontsize=12, fontweight="bold", color=color)

        # Draw checkpoints
        viewer_z = self._z
        registry = self._window.registry
        for checkpoint in spline.checkpoints():
            checkpoint_position = spline.from_position_on_axis(checkpoint.pos)
            if checkpoint_position is None:
                continue
            x, y, z = checkpoint_position
            dz = abs(z - viewer_z)
            if dz > 3:
                dz = 3
            marker = registry.get_marker_by_save_name(checkpoint.save_name)
            name = checkpoint.save_name
            background_color = "#dddddd"
            if name is not None:
                # Display name annotation
                if marker is not None and marker.applies_to(SplineCheckpoint):
                    name = marker.display_name
                    background_color = marker.mpl_color
                self._draw_annotation(Position(x, y, z), name, horizontalalignment="left",
                                      verticalalignment="bottom")

            self._ax.scatter(x, y, color=background_color, edgecolor="black", s=(10 - dz) ** 2, zorder=1,
                             marker="D")

    def _get_axis_label(self, axis_id: int) -> str:
        marker_name = self._experiment.splines.get_marker_name(axis_id)
        marker = self._window.registry.get_marker_by_save_name(marker_name)
        if marker is None:
            return f"Axis {axis_id}"
        return f"Axis {axis_id}: {marker.display_name}"

    def _on_key_press(self, event: KeyEvent):
        if event.key == "insert" or event.key == "enter":
            if event.xdata is None or event.ydata is None:
                return

            # Insert a checkpoint on the previously selected point
            if self._selected_checkpoint_location is not None and self._selected_checkpoint_location.time_point_number() == self._time_point.time_point_number():
                self._insert_checkpoint()
                return

            # Copy an existing spline to this time point
            if self._selected_spline_ref is not None and self._selected_spline_ref.time_point != self._time_point:
                self._copy_spline_to_current_time_point()
                return

            self._insert_spline_point(event.xdata, event.ydata)
        elif event.key == "delete" or event.key == "backspace":
            selected_spline = self._get_selected_spline_of_current_time_point()
            if selected_spline is None:
                self.update_status("No spline selected - cannot delete anything.")
                return
            selected_checkpoint = self._get_selected_checkpoint_of_current_time_point()
            if selected_checkpoint is not None:
                # Got a selected checkpoint - delete that one instead
                self._selected_checkpoint_location = None
                self._selected_checkpoint = None
                self._perform_action(ReversedAction(_AddCheckpointAction(selected_spline, selected_checkpoint)))
                return
            # Delete entire spline
            self._deselect_spline()
            self._perform_action(ReversedAction(_AddSplineAction(selected_spline, self._time_point)))
        elif event.key == "shift":
            # Move a checkpoint
            if event.xdata is None or event.ydata is None:
                return
            selected_spline = self._get_selected_spline_of_current_time_point()
            checkpoint = self._get_selected_checkpoint_of_current_time_point()
            if selected_spline is None or checkpoint is None:
                self.update_status("No checkpoint selected - cannot move anything.")
                return
            spline_position = selected_spline.to_position_on_axis(Position(event.xdata, event.ydata, self._z))
            if spline_position is None:
                self.update_status("Cannot move the checkpoint off the spline.")
                return
            self._selected_checkpoint = checkpoint.with_position(spline_position.pos)
            self._selected_checkpoint_location = Position(*selected_spline.from_position_on_axis(spline_position.pos), time_point=self._time_point)
            self._perform_action(_MoveCheckpointAction(selected_spline, checkpoint, spline_position.pos))
        else:
            super()._on_key_press(event)

    def _insert_spline_point(self, x: float, y: float):
        selected_spline = self._get_selected_spline_of_current_time_point()
        if selected_spline is None:
            # Time for a new path
            spline = Spline()
            spline.add_point(x, y, self._z)
            add_action = _AddSplineAction(spline, self._time_point)
            self._perform_action(add_action)
            self._deselect_spline()
            self._selected_spline_ref = _SplineRef(spline_id=add_action.spline_id, time_point=self._time_point)
        else:
            # Can modify existing spline
            point = Position(x, y, self._z, time_point=self._time_point)
            self._perform_action(_AddPointAction(selected_spline, point))
            self._selected_checkpoint = None
            self._selected_checkpoint_location = None

    def _insert_checkpoint(self):
        # Check if selection is suitable
        spline = self._get_selected_spline_of_current_time_point()
        if spline is None:
            self.update_status("No spline selected, cannot insert checkpoint.")
            return
        if self._selected_checkpoint_location is None or self._selected_checkpoint_location.time_point_number() != self._time_point.time_point_number():
            self.update_status("No location on spline selected, cannot insert checkpoint.")
            return
        if self._selected_checkpoint is not None:
            self.update_status("Selected location already has a checkpoint, cannot insert another one here.")
            return
        spline_position = spline.to_position_on_axis(self._selected_checkpoint_location)
        if spline_position is None:
            self.update_status("Selected location was outside spline, cannot insert checkpoint.")
            return

        # Prompt for marker type
        if self._last_used_checkpoint_type == _NO_PREVIOUS_CHECKPOINT_MARKER:
            available_types = []
            available_type_names = []
            for marker in self._window.registry.get_registered_markers(SplineCheckpoint):
                available_types.append(marker.save_name)
                available_type_names.append(marker.display_name)
            available_types.append(None)
            available_type_names.append("<no specific type>")

            chosen_type_index = option_choose_dialog.prompt_list("Checkpoint type", "Which type of checkpoint is this?", "Checkpoint type:", available_type_names)
            if chosen_type_index is None:
                return  # Cancelled
            chosen_type_save_name = available_types[chosen_type_index]
            self._last_used_checkpoint_type = chosen_type_save_name
        else:
            chosen_type_save_name = self._last_used_checkpoint_type

        # Add the checkpoint
        checkpoint = SplineCheckpoint(save_name=chosen_type_save_name, pos=spline_position.pos)
        self._selected_checkpoint = checkpoint
        self._perform_action(_AddCheckpointAction(spline, checkpoint))

    def _deselect_spline(self):
        """Removes any active selection."""
        self._selected_spline_ref = None
        self._selected_checkpoint_location = None
        self._selected_checkpoint = None

    def _copy_spline_to_current_time_point(self):
        """Copies the axis of the previous time point to the current time point."""
        if self._selected_spline_ref is None:
            self.update_status("No spline selected, cannot copy anything")
            return

        spline = self._experiment.splines.get_spline(self._selected_spline_ref.time_point, self._selected_spline_ref.spline_id)
        if spline is None:
            self.update_status("No spline selected, cannot copy anything")
            return

        spline_at_current_time_point = self._experiment.splines.get_spline(self._time_point, self._selected_spline_ref.spline_id)
        if spline_at_current_time_point is not None:
            # That spline is already at this time point. Maybe we can still copy over the checkpoint?
            if self._selected_checkpoint is None:
                self.update_status(f"Spline {self._selected_spline_ref.spline_id} already exists at this time point, cannot copy.")
                return
            if spline_at_current_time_point.get_closest_checkpoint(self._selected_checkpoint.pos, max_distance=1) is not None:
                self.update_status(f"Spline {self._selected_spline_ref.spline_id} already has a checkpoint at this height, cannot copy checkpoint.")
                return
            new_checkpoint_pos = spline_at_current_time_point.from_position_on_axis(self._selected_checkpoint.pos)
            if new_checkpoint_pos is None:
                self.update_status("Cannot insert selected checkpoint here, outside the range of the spline at this time point.")
                return
            # Checkpoint can be copied over, do so, and update selection
            self._selected_spline_ref = self._selected_spline_ref.with_time_point(self._time_point)
            self._selected_checkpoint_location = Position(*new_checkpoint_pos, time_point=self._time_point)
            self._perform_action(_AddCheckpointAction(spline_at_current_time_point, self._selected_checkpoint))
            return

        # Select the new spline and add it
        copied_spline = spline.copy()
        new_spline_ref = self._selected_spline_ref.with_time_point(self._time_point)
        self._deselect_spline()
        self._selected_spline_ref = new_spline_ref
        self._perform_action(_AddSplineAction(copied_spline, self._time_point, self._selected_spline_ref.spline_id))


    def _exit_view(self):
        from organoid_tracker.visualizer.link_and_position_editor import LinkAndPositionEditor
        data_editor = LinkAndPositionEditor(self._window)
        activate(data_editor)

    def _mark_spline_as_type(self, axis_marker: Marker):
        selected_spline = self._get_selected_spline_of_current_time_point()

        if selected_spline is None:
            self.update_status("No spline selected - cannot set type")
            return

        spline_id = self._selected_spline_ref.spline_id
        old_marker_name = self._experiment.splines.get_marker_name(spline_id)
        old_marker = self._window.registry.get_marker_by_save_name(old_marker_name)
        self._perform_action(_SetSplineMarkerAction(spline_id, axis_marker, old_marker))

    def _mark_spline_checkpoint_as_type(self, checkpoint_marker: Optional[Marker]):
        spline = self._get_selected_spline_of_current_time_point()
        checkpoint = self._get_selected_checkpoint_of_current_time_point()
        if spline is None or checkpoint is None:
            self.update_status("No spline checkpoint selected - cannot set type.")
            return
        old_marker_name = checkpoint.save_name
        self._last_used_checkpoint_type = None if checkpoint_marker is None else checkpoint_marker.save_name
        self._perform_action(_SetCheckpointMarkerAction(spline=spline, checkpoint=checkpoint, new_marker=checkpoint_marker))

    def _set_reference_time_point(self):
        """Asks the user for a new reference time point."""
        min_time_point_number = self._experiment.first_time_point_number()
        max_time_point_number = self._experiment.last_time_point_number()
        if min_time_point_number is None or max_time_point_number is None:
            raise UserError("Reference time point", "No data is loaded - cannot change reference time point")
        reference_time_point = self._experiment.splines.reference_time_point()
        if reference_time_point is None:
            reference_time_point = TimePoint(min_time_point_number)
        explanation = "Splines are used to follow positions over time across a trajectory. If you have multiple\n" \
                      " of such trajectories, then all positions need to be assigned to one of these splines.\n\n" \
                      f"Currently, each cell belongs to the spline that was the closest by in time point" \
                      f" {reference_time_point.time_point_number()}. Which\ntime point should it be instead?" \
                      f" ({min_time_point_number}-{max_time_point_number}, inclusive)"
        answer = dialog.prompt_int("Reference time point", explanation, minimum=min_time_point_number,
                                   maximum=max_time_point_number, default=self._time_point.time_point_number())
        if answer is None:
            return
        self._perform_action(_SetReferenceTimePointAction(reference_time_point, TimePoint(answer)))
