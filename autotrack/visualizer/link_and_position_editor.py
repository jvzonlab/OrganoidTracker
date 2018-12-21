from typing import Optional, List, Set

from matplotlib.backend_bases import KeyEvent, MouseEvent, LocationEvent

from autotrack import core
from autotrack.core import TimePoint
from autotrack.core.experiment import Experiment
from autotrack.core.positions import Position
from autotrack.core.shape import ParticleShape
from autotrack.gui.window import Window
from autotrack.linking_analysis import cell_error_finder, linking_markers
from autotrack.linking_analysis.linking_markers import EndMarker
from autotrack.visualizer import DisplaySettings, activate
from autotrack.visualizer.abstract_editor import AbstractEditor
from autotrack.gui.undo_redo import UndoableAction, ReversedAction


class _InsertLinkAction(UndoableAction):
    """Used to insert a link between two positions."""
    position1: Position
    position2: Position

    def __init__(self, position1: Position, position2: Position):
        self.position1 = position1
        self.position2 = position2
        if position1.time_point_number() == position2.time_point_number():
            raise ValueError(f"The {position1} is at the same time point as {position2}")

    def do(self, experiment: Experiment):
        experiment.links.add_link(self.position1, self.position2)
        cell_error_finder.apply_on(experiment, self.position1, self.position2)
        return f"Inserted link between {self.position1} and {self.position2}"

    def undo(self, experiment: Experiment):
        experiment.links.remove_link(self.position1, self.position2)
        cell_error_finder.apply_on(experiment, self.position1, self.position2)
        return f"Removed link between {self.position1} and {self.position2}"


class _InsertPositionAction(UndoableAction):
    """Used to insert a position."""

    position: Position
    linked_positions: List[Position]

    def __init__(self, position: Position, linked_positions: List[Position]):
        self.position = position
        self.linked_positions = linked_positions

    def do(self, experiment: Experiment):
        experiment.positions.add(self.position)
        for linked_position in self.linked_positions:
            experiment.links.add_link(self.position, linked_position)
        cell_error_finder.apply_on(experiment, self.position, *self.linked_positions)

        return_value = f"Added {self.position}"
        if len(self.linked_positions) > 1:
            return_value += " with connections to " + (" and ".join((str(p) for p in self.linked_positions)))
        if len(self.linked_positions) == 1:
            return_value += f" with a connection to {self.linked_positions[0]}"

        return return_value + "."

    def undo(self, experiment: Experiment):
        experiment.remove_position(self.position)
        cell_error_finder.apply_on(experiment, *self.linked_positions)
        return f"Removed {self.position}"


class _MovePositionAction(UndoableAction):
    """Used to move a position"""

    old_position: Position
    old_shape: ParticleShape
    new_position: Position

    def __init__(self, old_position: Position, old_shape: ParticleShape, new_position: Position):
        if old_position.time_point_number() != new_position.time_point_number():
            raise ValueError(f"{old_position} and {new_position} are in different time points")
        self.old_position = old_position
        self.old_shape = old_shape
        self.new_position = new_position

    def do(self, experiment: Experiment):
        experiment.move_position(self.old_position, self.new_position)
        cell_error_finder.apply_on(experiment, self.new_position)
        return f"Moved {self.old_position} to {self.new_position}"

    def undo(self, experiment: Experiment):
        experiment.move_position(self.new_position, self.old_position)
        experiment.positions.add(self.old_position, self.old_shape)
        cell_error_finder.apply_on(experiment, self.old_position)
        return f"Moved {self.new_position} back to {self.old_position}"


class _MarkLineageEndAction(UndoableAction):
    """Used to add a marker to the end of a lineage."""

    marker: Optional[EndMarker]  # Set to None to erase a marker
    old_marker: Optional[EndMarker]
    position: Position

    def __init__(self, position: Position, marker: Optional[EndMarker], old_marker: Optional[EndMarker]):
        self.position = position
        self.marker = marker
        self.old_marker = old_marker

    def do(self, experiment: Experiment) -> str:
        linking_markers.set_track_end_marker(experiment.links, self.position, self.marker)
        cell_error_finder.apply_on(experiment, self.position)
        if self.marker is None:
            return f"Removed the lineage end marker of {self.position}"
        return f"Added the {self.marker.get_display_name()}-marker to {self.position}"

    def undo(self, experiment: Experiment):
        linking_markers.set_track_end_marker(experiment.links, self.position, self.old_marker)
        cell_error_finder.apply_on(experiment, self.position)
        if self.old_marker is None:
            return f"Removed the lineage end marker again of {self.position}"
        return f"Re-added the {self.old_marker.get_display_name()}-marker to {self.position}"


class LinkAndPositionEditor(AbstractEditor):
    """Editor for cell links and positions. Use the Insert key to insert new cells or links, and Delete to delete
     them."""

    _selected1: Optional[Position] = None
    _selected2: Optional[Position] = None

    def __init__(self, window: Window, *, time_point: Optional[TimePoint] = None, z: int = 14,
                 display_settings: Optional[DisplaySettings] = None, selected_position: Optional[Position] = None):
        super().__init__(window, time_point=time_point, z=z, display_settings=display_settings)

        self._selected1 = selected_position

    def _draw_extra(self):
        if self._selected1 is not None and not self._experiment.positions.exists(self._selected1):
            self._selected1 = None
        if self._selected2 is not None and not self._experiment.positions.exists(self._selected2):
            self._selected2 = None

        self._draw_highlight(self._selected1)
        self._draw_highlight(self._selected2)

    def _draw_highlight(self, position: Optional[Position]):
        if position is None:
            return
        color = core.COLOR_CELL_CURRENT
        if position.time_point_number() < self._time_point.time_point_number():
            color = core.COLOR_CELL_PREVIOUS
        elif position.time_point_number() > self._time_point.time_point_number():
            color = core.COLOR_CELL_NEXT
        self._ax.plot(position.x, position.y, 'o', markersize=25, color=(0,0,0,0), markeredgecolor=color,
                      markeredgewidth=5)

    def _on_mouse_click(self, event: MouseEvent):
        if not event.dblclick:
            return
        new_selection = self._get_position_at(event.xdata, event.ydata)
        if new_selection is None:
            self._selected1, self._selected2 = None, None
            self.draw_view()
            self.update_status("Cannot find a cell here. Unselected both cells.")
            return
        if new_selection == self._selected1:
            self._selected1 = None  # Deselect
        elif new_selection == self._selected2:
            self._selected2 = None  # Deselect
        else:
            self._selected2 = self._selected1
            self._selected1 = new_selection
        self.draw_view()
        self.update_status("Selected:\n        " + str(self._selected1) + "\n        " + str(self._selected2))

    def get_extra_menu_options(self):
        return {
            **super().get_extra_menu_options(),
            "Edit//Experiment-Edit data axes... (A)": self._show_path_editor,
            "View//Linking-Linking errors and warnings (E)": self._show_linking_errors,
            "View//Linking-Lineage errors and warnings (L)": self._show_lineage_errors,
            "Edit//LineageEnd-Mark as cell death": lambda: self._try_set_end_marker(EndMarker.DEAD),
            "Edit//LineageEnd-Mark as moving out of view": lambda: self._try_set_end_marker(EndMarker.OUT_OF_VIEW),
            "Edit//LineageEnd-Remove end marker": lambda: self._try_set_end_marker(None)
        }

    def _on_key_press(self, event: KeyEvent):
        if event.key == "c":
            self._exit_view()
        elif event.key == "e":
            position = self._get_position_at(event.xdata, event.ydata)
            self._show_linking_errors(position)
        elif event.key == "l":
            self._show_lineage_errors()
        elif event.key == "a":
            self._show_path_editor()
        elif event.key == "insert":
            self._try_insert(event)
        elif event.key == "delete":
            self._try_delete()
        elif event.key == "shift":
            if self._selected1 is None or self._selected2 is not None:
                self.update_status("You need to have exactly one cell selected in order to move a cell.")
            elif self._selected1.time_point() != self._time_point:
                self.update_status(f"Cannot move {self._selected1} to this time point.")
            else:
                old_shape = self._experiment.positions.get_shape(self._selected1)
                new_position = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
                old_position = self._selected1
                self._selected1 = None
                self._perform_action(_MovePositionAction(old_position, old_shape, new_position))
        else:
            super()._on_key_press(event)

    def _try_delete(self):
        if self._selected1 is None:
            self.update_status("You need to select a cell first")
        elif self._selected2 is None:  # Delete cell and its links
            old_links = self._experiment.links.find_links_of(self._selected1)
            self._perform_action(ReversedAction(_InsertPositionAction(self._selected1, list(old_links))))
        elif self._experiment.links.contains_link(self._selected1, self._selected2):  # Delete link between cells
            position1, position2 = self._selected1, self._selected2
            self._selected1, self._selected2 = None, None
            self._perform_action(ReversedAction(_InsertLinkAction(position1, position2)))
        else:
            self.update_status("No link found between the two positions - nothing to delete")

    def _try_set_end_marker(self, marker: Optional[EndMarker]):
        if self._selected1 is None or self._selected2 is not None:
            self.update_status("You need to have exactly one cell selected in order to move a cell.")
            return

        links = self._experiment.links
        if len(links.find_futures(self._selected1)) > 0:
            self.update_status(f"The {self._selected1} is not a lineage end.")
            return
        current_marker = linking_markers.get_track_end_marker(links, self._selected1)
        if current_marker == marker:
            if marker is None:
                self.update_status("There is no lineage ending marker here, cannot delete anything.")
            else:
                self.update_status(f"This lineage end already has the {marker.get_display_name()} marker.")
            return
        self._perform_action(_MarkLineageEndAction(self._selected1, marker, current_marker))

    def _show_path_editor(self):
        from autotrack.visualizer.data_axis_editor import DataAxisEditor
        path_editor = DataAxisEditor(self._window, time_point=self._time_point, z=self._z,
                                     display_settings=self._display_settings)
        activate(path_editor)

    def _show_linking_errors(self, position: Optional[Position] = None):
        from autotrack.visualizer.errors_visualizer import ErrorsVisualizer
        warnings_visualizer = ErrorsVisualizer(self._window, position)
        activate(warnings_visualizer)

    def _show_lineage_errors(self):
        from autotrack.visualizer.lineage_errors_visualizer import LineageErrorsVisualizer
        editor = LineageErrorsVisualizer(self._window, time_point=self._time_point, z=self._z)
        activate(editor)

    def _try_insert(self, event: LocationEvent):
        if self._selected1 is None or self._selected2 is None:
            # Insert new position
            position = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
            connections = []
            if self._selected1 is not None \
                    and self._selected1.time_point_number() != self._time_point.time_point_number():
                connections.append(self._selected1)  # Add link to already selected position

            self._selected1 = position
            self._perform_action(_InsertPositionAction(position, connections))
        elif self._selected1.time_point_number() == self._selected2.time_point_number():
            self.update_status("The two selected cells are in exactly the same time point - cannot insert link.")
        elif abs(self._selected1.time_point_number() - self._selected2.time_point_number()) > 1:
            self.update_status("The two selected cells are not in consecutive time points - cannot insert link.")
        else:
            # Insert link between two positions
            position1, position2 = self._selected1, self._selected2
            self._selected1, self._selected2 = None, None
            self._perform_action(_InsertLinkAction(position1, position2))

