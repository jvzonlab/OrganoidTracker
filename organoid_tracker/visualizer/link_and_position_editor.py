from typing import Optional, List, Dict, Iterable, Tuple

from matplotlib.backend_bases import KeyEvent, MouseEvent, LocationEvent

from organoid_tracker import core
from organoid_tracker.core import Color, UserError, TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.full_position_snapshot import FullPositionSnapshot
from organoid_tracker.core.link_data import LinkData
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.marker import Marker
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.core.typing import DataType
from organoid_tracker.gui import dialog
from organoid_tracker.gui.undo_redo import UndoableAction, ReversedAction
from organoid_tracker.gui.window import Window
from organoid_tracker.linking_analysis import cell_error_finder, linking_markers, track_positions_finder, \
    lineage_markers
from organoid_tracker.linking_analysis.linking_markers import EndMarker
from organoid_tracker.position_analysis import position_markers
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.abstract_editor import AbstractEditor


class _InsertLinkAction(UndoableAction):
    """Used to insert a link between two positions. Will add interpolated positions in between if necessary."""
    all_positions = List[Position]

    def __init__(self, position1: Position, position2: Position):
        self.all_positions = position1.interpolate(position2)

    def do(self, experiment: Experiment):
        previous_position = None

        # Add the interpolated positions
        for position in self.all_positions[1:-1]:
            experiment.positions.add(position)

        # Add links
        for position in self.all_positions:
            if previous_position is not None:
                experiment.links.add_link(position, previous_position)
            previous_position = position

        cell_error_finder.find_errors_in_positions_links_and_all_dividing_cells(experiment, *self.all_positions)
        return f"Inserted link between {self.all_positions[0]} and {self.all_positions[-1]}"

    def undo(self, experiment: Experiment):
        if len(self.all_positions) == 2:
            # Remove just a link
            experiment.links.remove_link(*self.all_positions)
        else:
            # Remove links and interpolated positions
            for position in self.all_positions[1:-1]:
                experiment.remove_position(position)

        cell_error_finder.find_errors_in_positions_links_and_all_dividing_cells(experiment, self.all_positions[0],
                                                                                self.all_positions[-1])
        return f"Removed link between {self.all_positions[0]} and {self.all_positions[-1]}"


class _DeleteLinksAction(UndoableAction):
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
        return f"Removed {len(self.position_pairs)} links"

    def undo(self, experiment: Experiment) -> str:
        for position1, position2, data in self.position_pairs:
            experiment.links.add_link(position1, position2)
            for data_key, data_value in data.items():
                experiment.link_data.set_link_data(position1, position2, data_key, data_value)
        return f"Inserted {len(self.position_pairs)} links"


class _InsertPositionAction(UndoableAction):
    """Used to insert a position."""

    particle: FullPositionSnapshot

    def __init__(self, particle: FullPositionSnapshot):
        self.particle = particle

    def do(self, experiment: Experiment) -> str:
        self.particle.restore(experiment)
        cell_error_finder.find_errors_in_positions_links_and_all_dividing_cells(experiment, self.particle.position,
                                                                                *self.particle.links)

        return_value = f"Added {self.particle.position}"
        if len(self.particle.links) > 1:
            return_value += " with links to " + (" and ".join((str(p) for p in self.particle.links)))
        if len(self.particle.links) == 1:
            return_value += f" with a link to {self.particle.links[0]}"

        return return_value + "."

    def undo(self, experiment: Experiment) -> str:
        experiment.remove_position(self.particle.position)
        cell_error_finder.find_errors_in_positions_links_and_all_dividing_cells(experiment, *self.particle.links)
        return f"Removed {self.particle.position}"


class _DeletePositionsAction(UndoableAction):
    _snapshots: List[FullPositionSnapshot]

    def __init__(self, snapshots: Iterable[FullPositionSnapshot]):
        self._snapshots = list(snapshots)

    def do(self, experiment: Experiment):
        experiment.remove_positions((particle.position for particle in self._snapshots))
        for particle in self._snapshots:  # Check linked particles for errors
            cell_error_finder.find_errors_in_just_these_positions(experiment, *particle.links)
        cell_error_finder.find_errors_in_all_dividing_cells(experiment)
        return f"Removed {len(self._snapshots)} positions"

    def undo(self, experiment: Experiment):
        for particle in self._snapshots:
            particle.restore(experiment)
            cell_error_finder.find_errors_in_just_these_positions(experiment, particle.position, *particle.links)
        cell_error_finder.find_errors_in_all_dividing_cells(experiment)
        return f"Added {len(self._snapshots)} positions"


class _MovePositionAction(UndoableAction):
    """Used to move a position"""

    old_position: Position
    new_position: Position

    # Probabilities of the link to X. Removed upon move, restored upon undo
    old_link_probabilities: Dict[Position, float]

    def __init__(self, experiment: Experiment, old_position: Position, new_position: Position):
        if old_position.time_point_number() != new_position.time_point_number():
            raise ValueError(f"{old_position} and {new_position} are in different time points")
        self.old_position = old_position
        self.new_position = new_position

        # Collect old link probabilities (for the undo functionality)
        self.old_link_probabilities = dict()
        for link in experiment.links.find_links_of(self.old_position):
            self.old_link_probabilities[link] = experiment.link_data.get_link_data(self.old_position, link,
                                                                                   "link_probability")

    def do(self, experiment: Experiment):
        experiment.move_position(self.old_position, self.new_position)

        # Remove link probability, as it's no longer correct
        for link in experiment.links.find_links_of(self.new_position):
            experiment.link_data.set_link_data(self.new_position, link, "link_probability", None)

        # Recheck errors
        cell_error_finder.find_errors_in_positions_links_and_all_dividing_cells(experiment, self.new_position)
        return f"Moved {self.old_position} to {self.new_position}"

    def undo(self, experiment: Experiment):
        experiment.move_position(self.new_position, self.old_position)
        for link in experiment.links.find_links_of(self.old_position):
            experiment.link_data.set_link_data(self.old_position, link, "link_probability",
                                               self.old_link_probabilities.get(link))
        cell_error_finder.find_errors_in_positions_links_and_all_dividing_cells(experiment, self.old_position)
        return f"Moved {self.new_position} back to {self.old_position}"


class _MoveMultiplePositionsAction(UndoableAction):
    """Used to move multiple positions by a (dx, dy, dz) offset."""

    old_positions: List[Position]
    dx: float
    dy: float
    dz: float

    def __init__(self, old_positions: List[Position], *, dx: float = 0, dy: float = 0, dz: float = 0):
        if len(old_positions) == 0:
            raise ValueError("No positions supplied")
        self.old_positions = list(old_positions)
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def do(self, experiment: Experiment):
        new_positions = [position.with_offset(self.dx, self.dy, self.dz) for position in self.old_positions]
        for old_position, new_position in zip(self.old_positions, new_positions):
            experiment.move_position(old_position, new_position)

        # Recheck errors
        cell_error_finder.find_errors_in_positions_links_and_all_dividing_cells(experiment, *new_positions)
        return f"Moved {len(new_positions)} position(s) by ({self.dx:01}, {self.dy:01}, {self.dz:01})"

    def undo(self, experiment: Experiment):
        new_positions = [position.with_offset(self.dx, self.dy, self.dz) for position in self.old_positions]
        for old_position, new_position in zip(self.old_positions, new_positions):
            experiment.move_position(new_position, old_position)

        # Recheck errors
        cell_error_finder.find_errors_in_positions_links_and_all_dividing_cells(experiment, *self.old_positions)
        return f"Moved {len(new_positions)} position(s) back by ({self.dx:01}, {self.dy:01}, {self.dz:01})"


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
        linking_markers.set_track_end_marker(experiment.position_data, self.position, self.marker)
        cell_error_finder.find_errors_in_positions_links_and_all_dividing_cells(experiment, self.position)
        if self.marker is None:
            return f"Removed the lineage end marker of {self.position}"
        return f"Added the {self.marker.get_display_name()}-marker to {self.position}"

    def undo(self, experiment: Experiment):
        linking_markers.set_track_end_marker(experiment.position_data, self.position, self.old_marker)
        cell_error_finder.find_errors_in_positions_links_and_all_dividing_cells(experiment, self.position)
        if self.old_marker is None:
            return f"Removed the lineage end marker again of {self.position}"
        return f"Re-added the {self.old_marker.get_display_name()}-marker to {self.position}"


class _InsertConnectionAction(UndoableAction):
    _position1: Position
    _position2: Position

    def __init__(self, position1: Position, position2: Position):
        self._position1 = position1
        self._position2 = position2

    def do(self, experiment: Experiment) -> str:
        experiment.connections.add_connection(self._position1, self._position2)
        return f"Added connection between {self._position1} and {self._position2}"

    def undo(self, experiment: Experiment):
        experiment.connections.remove_connection(self._position1, self._position2)
        return f"Removed connection between {self._position1} and {self._position2}"


class _SetAllAsType(UndoableAction):
    _previous_position_types: Dict[Position, str]
    _type: Optional[Marker]

    def __init__(self, previous_position_types: Dict[Position, str], new_type: Optional[Marker]):
        self._previous_position_types = previous_position_types
        self._type = new_type

    def do(self, experiment: Experiment) -> str:
        position_data = experiment.position_data
        save_name = self._type.save_name if self._type is not None else None
        for position in self._previous_position_types.keys():
            position_markers.set_position_type(position_data, position, save_name)
        position_count = len(self._previous_position_types.keys())
        if self._type is None:
            return f"Removed the type of {position_count} position(s)"
        if position_count == 1:
            return f"Set the type of the selected position to \"{self._type.display_name}\""
        return f"{position_count} positions are now of the type \"{self._type.display_name}\""

    def undo(self, experiment: Experiment) -> str:
        position_data = experiment.position_data
        for position in self._previous_position_types.keys():
            position_markers.set_position_type(position_data, position, self._previous_position_types.get(position))
        return f"Reset all positions to their previous type"


class _SetLineageColor(UndoableAction):
    _track: LinkingTrack
    _old_color: Color
    _new_color: Color

    def __init__(self, track: LinkingTrack, old_color: Color, new_color: Color):
        self._track = track
        self._old_color = old_color
        self._new_color = new_color

    def do(self, experiment: Experiment) -> str:
        lineage_markers.set_color(experiment.links, self._track, self._new_color)
        if self._new_color.is_black():
            return "Removed the color of the lineage"
        return f"Set the color of the lineage to {self._new_color}"

    def undo(self, experiment: Experiment) -> str:
        lineage_markers.set_color(experiment.links, self._track, self._old_color)
        if self._old_color.is_black():
            return "Removed the color of the lineage again"
        return f"Changed the color of the linage back to {self._old_color}"


class _OverwritePositionAction(UndoableAction):
    _new_particle: FullPositionSnapshot
    _old_particle: FullPositionSnapshot

    def __init__(self, new_particle: FullPositionSnapshot, old_particle: FullPositionSnapshot):
        self._new_particle = new_particle
        self._old_particle = old_particle

    def do(self, experiment: Experiment) -> str:
        experiment.remove_position(self._old_particle.position)
        self._new_particle.restore(experiment)
        cell_error_finder.find_errors_in_just_these_positions(experiment, *self._old_particle.links)
        cell_error_finder.find_errors_in_positions_links_and_all_dividing_cells(experiment, self._new_particle.position)
        return f"Overwritten {self._old_particle.position} with {self._new_particle.position}"

    def undo(self, experiment: Experiment) -> str:
        experiment.remove_position(self._new_particle.position)
        self._old_particle.restore(experiment)
        cell_error_finder.find_errors_in_just_these_positions(experiment, *self._new_particle.links)
        cell_error_finder.find_errors_in_positions_links_and_all_dividing_cells(experiment, self._old_particle.position)
        return f"Restored {self._old_particle.position}"


class _MarkPositionAsSomethingAction(UndoableAction):
    _positions: List[Position]
    _name: str

    def __init__(self, positions: List[Position], data_name: str):
        self._positions = list(positions)
        self._name = data_name

    def do(self, experiment: Experiment) -> str:
        for position in self._positions:
            experiment.position_data.set_position_data(position, self._name, True)
        if self._name == linking_markers.UNCERTAIN_MARKER:
            cell_error_finder.find_errors_in_just_these_positions(experiment, *self._positions)
        if len(self._positions) == 1:
            return f"Marked {self._positions[0]} as {self._name}"
        return f"Marked {len(self._positions)} positions as {self._name}"

    def undo(self, experiment: Experiment) -> str:
        for position in self._positions:
            experiment.position_data.set_position_data(position, self._name, None)
        if self._name == linking_markers.UNCERTAIN_MARKER:
            cell_error_finder.find_errors_in_just_these_positions(experiment, *self._positions)
        if len(self._positions) == 1:
            return f"Marked that {self._positions[0]} is no longer {self._name}"
        return f"Marked that {len(self._positions)} positions are no longer {self._name}"


class LinkAndPositionEditor(AbstractEditor):
    """Editor for cell links and positions. Use the Insert or Enter key to insert new cells or links, and use the Delete
     or Backspace key to delete them."""

    _selected: List[Position]

    def __init__(self, window: Window, *, selected_positions: Iterable[Position] = ()):
        super().__init__(window)

        self._selected = list(selected_positions)

    def _get_figure_title(self) -> str:
        title_start = "Editing time point " + str(self._time_point.time_point_number()) + "    (z=" + str(
            self._z) + ")\n"
        if len(self._selected) == 1:
            return title_start + "1 position selected"
        elif len(self._selected) > 1:
            time_point_count = len({position.time_point_number() for position in self._selected})
            time_point_text = "1 time point" if time_point_count == 1 else f"{time_point_count} time points"
            return title_start + f"{len(self._selected)} positions selected across " + time_point_text
        else:
            return title_start

    def _draw_extra(self):
        to_unselect = set()
        for i in range(len(self._selected)):
            selected = self._selected[i]
            if self._experiment.positions.contains_position(selected):
                self._draw_highlight(selected)
            else:
                to_unselect.add(selected)  # Position doesn't exist anymore, remove from selection

        # Unselect positions that don't exist anymore
        if len(to_unselect) > 0:
            self._selected = [element for element in self._selected if element not in to_unselect]

    def _draw_highlight(self, position: Optional[Position]):
        if position is None or abs(position.time_point_number() - self._time_point.time_point_number()) > 2:
            return
        color = core.COLOR_CELL_CURRENT
        if position.time_point_number() < self._time_point.time_point_number():
            color = core.COLOR_CELL_PREVIOUS
        elif position.time_point_number() > self._time_point.time_point_number():
            color = core.COLOR_CELL_NEXT
        self._draw_selection(position, color)

    def _on_mouse_click(self, event: MouseEvent):
        if not event.dblclick:
            return
        new_selection = self._get_position_at(event.xdata, event.ydata)
        if new_selection is None:
            self._selected.clear()
            self.draw_view()
            self.update_status("Cannot find a cell here. Unselected all cells.")
            return

        if new_selection in self._selected:
            self._selected.remove(new_selection)  # Deselect
        else:
            self._selected.append(new_selection)  # Select
        self.draw_view()

        if len(self._selected) <= 2:
            selected_first = self._selected[0] if len(self._selected) > 0 else None
            selected_second = self._selected[1] if len(self._selected) > 1 else None
            self.update_status(
                "Selected:\n        " + self._position_string(selected_first) + "\n        " + self._position_string(
                    selected_second))
        else:
            self.update_status(f"Selected: {len(self._selected)} positions")

    def _position_string(self, position: Optional[Position]) -> str:
        """Gets a somewhat compact representation of the position. This will return its x, y, z, time point and some
        metadata."""
        if position is None:
            return "none"
        return_value = str(position)

        data_names = list()
        for data_name, value in self._experiment.position_data.find_all_data_of_position(position):
            if value:
                data_names.append("'" + data_name + "'")
        if len(data_names) > 10:
            data_names = data_names[0:10]
            data_names.append("...")
        if len(data_names) > 0:
            return_value += " (with " + ", ".join(data_names) + ")"
        return return_value

    def get_extra_menu_options(self):
        options = {
            **super().get_extra_menu_options(),
            "Edit//Experiment-Edit splines... [A]": self._show_spline_editor,
            "Edit//Experiment-Edit beacons... [B]": self._show_beacon_editor,
            "Edit//Experiment-Edit image offsets... [O]": self._show_offset_editor,
            "Edit//Batch-Delete selected positions... [Ctrl+Delete]": self._try_delete_all_selected,
            "Edit//Batch-Batch deletion//Delete data of current time point": self._delete_data_of_time_point,
            "Edit//Batch-Batch deletion//Delete data of multiple time points...": self._delete_data_of_multiple_time_points,
            "Edit//Batch-Batch deletion//Delete all tracks with errors...": self._delete_tracks_with_errors,
            "Edit//Batch-Batch deletion//Delete short lineages...": self._delete_short_lineages,
            "Edit//Batch-Batch deletion//Delete all tracks not in the first time point...": self._delete_tracks_not_in_first_time_point,
            "Edit//Batch-Batch deletion//Delete all positions without links...": self._delete_positions_without_links,
            "Edit//Batch-Batch deletion//Delete all links with low likelihood...": self._delete_unlikely_links,
            "Edit//LineageEnd-Mark as cell death [D]": lambda: self._try_set_end_marker(EndMarker.DEAD),
            "Edit//LineageEnd-Mark as cell shedding into lumen [S]": lambda: self._try_set_end_marker(EndMarker.SHED),
            "Edit//LineageEnd-Mark as cell shedding to outside": lambda: self._try_set_end_marker(
                EndMarker.SHED_OUTSIDE),
            "Edit//LineageEnd-Mark as cell stimulated shedding": lambda: self._try_set_end_marker(
                EndMarker.STIMULATED_SHED),
            "Edit//LineageEnd-Mark as moving out of view [V]": lambda: self._try_set_end_marker(EndMarker.OUT_OF_VIEW),
            "Edit//LineageEnd-Remove end marker": lambda: self._try_set_end_marker(None),
            "Edit//Marker-Set color of lineage...": self._set_color_of_lineage,
            "Select//Select-All positions in current time point [Ctrl+A]": self._select_all,
            "Select//Select-Expand selection to entire track [T]": self._select_track,
            "Select//Select-Select positions in a rectangle...": self._show_positions_in_rectangle_selector,
            "View//Linking-Linking errors and warnings (E)": self._show_linking_errors,
            "View//Linking-Lineage errors and warnings": self._show_lineage_errors,
            "Navigate//Layer-Layer of selected position [Space]": self._move_to_z_of_selected_position,
        }

        # Add options for adding position flags
        for flag_name in position_markers.get_position_flags(self._experiment):
            # Create copy of flag_name variable to avoid it changing in loop iteration
            add_action = lambda bound_flag_name=flag_name: self._try_mark_as(bound_flag_name, True)
            options["Edit//Marker-Flag position as//Type-" + flag_name] = add_action
        options["Edit//Marker-Flag position as//Action-New flag..."] = lambda: self._try_mark_as(None, True)
        for flag_name in position_markers.get_position_flags(self._experiment):
            # Create copy of flag_name variable to avoid it changing in loop iteration
            remove_action = lambda bound_flag_name=flag_name: self._try_mark_as(bound_flag_name, False)
            options["Edit//Marker-Flag position as//Action-Remove flag//Type-" + flag_name] = remove_action

        # Add options for changing position types
        for position_type in self.get_window().registry.get_registered_markers(Position):
            # Create copy of position_type variable to avoid it changing in loop iteration
            track_action = lambda bound_position_type=position_type: self._set_track_to_type(bound_position_type)
            options["Edit//Marker-Set type of track//Type-" + position_type.display_name] = track_action
            position_action = lambda bound_position_type=position_type: self._set_position_to_type(bound_position_type)
            options["Edit//Marker-Set type of position//Type-" + position_type.display_name] = position_action
        options["Edit//Marker-Set type of track//Clear-Remove type"] = lambda: self._set_track_to_type(None)
        options["Edit//Marker-Set type of position//Clear-Remove type"] = lambda: self._set_position_to_type(None)
        return options

    def _on_key_press(self, event: KeyEvent):
        if event.key == "c":
            self._exit_view()
        elif event.xdata is None or event.ydata is None:
            # Rest of the keys requires a mouse position
            super()._on_key_press(event)
        elif event.key == "e":
            position = self._get_position_at(event.xdata, event.ydata)
            self._show_linking_errors(position)
        elif event.key == "insert" or event.key == "enter":
            self._try_insert(event)
        elif event.key == "delete" or event.key == "backspace":
            self._try_delete()
        elif event.key == "ctrl+backspace":
            self._try_delete_all_selected()  # Ctrl + Delete also works, but is registered using the menu
        elif event.key == "alt+a":
            self._try_move_selected(dx=-1)
        elif event.key == "alt+s":
            self._try_move_selected(dy=1)
        elif event.key == "alt+d":
            self._try_move_selected(dx=1)
        elif event.key == "alt+w":
            self._try_move_selected(dy=-1)
        elif event.key == "alt+q":
            self._try_move_selected(dz=-1)
        elif event.key == "alt+e":
            self._try_move_selected(dz=1)
        elif event.key == "shift":
            if len(self._selected) != 1:
                self.update_status("You need to have exactly one position selected in order to move it."
                                   "\n  To move multiple positions at once, use Alt + A/S/D/W/Q/E.")
            elif self._selected[0].time_point() != self._time_point:
                self.update_status(f"Cannot move {self._selected[0]} to this time point.")
            else:
                new_position = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
                old_position = self._selected[0]
                self._selected.clear()
                self._perform_action(_MovePositionAction(self._experiment, old_position, new_position))
        else:
            super()._on_key_press(event)

    def _select_all(self):
        self._selected = list(self._experiment.positions.of_time_point(self._time_point))
        self.draw_view()
        self.update_status(f"Selected all {len(self._selected)} positions of this time point.")

    def _select_track(self):
        if len(self._selected) == 0:
            self.update_status("No positions selected. Cannot expand selection to entire track.")
            return

        to_select = set()
        for position in self._selected:
            track_of_position = self._experiment.links.get_track(position)
            if track_of_position is None:
                to_select.add(position)
                continue
            for track in track_of_position.find_all_previous_and_descending_tracks(include_self=True):
                for some_position in track.positions():
                    to_select.add(some_position)
        difference_count = len(to_select) - len(self._selected)
        if difference_count == 0:
            self.update_status("Couldn't add any positions to the selection - any linked positions are already added.")
            return
        self._selected = list(to_select)
        self.draw_view()
        self.update_status(f"Added all {difference_count} positions that came before or after the selected positions.")

    def _move_to_position(self, position: Position) -> bool:
        if position not in self._selected:
            # Select that position
            self._selected.append(position)
        return super()._move_to_position(position)

    def _move_to_z_of_selected_position(self):
        if len(self._selected) == 0:
            self.update_status("No position selected - cannot move to its z-layer.")
            return
        if len(self._selected) == 1:
            self._move_to_z(int(round(self._selected[0].z)))
            self.update_status("Moved to z-pos of selected position.")
            return
        z_values = [position.z for position in self._selected]
        z_mean = sum(z_values) / len(z_values)
        self._move_to_z(int(round(z_mean)))
        self.update_status("Moved to average z-pos of selected positions.")

    def _try_delete(self):
        if len(self._selected) == 0:
            self.update_status("You need to select a cell first")
        elif len(self._selected) == 1:  # Delete cell and its links
            snapshot_to_delete = FullPositionSnapshot.from_position(self._experiment, self._selected[0])
            self._perform_action(ReversedAction(_InsertPositionAction(snapshot_to_delete)))
        elif len(self._selected) == 2:
            if self._experiment.connections.contains_connection(self._selected[0],
                                                                self._selected[1]):  # Delete a connection
                position1, position2 = self._selected
                self._selected.clear()
                self._perform_action(ReversedAction(_InsertConnectionAction(position1, position2)))
            elif self._experiment.links.contains_link(self._selected[0],
                                                      self._selected[1]):  # Delete link between cells
                position1, position2 = self._selected
                self._selected.clear()
                self._perform_action(ReversedAction(_InsertLinkAction(position1, position2)))
            else:
                self.update_status("No link found between the two positions - nothing to delete.")
        else:
            self.update_status("Select a single position to delete it, or select two positions to delete the link or"
                               " connection between them.\n    To delete all selected positions at once, press"
                               " Ctrl+Delete.")

    def _try_delete_all_selected(self):
        if len(self._selected) == 0:
            self.update_status("No positions selected - cannot delete anything.")
            return
        snapshots = [FullPositionSnapshot.from_position(self._experiment, position) for position in self._selected]
        self._perform_action(_DeletePositionsAction(snapshots))

    def _try_set_end_marker(self, marker: Optional[EndMarker]):
        if len(self._selected) != 1:
            self.update_status("You need to have exactly one cell selected in order to set an end marker.")
            return

        links = self._experiment.links
        if len(links.find_futures(self._selected[0])) > 0:
            self.update_status(f"The {self._selected[0]} is not a lineage end.")
            return
        current_marker = linking_markers.get_track_end_marker(self._experiment.position_data, self._selected[0])
        if current_marker == marker:
            if marker is None:
                self.update_status("There is no lineage ending marker here, cannot delete anything.")
            else:
                self.update_status(f"This lineage end already has the {marker.get_display_name()} marker.")
            return
        self._perform_action(_MarkLineageEndAction(self._selected[0], marker, current_marker))

    def _try_mark_as(self, flag_name: Optional[str], new_value: bool):
        """Marks a position as having a certain flag. If the flag_name is None, the user will be prompted for a name."""
        update_menus = flag_name is None
        if flag_name is None:
            flag_name = dialog.prompt_str("Flag name", "As what do you want to flag the position(s)?\n\nYou can pick"
                                                       " any name. Possible examples would be \"ablated\",\n"
                                                       "\"responder\" or \"proliferative\". These flags have no"
                                                       " intrinsic\nmeaning, but you could use them from your own"
                                                       " scripts.")
            if flag_name is None:
                return

        position_data = self._experiment.position_data
        positions_that_need_changing = [selected for selected in self._selected
                                        if bool(position_data.get_position_data(selected, flag_name)) != new_value]
        insert = " is" if len(self._selected) == 1 else "s are"
        if new_value:
            # Mark all as True
            if len(positions_that_need_changing) == 0:
                self.update_status(f"Selected position{insert} already marked as {flag_name}.")
                return
            self._perform_action(_MarkPositionAsSomethingAction(positions_that_need_changing, flag_name))
        else:
            # Mark all as False
            if len(positions_that_need_changing) == 0:
                self.update_status(f"Selected position{insert} not marked as {flag_name}, cannot remove marker.")
                return
            self._perform_action(
                ReversedAction(_MarkPositionAsSomethingAction(positions_that_need_changing, flag_name)))

        if update_menus:
            self._window.redraw_all()  # So that the new flag appears in the menus

    def _show_spline_editor(self):
        from organoid_tracker.visualizer.spline_editor import SplineEditor
        spline_editor = SplineEditor(self._window)
        activate(spline_editor)

    def _show_beacon_editor(self):
        from organoid_tracker.visualizer.beacon_editor import BeaconEditor
        beacon_editor = BeaconEditor(self._window)
        activate(beacon_editor)

    def _show_offset_editor(self):
        from organoid_tracker.visualizer.image_offset_editor import ImageOffsetEditor
        offset_editor = ImageOffsetEditor(self._window)
        activate(offset_editor)

    def _show_linking_errors(self, position: Optional[Position] = None):
        from organoid_tracker.visualizer.errors_visualizer import ErrorsVisualizer
        warnings_visualizer = ErrorsVisualizer(self._window, position)
        activate(warnings_visualizer)

    def _show_lineage_errors(self):
        from organoid_tracker.visualizer.lineage_errors_visualizer import LineageErrorsVisualizer
        editor = LineageErrorsVisualizer(self._window)
        activate(editor)

    def _show_positions_in_rectangle_selector(self):
        from organoid_tracker.visualizer.position_in_rectangle_selector import PositionsInRectangleSelector
        editor = PositionsInRectangleSelector(self._window)
        activate(editor)

    def _delete_data_of_time_point(self):
        """Deletes all annotations of a given time point."""
        positions = self._experiment.positions.of_time_point(self._time_point)
        snapshots = (FullPositionSnapshot.from_position(self._experiment, position) for position in positions)
        self._perform_action(_DeletePositionsAction(snapshots))

    def _delete_data_of_multiple_time_points(self):
        """Deletes all annotations of a given time point range."""
        minimum, maximum = self._experiment.first_time_point_number(), self._experiment.last_time_point_number()
        if minimum is None or maximum is None:
            raise UserError("No data loaded", "No time points found. Did you load any data?")

        time_point_number_start = dialog.prompt_int("Start time point",
                                                    "At which time point should we start the deletion?",
                                                    minimum=minimum, maximum=maximum,
                                                    default=self._time_point.time_point_number())
        if time_point_number_start is None:
            return
        time_point_number_end = dialog.prompt_int("End time point",
                                                  "Up to and including which time point should we delete?",
                                                  minimum=time_point_number_start, maximum=maximum,
                                                  default=time_point_number_start)
        if time_point_number_end is None:
            return
        snapshots = []
        for time_point_number in range(time_point_number_start, time_point_number_end + 1):
            positions = self._experiment.positions.of_time_point(TimePoint(time_point_number))
            snapshots += [FullPositionSnapshot.from_position(self._experiment, position) for position in positions]
        self._perform_action(_DeletePositionsAction(snapshots))

    def _delete_positions_without_links(self):
        """Deletes all positions that have no links."""
        snapshots = []
        links = self._experiment.links
        for position in self._experiment.positions:
            if not links.contains_position(position):
                snapshots.append(FullPositionSnapshot.from_position(self._experiment, position))
        self._perform_action(_DeletePositionsAction(snapshots))

    def _delete_unlikely_links(self):
        """Deletes all links with a low score"""
        cutoff = dialog.prompt_float("Deleting unlikely links",
                                     "What is the minimum required likelihood for a link (0% - 100%)?"
                                     "\nLinks with a lower likelihood, as predicted by the neural network, will be removed.",
                                     minimum=0, maximum=100, decimals=1, default=10)
        if cutoff is None:
            return  # Cancelled
        cutoff_fraction = cutoff / 100
        to_remove = list()
        link_data = self._experiment.link_data
        for (position_a, position_b), value in link_data.find_all_links_with_data("link_probability"):
            if value < cutoff_fraction:
                to_remove.append((position_a, position_b))
        self._perform_action(_DeleteLinksAction(link_data, to_remove))

    def _delete_tracks_with_errors(self):
        """Deletes all lineages where at least a single error was present."""
        if not dialog.prompt_yes_no("Warning", "Are you sure you want to delete all tracks with at least a single error"
                                               " in them? This cannot be undone."):
            return
        self.get_window().get_gui_experiment().undo_redo.clear()

        from organoid_tracker.linking_analysis import lineage_error_finder
        lineage_error_finder.delete_problematic_lineages(self._experiment)

        self.get_window().redraw_data()

    def _delete_short_lineages(self):
        """Deletes all lineages where at least a single error was present."""
        min_time_points = dialog.prompt_int("Deleting short lineages", "For how many time points should a cell be "
                                                                       "visible in order to keep it? Cells (including their offspring) that are"
                                                                       " visible for less time points will be deleted.",
                                            minimum=1, default=4)
        if min_time_points is None:
            return

        snapshots_to_delete = []
        experiment = self._experiment
        links = experiment.links

        # Delete all positions without links (since their track length is 1 by defintion, and they don't always appear
        # in experiment.links)
        for position in self._experiment.positions:
            if not links.contains_position(position):
                snapshots_to_delete.append(FullPositionSnapshot.from_position(experiment, position))

        # Delete all positions from short tracks
        for track in links.find_starting_tracks():
            min_time_point_number = track.min_time_point_number()
            max_time_point_number = max([some_track.max_time_point_number()
                                         for some_track in track.find_all_descending_tracks(include_self=True)])
            duration_time_points = max_time_point_number - min_time_point_number + 1
            if duration_time_points < min_time_points:
                for some_track in track.find_all_descending_tracks(include_self=True):
                    for position in some_track.positions():
                        snapshots_to_delete.append(FullPositionSnapshot.from_position(experiment, position))

        # Perform the deletion
        self._perform_action(_DeletePositionsAction(snapshots_to_delete))

    def _delete_tracks_not_in_first_time_point(self):
        """Deletes all lineages where at least a single error was present."""
        if not dialog.prompt_yes_no("Warning", "Are you sure you want to delete all lineages that do not reach the"
                                               " first time point? This cannot be undone."):
            return
        self.get_window().get_gui_experiment().undo_redo.clear()

        from organoid_tracker.linking_analysis import links_filter
        links_filter.delete_lineages_not_in_first_time_point(self._experiment)

        self.get_window().redraw_data()

    def _try_insert(self, event: LocationEvent):
        if len(self._selected) == 0:
            # Add new position without links
            self._selected.append(Position(event.xdata, event.ydata, self._z, time_point=self._time_point))
            mouse_position = self._get_position_at(event.xdata, event.ydata)
            self._perform_action(_InsertPositionAction(FullPositionSnapshot.just_position(self._selected[0])))
        elif len(self._selected) == 1:
            # Insert new position with link to self._selected[0]
            # Find at which position the mouse is pointing
            mouse_position = self._get_position_at(event.xdata, event.ydata)
            is_new_position = False
            if mouse_position is None or abs(mouse_position.time_point_number() -
                                             self._selected[0].time_point_number()) != 1:
                # Just create a new position, position on mouse is not suitable
                mouse_position = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
                is_new_position = True
                if self._display_settings.show_next_time_point and self._selected[0].time_point() == self._time_point:
                    # Insert in next time point when showing two time points, and inserting in this time point wouldn't
                    # be possible
                    mouse_position = mouse_position.with_time_point_number(mouse_position.time_point_number() + 1)

            if abs(mouse_position.time_point_number() - self._selected[0].time_point_number()) == 1:
                connection = self._selected[0]

                # Insert link from selected point to new point
                if is_new_position:
                    self._selected[0] = mouse_position
                    new_particle = FullPositionSnapshot.position_with_links(mouse_position, links=[connection])
                    self._perform_action(_InsertPositionAction(new_particle))
                else:
                    # New point already exists, overwrite that point
                    old_particle = FullPositionSnapshot.from_position(self._experiment, mouse_position)
                    mouse_position = Position(event.xdata, event.ydata, self._z, time_point=mouse_position.time_point())
                    self._selected[0] = mouse_position
                    new_particle = FullPositionSnapshot.position_with_links(mouse_position, links=[connection,
                                                                                                   *(link for link in
                                                                                                     old_particle.links
                                                                                                     if
                                                                                                     link.time_point() != connection.time_point())])
                    self._perform_action(_OverwritePositionAction(new_particle, old_particle))
            else:
                inserting_position = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
                if inserting_position.distance_um(self._selected[0], ImageResolution.PIXELS) < 10:
                    # Mouse is close to an existing position
                    self.update_status("Cannot insert a position here - too close to already selected position.")
                    return
                self._selected[0] = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
                self._perform_action(_InsertPositionAction(FullPositionSnapshot.just_position(self._selected[0])))
        elif len(self._selected) == 2:
            if self._selected[0].time_point_number() == self._selected[1].time_point_number():
                # Insert connection between two positions
                if self._experiment.connections.contains_connection(self._selected[0], self._selected[1]):
                    self.update_status("A connection already exists between the selected positions.")
                    return
                position1, position2 = self._selected
                self._selected.clear()
                self._perform_action(_InsertConnectionAction(position1, position2))
            else:
                # Insert link between two positions
                if self._experiment.links.contains_link(self._selected[0], self._selected[1]):
                    self.update_status("A link already exists between the selected positions.")
                    return
                position1, position2 = self._selected
                self._selected.clear()
                self._perform_action(_InsertLinkAction(position1, position2))
        else:
            time_point_number_of_selection = self._get_time_point_number_of_selection()
            if time_point_number_of_selection is None:
                self.update_status(f"You currently have {len(self._selected)} positions selected of multiple time"
                                   f" points - cannot insert anything.")
            elif time_point_number_of_selection == self._time_point.time_point_number():
                self.update_status(f"You currently have {len(self._selected)} positions selected of the current time"
                                   f" point. To insert connections between them, select only two at a time.")
            elif abs(time_point_number_of_selection - self._time_point.time_point_number()) > 1:
                self.update_status(f"You currently have {len(self._selected)} positions selected in a time point"
                                   f" further away. Cannot insert them here.")
            else:
                # Copy over to this time point, with links to time_point_number_of_selection
                new_position_snapshots = [FullPositionSnapshot.position_with_links(
                    position=position.with_time_point(self._time_point), links=[position])
                    for position in self._selected]
                for position_snapshot in new_position_snapshots:
                    if self._experiment.positions.contains_position(position_snapshot.position):
                        raise UserError("Position already exists", "Cannot insert the selected positions in this time"
                                                                   " point, as at least one position already exists."
                                                                   " Maybe you copied the positions before?")
                self._selected = [snapshot.position for snapshot in new_position_snapshots]
                self._perform_action(ReversedAction(_DeletePositionsAction(new_position_snapshots)))

    def _get_time_point_number_of_selection(self):
        time_point_number_of_selection = self._selected[0].time_point_number()
        for position in self._selected:
            if position.time_point_number() != time_point_number_of_selection:
                time_point_number_of_selection = None
                break
        return time_point_number_of_selection

    def _set_track_to_type(self, position_type: Optional[Marker]):
        """Sets all cells in the selected track to the given type."""
        if len(self._selected) == 0:
            self.update_status("You need to select a position first.")
            return
        if len(self._selected) > 1:
            self.update_status("You have multiple positions selected - please unselect one.")
            return

        positions = track_positions_finder.find_all_positions_in_track_of(self._experiment.links, self._selected[0])
        old_position_types = position_markers.get_position_types(self._experiment.position_data, set(positions))
        self._perform_action(_SetAllAsType(old_position_types, position_type))

    def _set_position_to_type(self, position_type: Optional[Marker]):
        """Sets the selected cell to the given type."""
        if len(self._selected) == 0:
            self.update_status("You need to select a position first.")
            return

        positions = set(self._selected)
        old_position_types = position_markers.get_position_types(self._experiment.position_data, positions)
        self._perform_action(_SetAllAsType(old_position_types, position_type))

    def _set_color_of_lineage(self):
        if len(self._selected) == 0:
            self.update_status("You need to select a position first.")
            return
        if len(self._selected) > 1:
            self.update_status("You have multiple positions selected - please unselect one.")
            return

        links = self._experiment.links
        track = links.get_track(self._selected[0])
        if track is None:
            self.update_status("Selected position has no links, so it has no lineage and therefore we cannot color it.")
            return

        old_color = lineage_markers.get_color(self._experiment.links, track)
        color = dialog.prompt_color("Choose a color for the lineage", old_color)
        if color is not None:
            self._perform_action(_SetLineageColor(track, old_color, color))

    def _try_move_selected(self, *, dx: float = 0, dy: float = 0, dz: float = 0):
        if len(self._selected) == 0:
            self.update_status("No positions selected. Select at least one to move it.")
            return

        existing_positions = list(self._selected)
        self._selected = [position.with_offset(dx=dx, dy=dy, dz=dz) for position in existing_positions]
        self._perform_action(_MoveMultiplePositionsAction(existing_positions, dx=dx, dy=dy, dz=dz))
