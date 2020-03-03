from typing import Optional, List, Set, Dict, Iterable

from matplotlib.backend_bases import KeyEvent, MouseEvent, LocationEvent

from organoid_tracker import core
from organoid_tracker.core import Color, UserError
from organoid_tracker.core.connections import Connections
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.particle import Particle
from organoid_tracker.core.position import Position
from organoid_tracker.core.marker import Marker
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.linking_analysis import cell_error_finder, linking_markers, lineage_positions_finder, lineage_markers
from organoid_tracker.linking_analysis.linking_markers import EndMarker
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.abstract_editor import AbstractEditor
from organoid_tracker.gui.undo_redo import UndoableAction, ReversedAction


class _InsertLinkAction(UndoableAction):
    """Used to insert a link between two positions."""
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

        cell_error_finder.find_errors_in_positions_links_and_all_dividing_cells(experiment, self.all_positions[0], self.all_positions[-1])
        return f"Removed link between {self.all_positions[0]} and {self.all_positions[-1]}"


class _InsertPositionAction(UndoableAction):
    """Used to insert a position."""

    particle: Particle

    def __init__(self, particle: Particle):
        self.particle = particle

    def do(self, experiment: Experiment) -> str:
        self.particle.restore(experiment)
        cell_error_finder.find_errors_in_positions_links_and_all_dividing_cells(experiment, self.particle.position, *self.particle.links)

        return_value = f"Added {self.particle.position}"
        if len(self.particle.links) > 1:
            return_value += " with connections to " + (" and ".join((str(p) for p in self.particle.links)))
        if len(self.particle.links) == 1:
            return_value += f" with a connection to {self.particle.links[0]}"

        return return_value + "."

    def undo(self, experiment: Experiment) -> str:
        experiment.remove_position(self.particle.position)
        cell_error_finder.find_errors_in_positions_links_and_all_dividing_cells(experiment, *self.particle.links)
        return f"Removed {self.particle.position}"


class _DeletePositionsAction(UndoableAction):

    _particles: List[Particle]

    def __init__(self, particles: Iterable[Particle]):
        self._particles = list(particles)

    def do(self, experiment: Experiment):
        experiment.remove_positions((particle.position for particle in self._particles))
        for particle in self._particles:  # Check linked particles for errors
            cell_error_finder.find_errors_in_just_these_positions(experiment, *particle.links)
        cell_error_finder.find_errors_in_all_dividing_cells(experiment)
        return f"Removed {len(self._particles)} positions"

    def undo(self, experiment: Experiment):
        for particle in self._particles:
            particle.restore(experiment)
            cell_error_finder.find_errors_in_just_these_positions(experiment, particle.position, *particle.links)
        cell_error_finder.find_errors_in_all_dividing_cells(experiment)
        return f"Re-added {len(self._particles)} positions"


class _MovePositionAction(UndoableAction):
    """Used to move a position"""

    old_position: Position
    new_position: Position

    def __init__(self, old_position: Position, new_position: Position):
        if old_position.time_point_number() != new_position.time_point_number():
            raise ValueError(f"{old_position} and {new_position} are in different time points")
        self.old_position = old_position
        self.new_position = new_position

    def do(self, experiment: Experiment):
        experiment.move_position(self.old_position, self.new_position)
        cell_error_finder.find_errors_in_positions_links_and_all_dividing_cells(experiment, self.new_position)
        return f"Moved {self.old_position} to {self.new_position}"

    def undo(self, experiment: Experiment):
        experiment.move_position(self.new_position, self.old_position)
        cell_error_finder.find_errors_in_positions_links_and_all_dividing_cells(experiment, self.old_position)
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


class _ReplaceConnectionsAction(UndoableAction):

    _old_connections: Connections
    _new_connections: Connections

    def __init__(self, old_connections: Connections, new_connections: Connections):
        self._old_connections = old_connections
        self._new_connections = new_connections

    def do(self, experiment: Experiment) -> str:
        experiment.connections = self._new_connections
        return f"Created {len(self._new_connections)} new connections"

    def undo(self, experiment: Experiment) -> str:
        experiment.connections = self._old_connections
        return "Restored the previous connections"


class _SetAllAsType(UndoableAction):
    _previous_position_types: Dict[Position, str]
    _type: Marker

    def __init__(self, previous_position_types: Dict[Position, str], new_type: Marker):
        self._previous_position_types = previous_position_types
        self._type = new_type

    def do(self, experiment: Experiment) -> str:
        position_data = experiment.position_data
        for position in self._previous_position_types.keys():
            linking_markers.set_position_type(position_data, position, self._type.save_name)
        return f"All positions in the lineage tree are now of the type \"{self._type.display_name}\""

    def undo(self, experiment: Experiment) -> str:
        position_data = experiment.position_data
        for position in self._previous_position_types.keys():
            linking_markers.set_position_type(position_data, position, self._previous_position_types.get(position))
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

    _new_particle: Particle
    _old_particle: Particle

    def __init__(self, new_particle: Particle, old_particle: Particle):
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


class _MarkPositionAsUncertainAction(UndoableAction):

    _position: Position

    def __init__(self, position: Position):
        self._position = position

    def do(self, experiment: Experiment) -> str:
        linking_markers.set_uncertain(experiment.position_data, self._position, True)
        cell_error_finder.find_errors_in_just_these_positions(experiment, self._position)
        return f"Marked {self._position} as uncertain"

    def undo(self, experiment: Experiment) -> str:
        linking_markers.set_uncertain(experiment.position_data, self._position, False)
        cell_error_finder.find_errors_in_just_these_positions(experiment, self._position)
        return f"Marked that {self._position} is no longer uncertain"


class LinkAndPositionEditor(AbstractEditor):
    """Editor for cell links and positions. Use the Insert key to insert new cells or links, and Delete to delete
     them."""

    _selected1: Optional[Position] = None
    _selected2: Optional[Position] = None

    def __init__(self, window: Window, *, selected_position: Optional[Position] = None):
        super().__init__(window)

        self._selected1 = selected_position

    def _draw_extra(self):
        if self._selected1 is not None and not self._experiment.positions.contains_position(self._selected1):
            self._selected1 = None
        if self._selected2 is not None and not self._experiment.positions.contains_position(self._selected2):
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
        self._draw_selection(position, color)

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
        options = {
            **super().get_extra_menu_options(),
            "Edit//Experiment-Edit splines... [A]": self._show_spline_editor,
            "Edit//Experiment-Edit beacons... [B]": self._show_beacon_editor,
            "Edit//Experiment-Edit image offsets... [O]": self._show_offset_editor,
            "Edit//Batch-Batch deletion//Delete data of time point...": self._delete_data_of_time_point,
            "Edit//Batch-Batch deletion//Delete all tracks with errors...": self._delete_tracks_with_errors,
            "Edit//Batch-Batch deletion//Delete all tracks not in the first time point...": self._delete_tracks_not_in_first_time_point,
            "Edit//Batch-Batch deletion//Delete all positions in a rectangle...": self._show_positions_in_rectangle_deleter,
            "Edit//Batch-Batch deletion//Delete all positions without links...": self._delete_positions_without_links,
            "Edit//Batch-Batch connection//Connect positions by distance...": self._connect_positions_by_distance,
            "Edit//LineageEnd-Mark as cell death [D]": lambda: self._try_set_end_marker(EndMarker.DEAD),
            "Edit//LineageEnd-Mark as cell shedding [S]": lambda: self._try_set_end_marker(EndMarker.SHED),
            "Edit//LineageEnd-Mark as moving out of view [V]": lambda: self._try_set_end_marker(EndMarker.OUT_OF_VIEW),
            "Edit//LineageEnd-Remove end marker": lambda: self._try_set_end_marker(None),
            "Edit//Uncertain-Mark position as uncertain": lambda: self._try_mark_uncertainty(True),
            "Edit//Uncertain-Remove uncertainty marker": lambda: self._try_mark_uncertainty(False),
            "Edit//Track-Set color of lineage...": self._set_color_of_lineage,
            "Edit//Track-Delete entire lineage": self._delete_selected_lineage,
            "View//Linking-Linking errors and warnings (E)": self._show_linking_errors,
            "View//Linking-Lineage errors and warnings [L]": self._show_lineage_errors,
        }

        # Add options for changing position types
        for position_type in self.get_window().get_gui_experiment().get_registered_markers(Position):
            # Create copy of position_type variable to avoid it changing in loop iteration
            action = lambda bound_position_type = position_type: self._set_all_positions_to_type(bound_position_type)

            options["Edit//Track-Set type of track//" + position_type.display_name] = action
        return options

    def _on_key_press(self, event: KeyEvent):
        if event.key == "c":
            self._exit_view()
        elif event.key == "e":
            position = self._get_position_at(event.xdata, event.ydata)
            self._show_linking_errors(position)
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
                new_position = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
                old_position = self._selected1
                self._selected1 = None
                self._perform_action(_MovePositionAction(old_position, new_position))
        else:
            super()._on_key_press(event)

    def _try_delete(self):
        if self._selected1 is None:
            self.update_status("You need to select a cell first")
        elif self._selected2 is None:  # Delete cell and its links
            particle_to_delete = Particle.from_position(self._experiment, self._selected1)
            self._perform_action(ReversedAction(_InsertPositionAction(particle_to_delete)))
        elif self._experiment.connections.contains_connection(self._selected1, self._selected2):  # Delete a connection
            position1, position2 = self._selected1, self._selected2
            self._selected1, self._selected2 = None, None
            self._perform_action(ReversedAction(_InsertConnectionAction(position1, position2)))
        elif self._experiment.links.contains_link(self._selected1, self._selected2):  # Delete link between cells
            position1, position2 = self._selected1, self._selected2
            self._selected1, self._selected2 = None, None
            self._perform_action(ReversedAction(_InsertLinkAction(position1, position2)))
        else:
            self.update_status("No link found between the two positions - nothing to delete")

    def _try_set_end_marker(self, marker: Optional[EndMarker]):
        if self._selected1 is None or self._selected2 is not None:
            self.update_status("You need to have exactly one cell selected in order to set an end marker.")
            return

        links = self._experiment.links
        if len(links.find_futures(self._selected1)) > 0:
            self.update_status(f"The {self._selected1} is not a lineage end.")
            return
        current_marker = linking_markers.get_track_end_marker(self._experiment.position_data, self._selected1)
        if current_marker == marker:
            if marker is None:
                self.update_status("There is no lineage ending marker here, cannot delete anything.")
            else:
                self.update_status(f"This lineage end already has the {marker.get_display_name()} marker.")
            return
        self._perform_action(_MarkLineageEndAction(self._selected1, marker, current_marker))

    def _try_mark_uncertainty(self, uncertain: bool):
        if self._selected1 is None or self._selected2 is not None:
            self.update_status("You need to have exactly one cell selected.")
            return
        position_data = self._experiment.position_data
        if linking_markers.is_uncertain(position_data, self._selected1) == uncertain:
            if uncertain:
                self.update_status("Selected position is already marked as uncertain.")
            else:
                self.update_status("Selected position is not marked as uncertain, cannot remove marker.")
            return
        if uncertain:
            self._perform_action(_MarkPositionAsUncertainAction(self._selected1))
        else:
            self._perform_action(ReversedAction(_MarkPositionAsUncertainAction(self._selected1)))

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

    def _show_positions_in_rectangle_deleter(self):
        from organoid_tracker.visualizer.position_in_rectangle_deleter import PositionsInRectangleDeleter
        editor = PositionsInRectangleDeleter(self._window)
        activate(editor)

    def _delete_data_of_time_point(self):
        """Deletes all annotations of a given time point."""
        positions = self._experiment.positions.of_time_point(self._time_point)
        particles = (Particle.from_position(self._experiment, position) for position in positions)
        self._perform_action(_DeletePositionsAction(particles))

    def _delete_positions_without_links(self):
        """Deletes all positions that have no links."""
        particles_without_links = []
        links = self._experiment.links
        for position in self._experiment.positions:
            if not links.contains_position(position):
                particles_without_links.append(Particle.from_position(self._experiment, position))
        self._perform_action(_DeletePositionsAction(particles_without_links))

    def _delete_tracks_with_errors(self):
        """Deletes all lineages where at least a single error was present."""
        if not dialog.prompt_yes_no("Warning", "Are you sure you want to delete all tracks with at least a single error"
                                               " in them? This cannot be undone."):
            return
        self.get_window().get_gui_experiment().undo_redo.clear()

        from organoid_tracker.linking_analysis import lineage_error_finder
        lineage_error_finder.delete_problematic_lineages(self._experiment)

        self.get_window().redraw_data()

    def _delete_selected_lineage(self):
        if self._selected1 is None:
            raise UserError("No cell selected", "You need to select a cell first to delete all cells in that"
                                                    " lineage.")
        if self._selected2 is not None:
            raise UserError("Too many cell selected", "You have selected multiple cells - please select only one.")

        experiment = self._experiment
        particles_in_track = []

        # Find all positions in the track (and descending tracks)
        track = experiment.links.get_track(self._selected1)
        if track is None:
            particles_in_track.append(Particle.from_position(experiment, self._selected1))
        else:
            # Find starter of lineage tree
            previous_tracks = track.get_previous_tracks()
            while len(previous_tracks) == 1:
                track = previous_tracks.pop()
                previous_tracks = track.get_previous_tracks()

            # Find all positions in lineage tree
            for some_track in track.find_all_descending_tracks(include_self=True):
                for position in some_track.positions():
                    particles_in_track.append(Particle.from_position(experiment, position))

        self._perform_action(_DeletePositionsAction(particles_in_track))

    def _delete_tracks_not_in_first_time_point(self):
        """Deletes all lineages where at least a single error was present."""
        if not dialog.prompt_yes_no("Warning", "Are you sure you want to delete all lineages that do not reach the"
                                               " first time point? This cannot be undone."):
            return
        self.get_window().get_gui_experiment().undo_redo.clear()

        from organoid_tracker.linking_analysis import links_filter
        links_filter.delete_lineages_not_in_first_time_point(self._experiment)

        self.get_window().redraw_data()


    def _connect_positions_by_distance(self):
        distance_um = dialog.prompt_float("Maximum distance", "Up to what distance (μm) should all positions be"
                                                              " connected?", minimum=0)
        if distance_um is None:
            return

        from organoid_tracker.connecting.connector_by_distance import ConnectorByDistance
        connector = ConnectorByDistance(distance_um)
        connections = connector.create_connections(self._experiment)
        self._perform_action(_ReplaceConnectionsAction(self._experiment.connections, connections))

    def _try_insert(self, event: LocationEvent):
        if self._selected1 is None:
            # Add new position without links
            self._selected1 = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
            mouse_position = self._get_position_at(event.xdata, event.ydata)
            self._perform_action(_InsertPositionAction(Particle.just_position(self._selected1)))
        elif self._selected2 is None:
            # Insert new position with link to self._selected1
            # Find at which position the mouse is pointing
            mouse_position = self._get_position_at(event.xdata, event.ydata)
            is_new_position = False
            if mouse_position is None or abs(mouse_position.time_point_number() -
                                             self._selected1.time_point_number()) != 1:
                # Just create a new position, position on mouse is not suitable
                mouse_position = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
                is_new_position = True
                if self._display_settings.show_next_time_point and self._selected1.time_point() == self._time_point:
                    # Insert in next time point when showing two time points, and inserting in this time point wouldn't
                    # be possible
                    mouse_position = mouse_position.with_time_point_number(mouse_position.time_point_number() + 1)

            if abs(mouse_position.time_point_number() - self._selected1.time_point_number()) == 1:
                connection = self._selected1

                # Insert link from selected point to new point
                if is_new_position:
                    self._selected1 = mouse_position
                    new_particle = Particle.position_with_links(mouse_position, links=[connection])
                    self._perform_action(_InsertPositionAction(new_particle))
                else:
                    # New point already exists, overwrite that point
                    old_particle = Particle.from_position(self._experiment, mouse_position)
                    mouse_position = Position(event.xdata, event.ydata, self._z, time_point=mouse_position.time_point())
                    self._selected1 = mouse_position
                    new_particle = Particle.position_with_links(mouse_position, links=[connection,
                                             *(link for link in old_particle.links
                                               if link.time_point() != connection.time_point())])
                    self._perform_action(_OverwritePositionAction(new_particle, old_particle))
            else:
                inserting_position = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
                if inserting_position.distance_um(self._selected1, ImageResolution.PIXELS) < 10:
                    # Mouse is close to an existing position
                    self.update_status("Cannot insert a position here - too close to already selected position.")
                    return
                self._selected1 = Position(event.xdata, event.ydata, self._z, time_point=self._time_point)
                self._perform_action(_InsertPositionAction(Particle.just_position(self._selected1)))
        elif self._selected1.time_point_number() == self._selected2.time_point_number():
            # Insert connection between two positions
            if self._experiment.connections.contains_connection(self._selected1, self._selected2):
                self.update_status("A connection already exists between the selected positions.")
                return
            position1, position2 = self._selected1, self._selected2
            self._selected1, self._selected2 = None, None
            self._perform_action(_InsertConnectionAction(position1, position2))
        else:
            # Insert link between two positions
            if self._experiment.links.contains_link(self._selected1, self._selected2):
                self.update_status("A link already exists between the selected positions.")
                return
            position1, position2 = self._selected1, self._selected2
            self._selected1, self._selected2 = None, None
            self._perform_action(_InsertLinkAction(position1, position2))

    def _set_all_positions_to_type(self, position_type: Marker):
        """Sets all cells in the experiment to the given type."""
        if self._selected1 is None:
            self.update_status("You need to select a position first.")
            return
        if self._selected2 is not None:
            self.update_status("You have multiple positions selected - please unselect one.")
            return

        positions = lineage_positions_finder.find_all_positions_in_lineage_of(self._experiment.links, self._selected1)
        old_position_types = linking_markers.get_position_types(self._experiment.position_data, set(positions))
        self._perform_action(_SetAllAsType(old_position_types, position_type))

    def _set_color_of_lineage(self):
        if self._selected1 is None:
            self.update_status("You need to select a position first.")
            return
        if self._selected2 is not None:
            self.update_status("You have multiple positions selected - please unselect one.")
            return

        links = self._experiment.links
        track = links.get_track(self._selected1)
        if track is None:
            self.update_status("Selected position has no links, so it has no lineage and therefore we cannot color it.")
            return

        old_color = lineage_markers.get_color(self._experiment.links, track)
        color = dialog.prompt_color("Choose a color for the lineage", old_color)
        if color is not None:
            self._perform_action(_SetLineageColor(track, old_color, color))
