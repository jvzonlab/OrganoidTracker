from typing import Optional, List, Set

from matplotlib.backend_bases import KeyEvent, MouseEvent, LocationEvent

from autotrack import core
from autotrack.core.experiment import Experiment
from autotrack.core.particles import Particle
from autotrack.core.shape import ParticleShape
from autotrack.gui.window import Window
from autotrack.linking_analysis import cell_error_finder, linking_markers
from autotrack.linking_analysis.linking_markers import EndMarker
from autotrack.visualizer import DisplaySettings, activate
from autotrack.visualizer.exitable_image_visualizer import ExitableImageVisualizer
from autotrack.gui.undo_redo import UndoableAction


class _InsertLinkAction(UndoableAction):
    """Used to insert a link between two particles."""
    particle1: Particle
    particle2: Particle

    def __init__(self, particle1: Particle, particle2: Particle):
        self.particle1 = particle1
        self.particle2 = particle2
        if particle1.time_point_number() == particle2.time_point_number():
            raise ValueError(f"The {particle1} is at the same time point as {particle2}")

    def do(self, experiment: Experiment):
        experiment.links.add_link(self.particle1, self.particle2)
        cell_error_finder.apply_on(experiment, self.particle1, self.particle2)
        return f"Inserted link between {self.particle1} and {self.particle2}"

    def undo(self, experiment: Experiment):
        experiment.links.remove_link(self.particle1, self.particle2)
        cell_error_finder.apply_on(experiment, self.particle1, self.particle2)
        return f"Removed link between {self.particle1} and {self.particle2}"


class _ReverseAction(UndoableAction):
    """Does exactly the opposite of another action. It works by switching the do and undo methods."""
    inverse: UndoableAction

    def __init__(self, action: UndoableAction):
        """Note: there must be a link between the two particles."""
        self.inverse = action

    def do(self, experiment: Experiment):
        return self.inverse.undo(experiment)

    def undo(self, experiment: Experiment):
        return self.inverse.do(experiment)


class _InsertParticleAction(UndoableAction):
    """Used to insert a particle."""

    particle: Particle
    linked_particles: List[Particle]

    def __init__(self, particle: Particle, linked_particles: List[Particle]):
        self.particle = particle
        self.linked_particles = linked_particles

    def do(self, experiment: Experiment):
        experiment.particles.add(self.particle)
        for linked_particle in self.linked_particles:
            experiment.links.add_link(self.particle, linked_particle)
        cell_error_finder.apply_on(experiment, self.particle, *self.linked_particles)

        return_value = f"Added {self.particle}"
        if len(self.linked_particles) > 1:
            return_value += " with connections to " + (" and ".join((str(p) for p in self.linked_particles)))
        if len(self.linked_particles) == 1:
            return_value += f" with a connection to {self.linked_particles[0]}"

        return return_value + "."

    def undo(self, experiment: Experiment):
        experiment.remove_particle(self.particle)
        cell_error_finder.apply_on(experiment, *self.linked_particles)
        return f"Removed {self.particle}"


class _MoveParticleAction(UndoableAction):
    """Used to move a particle"""

    old_position: Particle
    old_shape: ParticleShape
    new_position: Particle

    def __init__(self, old_position: Particle, old_shape: ParticleShape, new_position: Particle):
        if old_position.time_point_number() != new_position.time_point_number():
            raise ValueError(f"{old_position} and {new_position} are in different time points")
        self.old_position = old_position
        self.old_shape = old_shape
        self.new_position = new_position

    def do(self, experiment: Experiment):
        experiment.move_particle(self.old_position, self.new_position)
        cell_error_finder.apply_on(experiment, self.new_position)
        return f"Moved {self.old_position} to {self.new_position}"

    def undo(self, experiment: Experiment):
        experiment.move_particle(self.new_position, self.old_position)
        experiment.particles.add(self.old_position, self.old_shape)
        cell_error_finder.apply_on(experiment, self.old_position)
        return f"Moved {self.new_position} back to {self.old_position}"


class _MarkLineageEndAction(UndoableAction):
    """Used to add a marker to the end of a lineage."""

    marker: Optional[EndMarker]  # Set to None to erase a marker
    old_marker: Optional[EndMarker]
    particle: Particle

    def __init__(self, particle: Particle, marker: Optional[EndMarker], old_marker: Optional[EndMarker]):
        self.particle = particle
        self.marker = marker
        self.old_marker = old_marker

    def do(self, experiment: Experiment) -> str:
        linking_markers.set_track_end_marker(experiment.links, self.particle, self.marker)
        cell_error_finder.apply_on(experiment, self.particle)
        if self.marker is None:
            return f"Removed the lineage end marker of {self.particle}"
        return f"Added the {self.marker.get_display_name()}-marker to {self.particle}"

    def undo(self, experiment: Experiment):
        linking_markers.set_track_end_marker(experiment.links, self.particle, self.old_marker)
        cell_error_finder.apply_on(experiment, self.particle)
        if self.old_marker is None:
            return f"Removed the lineage end marker again of {self.particle}"
        return f"Re-added the {self.old_marker.get_display_name()}-marker to {self.particle}"


class LinkAndPositionEditor(ExitableImageVisualizer):
    """Editor for cell links and positions. Use the Insert key to insert new cells or links, and Delete to delete
     them."""

    _selected1: Optional[Particle] = None
    _selected2: Optional[Particle] = None

    def __init__(self, window: Window, *, time_point_number: int = 1, z: int = 14,
                 selected_particle: Optional[Particle] = None):
        super().__init__(window, time_point_number=time_point_number, z=z,
                         display_settings=DisplaySettings(show_reconstruction=False))

        self._selected1 = selected_particle

    def _get_figure_title(self) -> str:
        return "Editing time point " + str(self._time_point.time_point_number()) + "    (z=" + str(self._z) + ")"

    def _get_window_title(self) -> str:
        return "Manual data editing"

    def _draw_extra(self):
        if self._selected1 is not None and not self._experiment.particles.exists(self._selected1):
            self._selected1 = None
        if self._selected2 is not None and not self._experiment.particles.exists(self._selected2):
            self._selected2 = None

        self._draw_highlight(self._selected1)
        self._draw_highlight(self._selected2)

    def _draw_highlight(self, particle: Optional[Particle]):
        if particle is None:
            return
        color = core.COLOR_CELL_CURRENT
        if particle.time_point_number() < self._time_point.time_point_number():
            color = core.COLOR_CELL_PREVIOUS
        elif particle.time_point_number() > self._time_point.time_point_number():
            color = core.COLOR_CELL_NEXT
        self._ax.plot(particle.x, particle.y, 'o', markersize=25, color=(0,0,0,0), markeredgecolor=color,
                      markeredgewidth=5)

    def _on_mouse_click(self, event: MouseEvent):
        if not event.dblclick:
            return
        new_selection = self._get_particle_at(event.xdata, event.ydata)
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
            "Edit/Editor-Undo (Ctrl+z)": self._undo,
            "Edit/Editor-Redo (Ctrl+y)": self._redo,
            "View/Linking-Linking errors and warnings (E)": self._show_linking_errors,
            "View/Linking-Lineage errors and warnings (L)": self._show_lineage_errors,
            "Edit/LineageEnd-Mark as cell death": lambda: self._try_set_end_marker(EndMarker.DEAD),
            "Edit/LineageEnd-Mark as moving out of view": lambda: self._try_set_end_marker(EndMarker.OUT_OF_VIEW),
            "Edit/LineageEnd-Remove end marker": lambda: self._try_set_end_marker(None)
        }

    def _on_key_press(self, event: KeyEvent):
        if event.key == "c":
            self._exit_view()
        elif event.key == "e":
            particle = self._get_particle_at(event.xdata, event.ydata)
            self._show_linking_errors(particle)
        elif event.key == "l":
            self._show_lineage_errors()
        elif event.key == "ctrl+z":
            self._undo()
        elif event.key == "ctrl+y":
            self._redo()
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
                old_shape = self._experiment.particles.get_shape(self._selected1)
                new_position = Particle(event.xdata, event.ydata, self._z).with_time_point(self._time_point)
                old_position = self._selected1
                self._selected1 = None
                self._perform_action(_MoveParticleAction(old_position, old_shape, new_position))
        else:
            super()._on_key_press(event)

    def _try_delete(self):
        if self._selected1 is None:
            self.update_status("You need to select a cell first")
        elif self._selected2 is None:  # Delete cell and its links
            old_links = self._experiment.links.find_links_of(self._selected1)
            self._perform_action(_ReverseAction(_InsertParticleAction(self._selected1, list(old_links))))
        elif self._experiment.links.contains_link(self._selected1, self._selected2):  # Delete link between cells
            particle1, particle2 = self._selected1, self._selected2
            self._selected1, self._selected2 = None, None
            self._perform_action(_ReverseAction(_InsertLinkAction(particle1, particle2)))
        else:
            self.update_status("No link found between the two particles - nothing to delete")

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

    def _show_linking_errors(self, particle: Optional[Particle] = None):
        from autotrack.visualizer.errors_visualizer import ErrorsVisualizer
        warnings_visualizer = ErrorsVisualizer(self._window, particle)
        activate(warnings_visualizer)

    def _show_lineage_errors(self):
        from autotrack.visualizer.lineage_errors_visualizer import LineageErrorsVisualizer
        editor = LineageErrorsVisualizer(self._window, time_point_number=self._time_point.time_point_number(),
                                         z=self._z)
        activate(editor)

    def _try_insert(self, event: LocationEvent):
        if self._selected1 is None or self._selected2 is None:
            # Insert new particle
            particle = Particle(event.xdata, event.ydata, self._z).with_time_point(self._time_point)
            connections = []
            if self._selected1 is not None \
                    and self._selected1.time_point_number() != self._time_point.time_point_number():
                connections.append(self._selected1)  # Add link to already selected particle

            self._selected1 = particle
            self._perform_action(_InsertParticleAction(particle, connections))
        elif self._selected1.time_point_number() == self._selected2.time_point_number():
            self.update_status("The two selected cells are in exactly the same time point - cannot insert link.")
        elif abs(self._selected1.time_point_number() - self._selected2.time_point_number()) > 1:
            self.update_status("The two selected cells are not in consecutive time points - cannot insert link.")
        else:
            # Insert link between two particles
            particle1, particle2 = self._selected1, self._selected2
            self._selected1, self._selected2 = None, None
            self._perform_action(_InsertLinkAction(particle1, particle2))

    def _perform_action(self, action: UndoableAction):
        status = self._window.get_undo_redo().do(action, self._experiment)
        self.get_window().redraw_data()
        self.update_status(status)

    def _undo(self):
        status = self._window.get_undo_redo().undo(self._experiment)
        self.get_window().redraw_data()
        self.update_status(status)

    def _redo(self):
        status = self._window.get_undo_redo().redo(self._experiment)
        self.get_window().redraw_data()
        self.update_status(status)
