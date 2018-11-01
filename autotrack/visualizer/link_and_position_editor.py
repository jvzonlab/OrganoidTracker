import collections
from typing import Optional, Deque, List

from matplotlib.backend_bases import KeyEvent, MouseEvent, LocationEvent
from networkx import Graph

from autotrack import core
from autotrack.core.experiment import Experiment
from autotrack.core.links import LinkType
from autotrack.core.particles import Particle
from autotrack.gui import Window
from autotrack.visualizer import DisplaySettings, activate
from autotrack.visualizer.image_visualizer import AbstractImageVisualizer


def _initialize_links(experiment: Experiment):
    if experiment.links.scratch is None:
        # Scratch links are missing - set scratch links from baseline
        baseline_links = experiment.links.baseline
        if baseline_links is None:
            baseline_links = Graph()
        experiment.links.set_links(LinkType.SCRATCH, baseline_links)
    experiment.links.remove_links(LinkType.BASELINE)


def _commit_links(experiment: Experiment):
    """Makes all dotted links "real" links."""
    experiment.links.set_links(LinkType.BASELINE, experiment.links.scratch)
    experiment.links.remove_links(LinkType.SCRATCH)


class _Action:

    def do(self, experiment: Experiment) -> str:
        """Performs the action. The scratch links in the experiment will be initialized. Returns an user-friendly
        message of what just happened."""
        raise NotImplementedError()

    def undo(self, experiment: Experiment) -> str:
        """Undoes the action. The scratch links in the experiment will be initialized. Returns an user-friendly
        message of what just happened."""
        raise NotImplementedError()


class _InsertLinkAction(_Action):
    """Used to insert a link between two particles."""
    particle1: Particle
    particle2: Particle

    def __init__(self, particle1: Particle, particle2: Particle):
        self.particle1 = particle1
        self.particle2 = particle2
        if particle1.time_point_number() == particle2.time_point_number():
            raise ValueError(f"The {particle1} is at the same time point as {particle2}")

    def do(self, experiment: Experiment):
        links = experiment.links.scratch
        if self.particle1 not in links:
            links.add_node(self.particle1)
        if self.particle2 not in links:
            links.add_node(self.particle2)

        links.add_edge(self.particle1, self.particle2)
        return f"Inserted link between {self.particle1} and {self.particle2}"

    def undo(self, experiment: Experiment):
        links = experiment.links.scratch
        links.remove_edge(self.particle1, self.particle2)
        return f"Removed link between {self.particle1} and {self.particle2}"


class _ReverseAction(_Action):
    """Does exactly the opposite of another action. It works by switching the do and undo methods."""
    inverse: _Action

    def __init__(self, action: _Action):
        """Note: there must be a link between the two particles."""
        self.inverse = action

    def do(self, experiment: Experiment):
        return self.inverse.undo(experiment)

    def undo(self, experiment: Experiment):
        return self.inverse.do(experiment)


class _InsertParticleAction(_Action):
    """Used to insert a particle."""

    particle: Particle
    linked_particles: List[Particle]

    def __init__(self, particle: Particle, linked_particles: List[Particle]):
        self.particle = particle
        self.linked_particles = linked_particles

    def do(self, experiment: Experiment):
        experiment.add_particle(self.particle)
        for linked_particle in self.linked_particles:
            experiment.links.scratch.add_edge(self.particle, linked_particle)
        return_value = f"Added {self.particle}"
        if len(self.linked_particles) > 1:
            return_value += " with connections to " + (" and ".join((str(p) for p in self.linked_particles)))
        if len(self.linked_particles) == 1:
            return_value += f" with a connection to {self.linked_particles[0]}"
        return return_value + "."

    def undo(self, experiment: Experiment):
        experiment.remove_particle(self.particle)
        return f"Removed {self.particle}"


class LinkAndPositionEditor(AbstractImageVisualizer):
    """Editor for cell links and positions. Use the Insert key to insert new cells or links, and Delete to delete
     them."""

    _selected1: Optional[Particle] = None
    _selected2: Optional[Particle] = None

    _undo_queue: Deque[_Action]
    _redo_queue: Deque[_Action]

    def __init__(self, window: Window, time_point_number: int = 1, z: int = 14):
        super().__init__(window, time_point_number, z, DisplaySettings(show_reconstruction=False))

        self._undo_queue = collections.deque(maxlen=50)
        self._redo_queue = collections.deque(maxlen=50)

        _initialize_links(self._experiment)

    def _get_figure_title(self, errors: int) -> str:
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
            self.update_status("Cannot find a particle here")
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
            "View/Exit-Exit this view (C)": self._exit_view
        }

    def _on_command(self, command: str):
        if command == "exit":
            self._exit_view()
        elif command == "help":
            self.update_status("/exit: Exit this view (you can also press C)"
                               "\n/t20: Jump to time point 20 (also works for other time points)")
        else:
            super()._on_command(command)

    def _on_key_press(self, event: KeyEvent):
        if event.key == "c":
            self._exit_view()
        elif event.key == "e":
            particle = self._get_particle_at(event.xdata, event.ydata)
            self._show_linking_errors(particle)
        elif event.key == "ctrl+z":
            self._undo()
        elif event.key == "ctrl+y":
            self._redo()
        elif event.key == "insert":
            self._try_insert(event)
        elif event.key == "delete":
            if self._selected1 is None:
                self.update_status("You need to select a cell first")
            elif self._selected2 is None:  # Delete cell and its links
                old_links = list(self._experiment.links.scratch[self._selected1])
                self._perform_action(_ReverseAction(_InsertParticleAction(self._selected1, old_links)))
            elif self._experiment.links.scratch.has_edge(self._selected1, self._selected2): # Delete link between cells
                self._perform_action(_ReverseAction(_InsertLinkAction(self._selected1, self._selected2)))
            else:
                self.update_status("No link found between the two particles - nothing to delete")
        else:
            super()._on_key_press(event)

    def _show_linking_errors(self, particle: Optional[Particle] = None):
        from autotrack.visualizer.errors_visualizer import ErrorsVisualizer
        warnings_visualizer = ErrorsVisualizer(self._window, particle)
        activate(warnings_visualizer)

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
        else:
            # Insert link between two particles
            particle1, particle2 = self._selected1, self._selected2
            self._selected1, self._selected2 = None, None
            self._perform_action(_InsertLinkAction(particle1, particle2))

    def _perform_action(self, action: _Action):
        """Performs an action, and stores it so that we can undo it"""
        result_string = action.do(self._experiment)
        self._undo_queue.append(action)
        self._redo_queue.clear()
        self.draw_view()  # Refresh
        self.update_status(result_string)

    def _undo(self):
        try:
            action = self._undo_queue.pop()
            result_string = action.undo(self._experiment)
            self._redo_queue.append(action)
            self.draw_view()  # Refresh
            self.update_status(result_string)
        except IndexError:
            self.update_status("No more actions to undo.")

    def _redo(self):
        try:
            action = self._redo_queue.pop()
            result_string = action.do(self._experiment)
            self._undo_queue.append(action)
            self.draw_view()  # Refresh
            self.update_status(result_string)
        except IndexError:
            self.update_status("No more actions to redo.")

    def _exit_view(self):
        _commit_links(self._experiment)
        from autotrack.visualizer.image_visualizer import StandardImageVisualizer
        image_visualizer = StandardImageVisualizer(self._window, time_point_number=self._time_point.time_point_number(),
                                                   z=self._z, display_settings=self._display_settings)
        activate(image_visualizer)
