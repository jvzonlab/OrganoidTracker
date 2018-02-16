from typing import Optional

from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.figure import Figure

from imaging import Experiment, Particle, io
from imaging.image_visualizer import AbstractImageVisualizer

from networkx import Graph, NetworkXError

from imaging.visualizer import activate


class LinkEditor(AbstractImageVisualizer):
    """Editor for links.
    Select cells by double-clicking. You can only select cells from the current time step.
    Press Insert/Delete to insert or delete a link between two cells
    Type /help to see how to save or revert changes."""

    _selected1: Optional[Particle]
    _selected2: Optional[Particle]
    _has_uncommitted_changes: bool

    def __init__(self, experiment: Experiment, figure: Figure, frame_number: int = 1, z: int = 14):
        super().__init__(experiment, figure, frame_number=frame_number, z=z)
        self._selected1 = None
        self._selected2 = None
        self._has_uncommitted_changes = False

        # Check if graph exists
        graph = experiment.particle_links_scratch()
        if graph is None:
            baseline_graph = experiment.particle_links()
            if baseline_graph is None:
                experiment.particle_links_scratch(Graph())
            else:
                experiment.particle_links_scratch(baseline_graph.copy())

    def get_title(self, errors: int) -> str:
        return "Editing frame " + str(self._frame.frame_number()) + "    (z=" + str(self._z) + ")"

    def draw_extra(self):
        self._draw_highlight(self._selected1)
        self._draw_highlight(self._selected2)

    def _draw_highlight(self, particle: Optional[Particle]):
        if particle is None:
            return
        color = "red"
        if particle.frame_number() < self._frame.frame_number():
            color = "darkred"
        elif particle.frame_number() > self._frame.frame_number():
            color = "orange"
        self._ax.plot(particle.x, particle.y, 'o', markersize=25, color=(0,0,0,0), markeredgecolor=color,
                      markeredgewidth=5)

    def _on_mouse_click(self, event: MouseEvent):
        if not event.dblclick:
            return
        new_selection = self.get_closest_particle(self._frame.particles(), event.xdata, event.ydata, self._z, max_distance=20)
        if new_selection is None:
            self.update_status("Cannot find a particle here")
            return
        self._selected1 = self._selected2
        self._selected2 = new_selection
        self.draw_view()
        self.update_status("Selected:\n        " + str(self._selected2) + "\n        " + str(self._selected1))

    def _on_key_press(self, event: KeyEvent):
        if event.key == "insert":
            if self._check_selection():
                self._experiment.particle_links_scratch().add_edge(self._selected1, self._selected2)
                self._after_modification()
                self.update_status("Added link")
        elif event.key == "delete":
            if self._check_selection():
                try:
                    self._experiment.particle_links_scratch().remove_edge(self._selected1, self._selected2)
                    self._after_modification()
                    self.update_status("Removed link")
                except NetworkXError:
                    self.update_status("Cannot delete link: there was no link between selected particles")
        elif event.key == "l":
            from imaging.image_visualizer import StandardImageVisualizer
            image_visualizer = StandardImageVisualizer(self._experiment, self._fig,
                                                       frame_number=self._frame.frame_number(), z=self._z)
            activate(image_visualizer)
        else:
            super()._on_key_press(event)

    def _check_selection(self) -> bool:
        if self._selected1 is None or self._selected2 is None:
            self.update_status("Need to select two particles first")
            return False
        delta_frame_number = abs(self._selected1.frame_number() - self._selected2.frame_number())
        if delta_frame_number == 0:
            self.update_status("Cannot link two cells from the same time point together")
            return False
        if delta_frame_number > 5:
            self.update_status("Cannot link two cells together that are"  + str(delta_frame_number) + " time points apart")
            return False
        return True

    def _on_command(self, command: str):
        if command == "commit":
            our_links = self._experiment.particle_links_scratch()
            self._experiment.particle_links(our_links.copy())
            self._has_uncommitted_changes = False
            self.draw_view()
            self.update_status("Committed all changes.")
            return True
        if command == "revert":
            old_links = self._experiment.particle_links()
            self._experiment.particle_links_scratch(old_links.copy())
            self._has_uncommitted_changes = False
            self.draw_view()
            self.update_status("Reverted all uncommitted changes.")
            return True
        if command.startswith("export "):
            if self._has_uncommitted_changes:
                self.update_status("There are uncommitted changes. /commit or /revert them first.")
                return True
            filename = command[7:].strip()
            if len(filename) == 0:
                self.update_status("Please give a file name to save to")
                return True
            if not filename.endswith(".json"):
                filename+= ".json"
            io.save_links_to_json(self._experiment.particle_links(), filename)
            self.update_status("Saved committed links to " + filename)
            return True
        if command == "help":
            self.update_status("/commit - commits all changes, so that they cannot be reverted\n"
                               "/revert - reverst all uncommitted changes\n"
                               "/export - exports all committed changes to a file")
            return True
        return super()._on_command(command)

    def _after_modification(self):
        self._selected1 = None
        self._selected2 = None
        self._has_uncommitted_changes = True
        self.draw_view()