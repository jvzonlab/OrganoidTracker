from typing import Optional

from matplotlib.backend_bases import KeyEvent, MouseEvent
from networkx import Graph, NetworkXError

from autotrack import core
from autotrack.core.particles import Particle
from autotrack.gui import Window
from autotrack.imaging import io
from autotrack.visualizer.image_visualizer import AbstractImageVisualizer
from autotrack.visualizer import activate


class LinkEditor(AbstractImageVisualizer):
    """Editor for links.
    Select cells by double-clicking. You can only select cells from the current time step.
    Press Insert/Delete to insert or delete a link between two cells
    Type /help to see how to save or revert changes."""

    _selected1: Optional[Particle]
    _selected2: Optional[Particle]
    _has_uncommitted_changes: bool

    def __init__(self, window: Window, time_point_number: int = 1, z: int = 14):
        super().__init__( window, time_point_number=time_point_number, z=z)
        self._selected1 = None
        self._selected2 = None
        self._has_uncommitted_changes = False

        # Check if graph exists
        experiment = window.get_experiment()
        graph = experiment.particle_links_scratch()
        if graph is None:
            baseline_graph = experiment.particle_links()
            if baseline_graph is None:
                experiment.particle_links_scratch(Graph())
            else:
                experiment.particle_links_scratch(baseline_graph.copy())

    def _get_figure_title(self, errors: int) -> str:
        return "Editing links of time point " + str(self._time_point.time_point_number()) + "    (z=" + str(self._z) + ")"

    def _get_window_title(self) -> str:
        return "Link editing"

    def get_extra_menu_options(self):
        return {
            **super().get_extra_menu_options(),
            "Edit/Commit-Commit dotted lines (/commit)": self._commit,
            "Edit/Commit-Revert dotted lines (/revert)": self._revert,
            "Edit/Manual-Positions... (P)": self._show_position_editor,
            "View/Exit-Exit this view (L)": self._exit
        }

    def _draw_extra(self):
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
            self._exit()
        elif event.key == "p":
            self._show_position_editor()
        else:
            super()._on_key_press(event)

    def _exit(self):
        from autotrack.visualizer.image_visualizer import StandardImageVisualizer
        image_visualizer = StandardImageVisualizer(self._window,
                                                   time_point_number=self._time_point.time_point_number(), z=self._z)
        activate(image_visualizer)

    def _commit(self):
        our_links = self._experiment.particle_links_scratch()
        self._experiment.particle_links(our_links.copy())
        self._has_uncommitted_changes = False
        self.draw_view()
        self.update_status("Committed all changes.")

    def _revert(self):
        old_links = self._experiment.particle_links()
        self._experiment.particle_links_scratch(old_links.copy())
        self._has_uncommitted_changes = False
        self.draw_view()
        self.update_status("Reverted all uncommitted changes.")

    def _check_selection(self) -> bool:
        if self._selected1 is None or self._selected2 is None:
            self.update_status("Need to select two particles first")
            return False
        delta_time_point_number = abs(self._selected1.time_point_number() - self._selected2.time_point_number())
        if delta_time_point_number == 0:
            self.update_status("Cannot link two cells from the same time point together")
            return False
        if delta_time_point_number > 5:
            self.update_status("Cannot link two cells together that are" + str(delta_time_point_number) + " time points apart")
            return False
        return True

    def _on_command(self, command: str):
        if command == "commit":
            self._commit()
            return True
        if command == "revert":
            self._revert()
            return True
        if command == "exit":
            self._exit()
            return True
        if command.startswith("export "):
            self._export_links(filename=command[7:].strip())
            return True
        if command == "help":
            self.update_status("/commit - commits all changes, so that they cannot be reverted\n"
                               "/revert - reverst all uncommitted changes\n"
                               "/export - exports all committed changes to a file")
            return True
        return super()._on_command(command)

    def _export_links(self, filename: str):
        if self._has_uncommitted_changes:
            self.update_status("There are uncommitted changes. /commit or /revert them first.")
            return
        if len(filename) == 0:
            self.update_status("Please give a file name to save to")
            return
        if not filename.endswith(".json"):
            filename += ".json"
        io.save_links_to_json(self._experiment.particle_links(), filename)
        self.update_status("Saved committed links to " + filename)

    def _after_modification(self):
        graph = self._experiment.particle_links_scratch()
        graph.add_node(self._selected1, edited=True)  # Mark as edited, so that warnings displayer knows this
        graph.add_node(self._selected2, edited=True)
        self._selected1 = None
        self._selected2 = None
        self._has_uncommitted_changes = True
        self.draw_view()

    def _show_position_editor(self):
        from autotrack.visualizer.position_editor import PositionEditor
        position_editor = PositionEditor(self._window, time_point_number=self._time_point.time_point_number(),
                                         z=self._z)
        activate(position_editor)