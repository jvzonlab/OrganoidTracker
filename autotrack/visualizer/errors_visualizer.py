from typing import List, Optional, Dict, Any

from matplotlib.backend_bases import KeyEvent

from autotrack.core.particles import Particle
from autotrack.gui.window import Window
from autotrack.linking_analysis import linking_markers, lineage_checks, logical_tests
from autotrack.linking_analysis.lineage_checks import LineageWithErrors
from autotrack.visualizer import activate
from autotrack.visualizer.particle_list_visualizer import ParticleListVisualizer


class ErrorsVisualizer(ParticleListVisualizer):
    """Shows all errors and warnings in the experiment.
    Press Left/Right to view the previous/next error.
    Press Delete to delete (suppress) the shown error.
    Press C to change (fix) the data and press E to exit this view.
    """

    _problematic_lineages: List[LineageWithErrors]
    _current_lineage_index: int = -1
    _total_number_of_warnings: int

    def __init__(self, window: Window, start_particle: Optional[Particle]):
        links = window.get_experiment().links
        crumb_particles = set()
        if start_particle is not None:
            crumb_particles.add(start_particle)
        if self._get_last_particle() is not None:
            crumb_particles.add(self._get_last_particle())
        self._problematic_lineages = lineage_checks.get_problematic_lineages(links, crumb_particles)
        self._total_number_of_warnings = sum((len(lineage.errored_particles) for lineage in self._problematic_lineages))

        super().__init__(window, chosen_particle=start_particle, all_particles=[])

    def _show_closest_or_stored_particle(self, particle: Optional[Particle]):
        if particle is None:
            particle = self._get_last_particle()

        lineage_index = lineage_checks.find_lineage_index_with_crumb(self._problematic_lineages, particle)
        if lineage_index is None:
            # Try again, now with last particle
            particle = self._get_last_particle()
            lineage_index = lineage_checks.find_lineage_index_with_crumb(self._problematic_lineages, particle)
            if lineage_index is None:
                return

        self._current_lineage_index = lineage_index  # Found the lineage the cell is in
        self._particle_list = self._problematic_lineages[self._current_lineage_index].errored_particles
        try:
            # We even found the cell itself
            self._current_particle_index = self._particle_list.index(particle)
        except ValueError:
            self._current_particle_index = -1

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "Edit/Errors-Suppress this error (Del)": self._suppress_error,
            "Edit/Errors-Recheck all errors (/recheck)": self._recheck_errors,
            "Navigate/Lineage-Next lineage (Up)": self.__goto_next_lineage,
            "Navigate/Lineage-Previous lineage (Down)": self.__goto_previous_lineage
        }

    def get_message_no_particles(self):
        if len(self._problematic_lineages) > 0:
            return "No warnings or errors found at position.\n" \
                   "Press the up arrow key to view the first lineage tree with warnings."
        return "No warnings or errors found. Hurray?"

    def get_message_press_right(self):
        return "No warnings or errors found in at position." \
               "\nPress the right arrow key to view the first warning in the lineage."

    def get_title(self, particle_list: List[Particle], current_particle_index: int):
        particle = particle_list[current_particle_index]
        error = linking_markers.get_error_marker(self._experiment.links, particle)
        return f"{error.get_severity().name} {current_particle_index + 1} / {len(particle_list)} "\
            f" of lineage {self._current_lineage_index + 1} / {len(self._problematic_lineages)} " \
               f"  ({self._total_number_of_warnings} warnings in total)" +\
            "\n" + error.get_message() + "\n" + str(particle)

    def _on_command(self, command: str) -> bool:
        if command == "recheck":
            self._recheck_errors()
            return True
        return super()._on_command(command)

    def _recheck_errors(self):
        logical_tests.apply(self._experiment)
        # Recalculate everything
        selected_particle = None
        if 0 <= self._current_particle_index < len(self._particle_list):
            selected_particle = self._particle_list[self._current_particle_index]
        activate(ErrorsVisualizer(self.get_window(), selected_particle))
        self.update_status("Rechecked all cells in the experiment. "
                           "Please note that suppressed warnings remain suppressed.")

    def _on_key_press(self, event: KeyEvent):
        if event.key == "up":
            self.__goto_next_lineage()
        elif event.key == "down":
            self.__goto_previous_lineage()
        elif event.key == "e":
            self._exit_view()
        elif event.key == "delete":
            self._suppress_error()
        else:
            super()._on_key_press(event)

    def __goto_previous_lineage(self):
        if len(self._problematic_lineages) < 1:
            return
        self._current_lineage_index -= 1
        if self._current_lineage_index < 0:
            self._current_lineage_index = len(self._problematic_lineages) - 1
        self._particle_list = self._problematic_lineages[self._current_lineage_index].errored_particles
        self._current_particle_index = 0
        self.draw_view()

    def __goto_next_lineage(self):
        if len(self._problematic_lineages) < 1:
            return
        self._current_lineage_index += 1
        if self._current_lineage_index >= len(self._problematic_lineages):
            self._current_lineage_index = 0
        self._particle_list = self._problematic_lineages[self._current_lineage_index].errored_particles
        self._current_particle_index = 0
        self.draw_view()

    def _exit_view(self):
        from autotrack.visualizer.link_and_position_editor import LinkAndPositionEditor

        if self._current_particle_index < 0 or self._current_particle_index >= len(self._particle_list):
            # Don't know where to go
            data_editor = LinkAndPositionEditor(self._window)
        else:
            viewed_particle = self._particle_list[self._current_particle_index]
            data_editor = LinkAndPositionEditor(self._window,
                                                time_point_number=viewed_particle.time_point_number(),
                                                z=int(viewed_particle.z),
                                                selected_particle=viewed_particle)
        activate(data_editor)

    def _suppress_error(self):
        if self._current_particle_index < 0 or self._current_particle_index >= len(self._particle_list):
            return
        particle = self._particle_list[self._current_particle_index]
        links = self._experiment.links
        error = linking_markers.get_error_marker(links, particle)
        linking_markers.suppress_error_marker(links, particle, error)
        del self._particle_list[self._current_particle_index]
        self.draw_view()
        self.update_status(f"Suppressed warning for {particle}")

