from typing import List, Optional, Tuple, Dict, Any

from matplotlib.backend_bases import KeyEvent
from networkx import Graph

from autotrack.core.experiment import Experiment
from autotrack.core.particles import Particle
from autotrack.gui import Window
from autotrack.linking import existing_connections
from autotrack.linking_analysis import errors, logical_tests, linking_markers, cell_appearance_finder
from autotrack.linking_analysis.errors import Error
from autotrack.visualizer import DisplaySettings, activate
from autotrack.visualizer.particle_list_visualizer import ParticleListVisualizer

class _LineageWithErrors:
    start: Particle
    errored_particles: List[Particle]
    contains_crumb: bool = False  # The particle the user has selected is called the "crumb"

    def __init__(self, start: Particle):
        self.start = start
        self.errored_particles = []


def _get_problematic_particles(experiment: Experiment, crumb: Optional[Particle]) -> List[_LineageWithErrors]:
    graph = experiment.links.get_scratch_else_baseline()
    lineages_with_errors = []
    for starting_cell in cell_appearance_finder.find_appeared_cells(graph):
        lineage = _LineageWithErrors(starting_cell)
        _find_errors_in_lineage(graph, lineage, starting_cell, crumb)
        if len(lineage.errored_particles) > 0:
            lineages_with_errors.append(lineage)
    return lineages_with_errors


def _find_errors_in_lineage(graph: Graph, lineage: _LineageWithErrors, particle: Particle, crumb: Optional[Particle]):
    while True:
        if particle == crumb:
            print("Found crumb")
            lineage.contains_crumb = True

        error = linking_markers.get_error_marker(graph, particle)
        if error is not None:
            lineage.errored_particles.append(particle)
        future_particles = existing_connections.find_future_particles(graph, particle)

        if len(future_particles) > 1:
            # Branch out
            for future_particle in future_particles:
                _find_errors_in_lineage(graph, lineage, future_particle, crumb)
            return
        if len(future_particles) < 1:
            # Stop
            return
        # Continue
        particle = future_particles.pop()


def _find_lineage_index_with_crumb(lineages: List[_LineageWithErrors]):
    """Attempts to find the given particle in the lineages. Returns 0 if the particle is None or not in the lineages."""
    for index, lineage in enumerate(lineages):
        if lineage.contains_crumb:
            print("Crumb in lineage", index)
            return index
    return 0


class ErrorsVisualizer(ParticleListVisualizer):
    """Shows all errors and warnings in the sample.
    Press E to exit this view (so that you can for example make a correction to the data).
    """

    _problematic_lineages: List[_LineageWithErrors]
    _lineage_index: int = 0
    _total_number_of_warnings: int

    def __init__(self, window: Window, start_particle: Optional[Particle]):
        self._problematic_lineages = _get_problematic_particles(window.get_experiment(), start_particle)
        self._lineage_index = _find_lineage_index_with_crumb(self._problematic_lineages)
        particles = []
        if len(self._problematic_lineages) > 0:
            particles = self._problematic_lineages[self._lineage_index].errored_particles
        self._total_number_of_warnings = sum((len(lineage.errored_particles) for lineage in self._problematic_lineages))
        super().__init__(window,
                         chosen_particle=start_particle,
                         all_particles=particles)

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "View/Exit-Exit this view (/exit)": lambda: self.goto_full_image(),
            "Navigate/Lineage-Next lineage (Up)": self.__goto_next_lineage,
            "Navigate/Lineage-Previous lineage (Down)": self.__goto_previous_lineage
        }

    def get_message_no_particles(self):
        return "No warnings or errors found. Hurray?"

    def get_message_press_right(self):
        return "No warnings or errors found in lineage." \
               "\nPress the right arrow key to view the first warning in the experiment."

    def get_title(self, particle_list: List[Particle], current_particle_index: int):
        particle = particle_list[current_particle_index]
        error = linking_markers.get_error_marker(self._experiment.links.get_scratch_else_baseline(), particle)
        return f"{error.get_severity().name} {current_particle_index + 1} / {len(particle_list)} "\
            f" of lineage {self._lineage_index + 1} / {len(self._problematic_lineages)} " \
               f"  ({self._total_number_of_warnings} warnings in total)" +\
            "\n" + error.get_message() + "\n" + str(particle)

    def _on_key_press(self, event: KeyEvent):
        if event.key == "up":
            self.__goto_next_lineage()
        elif event.key == "down":
            self.__goto_previous_lineage()
        elif event.key == "e":
            self.goto_full_image()
        elif event.key == "c":
            self._edit_data()
        else:
            super()._on_key_press(event)

    def __goto_previous_lineage(self):
        if len(self._problematic_lineages) < 1:
            return
        self._lineage_index -= 1
        if self._lineage_index < 0:
            self._lineage_index = len(self._problematic_lineages) - 1
        self._particle_list = self._problematic_lineages[self._lineage_index].errored_particles
        self._current_particle_index = 0
        self.draw_view()

    def __goto_next_lineage(self):
        if len(self._problematic_lineages) < 1:
            return
        self._lineage_index += 1
        if self._lineage_index >= len(self._problematic_lineages):
            self._lineage_index = 0
        self._particle_list = self._problematic_lineages[self._lineage_index].errored_particles
        self._current_particle_index = 0
        self.draw_view()

    def _edit_data(self):
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
