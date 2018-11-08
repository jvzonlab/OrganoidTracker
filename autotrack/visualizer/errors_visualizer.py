from typing import List, Optional, Tuple, Dict, Any

from matplotlib.backend_bases import KeyEvent
from networkx import Graph

from autotrack.core.experiment import Experiment
from autotrack.core.particles import Particle
from autotrack.gui import Window
from autotrack.linking import existing_connections
from autotrack.linking_analysis import errors, logical_tests, linking_markers, cell_appearance_finder, lineage_checks
from autotrack.linking_analysis.errors import Error
from autotrack.linking_analysis.lineage_checks import LineageWithErrors
from autotrack.visualizer import DisplaySettings, activate
from autotrack.visualizer.particle_list_visualizer import ParticleListVisualizer


class ErrorsVisualizer(ParticleListVisualizer):
    """Shows all errors and warnings in the experiment.
    Press Left/Right to view the previous/next error.
    Press E to exit this view (so that you can for example make a correction to the data).
    """

    _problematic_lineages: List[LineageWithErrors]
    _lineage_index: int = 0
    _total_number_of_warnings: int

    def __init__(self, window: Window, start_particle: Optional[Particle]):
        links = window.get_experiment().links.get_scratch_else_baseline()
        start_particles = set() if start_particle is None else {start_particle}
        self._problematic_lineages = lineage_checks.get_problematic_lineages(links, start_particles)
        lineage_index = lineage_checks.find_lineage_index_with_crumb(self._problematic_lineages, start_particle)
        self._lineage_index = 0 if lineage_index is None else lineage_index
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
