from typing import List, Optional, Tuple

from matplotlib.backend_bases import KeyEvent
from networkx import Graph

from autotrack.core.experiment import Experiment
from autotrack.core.particles import Particle
from autotrack.gui import Window
from autotrack.linking_analysis import errors, logical_tests, linking_markers
from autotrack.linking_analysis.errors import Error
from autotrack.visualizer import DisplaySettings, activate
from autotrack.visualizer.particle_list_visualizer import ParticleListVisualizer


def _get_problematic_particles(experiment: Experiment) -> List[Particle]:
    particles = []
    graph = _get_links(experiment)
    if graph is None:
        return []
    for particle, data in graph.nodes(data=True):
        if "error" in data and data["error"] is not None:
            particles.append(particle)
    return particles


def _get_links(experiment: Experiment) -> Optional[Graph]:
    return experiment.links.get_scratch_else_baseline()


class ErrorsVisualizer(ParticleListVisualizer):
    """Shows all errors and warnings in the sample.
    Press E to exit this view (so that you can for example make a correction to the data).
    """

    def __init__(self, window: Window, start_particle: Optional[Particle]):
        super().__init__(window,
                         chosen_particle=start_particle,
                         all_particles=_get_problematic_particles(window.get_experiment()))

    def get_message_no_particles(self):
        return "No warnings or errors found. Hurray?"

    def get_message_press_right(self):
        return "No warnings or errors found at mouse position." \
               "\nPress the right arrow key to view the first warning in the sample."

    def get_title(self, particle_list: List[Particle], current_particle_index: int):
        particle = particle_list[current_particle_index]
        error = linking_markers.get_error_marker(self._experiment.links.get_scratch_else_baseline(), particle)
        return error.get_severity().name + " " + str(current_particle_index + 1) + "/" + str(len(particle_list)) + \
            "\n" + error.get_message() + "\n" + str(particle)

    def _on_key_press(self, event: KeyEvent):
        if event.key == "e":
            self.goto_full_image()
        elif event.key == "c":
            self._edit_data()
        else:
            super()._on_key_press(event)

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
