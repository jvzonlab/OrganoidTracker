from typing import List, Optional, Tuple

from matplotlib.backend_bases import KeyEvent
from networkx import Graph

from autotrack.core.experiment import Experiment
from autotrack.core.particles import Particle
from autotrack.gui import Window
from autotrack.linking import errors, logical_tests
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
    scratch_graph = experiment.particle_links_scratch()
    if scratch_graph is not None:
        return scratch_graph
    official_graph = experiment.particle_links()
    if official_graph is None:
        return None
    scratch_graph = official_graph.copy()
    experiment.particle_links_scratch(scratch_graph)
    return scratch_graph


class ErrorsVisualizer(ParticleListVisualizer):
    """Shows all errors and warnings in the sample.
    Press E to exit this view (so that you can for example make a correction to the data).
    Press DELETE to delete an error or warning.
    Type /recheck to recheck the experiment for obvious errors.
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
        error_type, message, is_edited = self._get_warning_info(particle)
        outdated_str = ""
        if is_edited:
            outdated_str = " (edited afterwards)"
        return error_type + " " + str(current_particle_index + 1) + "/" + str(len(particle_list)) + outdated_str + \
            "\n" + message + "\n" + str(particle)

    def _get_warning_info(self, particle: Particle) -> Tuple[str, str, bool]:
        graph = self._experiment.particle_links_scratch()
        data = graph.nodes[particle]
        is_edited = "edited" in data and data["edited"]
        if "error" in data:
            error = data["error"]
            return errors.get_severity(error).name, errors.get_message(error), is_edited
        return "UNKNOWN", str(data), is_edited

    def _delete_warning(self):
        if self._current_particle_index < 0 or self._current_particle_index >= len(self._particle_list):
            return
        particle = self._particle_list[self._current_particle_index]
        self._experiment.particle_links_scratch().add_node(particle, error=None, warning=None)
        self._particle_list.remove(particle)
        if self._current_particle_index >= len(self._particle_list):
            self._current_particle_index -= 1  # Deleted last particle, go back to previous
        self.draw_view()

    def _on_key_press(self, event: KeyEvent):
        if event.key == "e":
            self.goto_full_image()
        elif event.key == "delete":
            self._delete_warning()
        else:
            super()._on_key_press(event)

    def _on_command(self, command: str):
        if command == "recheck":
            logical_tests.apply(self._experiment, _get_links(self._experiment))
            self._particle_list = _get_problematic_particles(self._experiment)
            self.draw_view()
            self.update_status("Checked links for errors")
            return True
        if command == "help":
            self.update_status("Available commands:\n"
                               "/recheck - checks all links for obvious errors\n"
                               "/exit - exits this view")
            return True
        return super()._on_command(command)
