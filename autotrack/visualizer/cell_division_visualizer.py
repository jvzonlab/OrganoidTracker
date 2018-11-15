from typing import List, Optional

from matplotlib.backend_bases import KeyEvent

from autotrack.core.experiment import Experiment
from autotrack.core.particles import Particle
from autotrack.linking.existing_connections import find_future_particles
from autotrack.visualizer.particle_list_visualizer import ParticleListVisualizer
from autotrack.gui import Window
from autotrack.linking import mother_finder


def _get_mothers(experiment: Experiment) -> List[Particle]:
    graph = experiment.links.graph
    if graph is None:
        return []
    all_mothers = list(mother_finder.find_mothers(graph))
    return all_mothers


class CellDivisionVisualizer(ParticleListVisualizer):
    """Shows cells that are about to divide.
    Use the left/right arrow keys to move to the next cell division.
    Press M to exit this view."""

    def __init__(self, window: Window):
        super().__init__(window, all_particles=_get_mothers(window.get_experiment()),
                         show_next_image=True)

    def get_message_no_particles(self):
        return "No mothers found. Is the linking data missing?"

    def get_message_press_right(self):
        return "No mother found at mouse position.\nPress the right arrow key to view the first mother in the sample."

    def get_title(self, all_cells: List[Particle], cell_index: int):
        mother = all_cells[cell_index]
        recognized_str = ""
        return "Mother " + str(self._current_particle_index + 1) + "/" + str(len(self._particle_list))\
               + recognized_str + "\n" + str(mother)

    def _on_key_press(self, event: KeyEvent):
        if event.key == "m":
            self.goto_full_image()
        else:
            super()._on_key_press(event)
