from matplotlib.backend_bases import KeyEvent
from matplotlib.figure import Figure

from gui import Window
from imaging import Particle, Experiment, cell
from imaging.particle_list_visualizer import ParticleListVisualizer
from linking import mother_finder
from typing import List, Optional

from linking_analysis import cell_death_finder


def _get_deaths(experiment: Experiment) -> List[Particle]:
    graph = experiment.particle_links()
    if graph is None:
        return []
    all_deaths = list(cell_death_finder.find_cell_deaths(experiment, graph))
    return all_deaths


class CellDeathVisualizer(ParticleListVisualizer):
    """Shows cells that are about to divide.
    Use the left/right arrow keys to move to the next cell division.
    Press M to exit this view."""

    def __init__(self, experiment: Experiment, window: Window, mother: Optional[Particle]):
        super().__init__(experiment, window, chosen_particle=mother, all_particles=_get_deaths(experiment),
                         show_next_image=True)

    def get_message_no_particles(self):
        return "No cell deaths found. Is the linking data missing?"

    def get_message_press_right(self):
        return "No cell death found at mouse position." \
               "\nPress the right arrow key to view the first mother in the sample."

    def get_title(self, all_cells: List[Particle], cell_index: int):
        cell = all_cells[cell_index]
        recognized_str = ""
        if self._was_recognized(cell) is False:
            recognized_str = "    (NOT RECOGNIZED)"
        return "Cell death " + str(self._current_particle_index + 1) + "/" + str(len(self._particle_list))\
               + recognized_str + "\n" + str(cell)

    def _was_recognized(self, mother: Particle) -> Optional[bool]:
        """Gets if a death was correctly recognized by the scratch graph. Returns None if there is no scratch graph."""
        main_graph = self._experiment.particle_links()
        scratch_graph = self._experiment.particle_links_scratch()
        if main_graph is None or scratch_graph is None:
            return None

        try:
            connections_scratch = cell.find_future_particles(scratch_graph, mother)
            return len(connections_scratch) == 0
        except KeyError:
            return False
