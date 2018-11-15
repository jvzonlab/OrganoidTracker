from typing import List, Optional

from autotrack.core.experiment import Experiment
from autotrack.core.particles import Particle
from autotrack.linking.existing_connections import find_future_particles
from autotrack.linking_analysis.linking_markers import EndMarker
from autotrack.visualizer.particle_list_visualizer import ParticleListVisualizer
from autotrack.gui import Window
from autotrack.linking_analysis import cell_death_finder, linking_markers


def _get_end_of_tracks(experiment: Experiment) -> List[Particle]:
    graph = experiment.links.graph
    if graph is None:
        return []
    all_deaths = list(cell_death_finder.find_ended_tracks(graph, experiment.last_time_point_number()))
    return all_deaths


class CellTrackEndVisualizer(ParticleListVisualizer):
    """Shows cells that are about to divide.
    Use the left/right arrow keys to move to the next cell division.
    Type /exit to exit this view."""

    def __init__(self, window: Window, focus_particle: Optional[Particle]):
        super().__init__(window, chosen_particle=focus_particle,
                         all_particles=_get_end_of_tracks(window.get_experiment()), show_next_image=False)

    def get_message_no_particles(self):
        return "No ending cell tracks found. Is the linking data missing?"

    def get_message_press_right(self):
        return "No end of cell track found at mouse position." \
               "\nPress the right arrow key to view the first end of a cell track in the sample."

    def get_title(self, all_cells: List[Particle], cell_index: int):
        cell = all_cells[cell_index]
        end_reason = self._get_end_cause(cell)

        description = ""
        if end_reason is not None:
            description = "    (" + end_reason.get_display_name() + ")"
        return "Track end " + str(self._current_particle_index + 1) + "/" + str(len(self._particle_list))\
               + description + "\n" + str(cell)

    def _get_end_cause(self, particle: Particle) -> Optional[EndMarker]:
        links = self._experiment.links.graph
        return linking_markers.get_track_end_marker(links, particle)
