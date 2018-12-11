from typing import List, Optional

from autotrack.core.experiment import Experiment
from autotrack.core.particles import Particle
from autotrack.gui.window import Window
from autotrack.linking_analysis import lineage_end_finder, linking_markers
from autotrack.linking_analysis.errors import Error
from autotrack.visualizer.particle_list_visualizer import ParticleListVisualizer


def _get_end_of_tracks(experiment: Experiment) -> List[Particle]:
    return list(lineage_end_finder.find_ended_tracks(experiment.links, experiment.last_time_point_number()))


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

        return f"Track end {self._current_particle_index + 1}/{len(self._particle_list)}    ({end_reason})" \
               f"\n{cell}"

    def _get_end_cause(self, particle: Particle) -> str:
        links = self._experiment.links

        end_reason = linking_markers.get_track_end_marker(links, particle)
        if end_reason is None:
            if linking_markers.is_error_suppressed(links, particle, Error.NO_FUTURE_POSITION):
                return "analyzed, but no conclusion"
            return "not analyzed"
        return end_reason.get_display_name()
