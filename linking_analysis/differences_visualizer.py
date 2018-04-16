import networkx
from matplotlib.backend_bases import KeyEvent
from matplotlib.figure import Figure

from imaging import Particle, Experiment
from imaging.particle_list_visualizer import ParticleListVisualizer
from typing import List, Optional


def _get_differences(experiment: Experiment) -> List[Particle]:
    base_links = experiment.particle_links()
    scratch_links = experiment.particle_links_scratch()
    if scratch_links is None or base_links is None:
        return []

    # The following code detects all links that were missed or made up. The viewer expects a list of particles, so we
    # pick a an arbitrary particle at the link, and add that to the list. (But only if the other particle was not
    # already in the list.)
    all_differences = set()
    only_in_scratch = networkx.difference(scratch_links, base_links)
    only_in_base = networkx.difference(base_links, scratch_links)
    for particle1, particle2 in only_in_scratch.edges():
        if not particle2 in all_differences:
            all_differences.add(particle1)
    for particle1, particle2 in only_in_base.edges():
        if not particle2 in all_differences:
            all_differences.add(particle1)
    return list(all_differences)


class DifferencesVisualizer(ParticleListVisualizer):
    """Shows all differences between the scratch and official data.
    Press left or right to move to the next difference.
    Press D to exit this view.
    """

    def __init__(self, experiment: Experiment, figure: Figure, start_particle: Optional[Particle]):
        super().__init__(experiment, figure,
                         chosen_particle=start_particle,
                         all_particles=_get_differences(experiment))

    def get_message_no_particles(self):
        return "No differences found between scratch and official data"

    def get_message_press_right(self):
        return "No differences found here between scratch and official data." \
               "\nPress the right arrow key to view the first difference in the sample."

    def get_title(self, particle_list: List[Particle], current_particle_index: int):
        particle = self._particle_list[self._current_particle_index]
        return "Difference " + str(current_particle_index + 1) + "/" + str(len(particle_list)) + "\n" + str(particle)

    def _on_key_press(self, event: KeyEvent):
        if event.key == "d":
            self.goto_full_image()
        else:
            super()._on_key_press(event)

