from typing import Dict, Optional

from autotrack.core import TimePoint
from autotrack.core.particles import Particle
from autotrack.linking_analysis import cell_fates
from autotrack.linking_analysis.cell_fates import CellFateType, CellFate
from autotrack.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def _cell_fate_to_text(cell_fate: CellFate):
    if cell_fate.type == CellFateType.JUST_MOVING:
        return "~"
    if cell_fate.type == CellFateType.WILL_DIE:
        return "X in " + str(cell_fate.time_points_remaining)
    if cell_fate.type == CellFateType.WILL_DIVIDE:
        return "Div in " + str(cell_fate.time_points_remaining)
    return "?"


def _cell_fate_to_color(cell_fate: CellFate):
    if cell_fate.type == CellFateType.WILL_DIVIDE:
        return "green"
    if cell_fate.type == CellFateType.WILL_DIE:
        return "blue"
    return "black"


class CellFateVisualizer(ExitableImageVisualizer):
    """Shows how each cell will develop during the experiment. Note: time points past the current time point are not
    included. Legend:
    ?         unknown cell fate - cell moved out of view, the experiment will end soon or there are unresolved warnings
    X in 13   cell will die in 13 time points
    < in 16   cell will divide in 16 time points
    ~         no events, just movement."""

    _cell_fates: Dict[Particle, CellFate] = dict()

    def _load_time_point(self, time_point: TimePoint):
        super()._load_time_point(time_point)

        # Check what lineages contain errors
        links = self._experiment.links
        if not links.has_links():
            self._cell_fates = dict()
            return

        particles = self._experiment.particles.of_time_point(time_point)
        result = dict()
        for particle in particles:
            result[particle] = cell_fates.get_fate(self._experiment, particle)
        self._cell_fates = result

    def _draw_particle(self, particle: Particle, color: str, dz: int, dt: int):
        if dt == 0 and abs(dz) <= 3:
            cell_fate = self._cell_fates.get(particle)
            if cell_fate is None:
                cell_fate = CellFate(CellFateType.UNKNOWN, None)
            color = _cell_fate_to_color(cell_fate)
            self._ax.annotate(_cell_fate_to_text(cell_fate), (particle.x, particle.y), fontsize=8 - abs(dz / 2),
                              fontweight="bold", color=color, backgroundcolor=(1,1,1,0.8))
