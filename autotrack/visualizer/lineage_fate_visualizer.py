from typing import Dict, Optional

from autotrack.core import TimePoint
from autotrack.core.particles import Particle
from autotrack.linking_analysis import lineage_fates
from autotrack.linking_analysis.lineage_fates import LineageFate
from autotrack.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def _lineage_fate_to_text(lineage_fate: Optional[LineageFate]):
    if lineage_fate is None or lineage_fate.errors > 0:
        return "?"
    if lineage_fate.divisions > 0:
        if lineage_fate.deaths > 0:
            return f"{lineage_fate.divisions}, {lineage_fate.deaths}X"
        return str(lineage_fate.divisions)
    if lineage_fate.deaths > 0:
        return "X"
    if lineage_fate.ends > 0:
        return "~|"
    return "~"


def _lineage_fate_to_color(lineage_fate):
    if lineage_fate is None or lineage_fate.errors > 0:
        return "black"
    if lineage_fate.divisions > 0:
        return "green"
    if lineage_fate.deaths > 0:
        return "blue"
    return "black"


class LineageFateVisualizer(ExitableImageVisualizer):
    """Shows how each cell will develop during the experiment. Note: time points past the current time point are not
    included. Legend:
    ?   no reliable linking data available.
    X   cell died,   ~|   lineage ended for some other reason.
    4   cell divided four times. "4, 1X" means cell divided four times, one offspring cell died."
    ~   no events, just movement during the complete experiment."""

    _lineage_fates: Dict[Particle, LineageFate] = dict()

    def _load_time_point(self, time_point: TimePoint):
        super()._load_time_point(time_point)

        # Check what lineages contain errors
        links = self._experiment.links
        if not links.has_links():
            self._lineage_fates = dict()
            return

        particles = self._experiment.particles.of_time_point(time_point)
        links = self._experiment.links
        last_time_point_number = self._experiment.last_time_point_number()
        result = dict()
        for particle in particles:
            result[particle] = lineage_fates.get_lineage_fate(particle, links, last_time_point_number)
        self._lineage_fates = result

    def _draw_particle(self, particle: Particle, color: str, dz: int, dt: int):
        if dt == 0 and abs(dz) <= 3:
            lineage_fate = self._lineage_fates.get(particle)
            color = _lineage_fate_to_color(lineage_fate)
            self._ax.annotate( _lineage_fate_to_text(lineage_fate), (particle.x, particle.y), fontsize=12 - abs(dz),
                               fontweight="bold", color=color, backgroundcolor=(1,1,1,0.5))
