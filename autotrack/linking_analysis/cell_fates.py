from enum import Enum
from typing import Optional

from networkx import Graph

from autotrack.core.experiment import Experiment
from autotrack.core.particles import Particle
from autotrack.linking import cell_cycle


class CellFate(Enum):
    """Cells with either divide another time, or will never divide."""
    UNKNOWN = 0
    NON_DIVIDING = 1
    WILL_DIVIDE = 2


def get_fate(experiment: Experiment, links: Graph, particle: Particle) -> CellFate:
    """Checks if a cell will undergo a division later in the experiment. Returns None if not sure, because we are near
    the end of the experiment. max_time_point_number is the number of the last time point in the experiment."""
    max_time_point_number = experiment.last_time_point_number()
    next_division = cell_cycle.get_next_division(links, particle)
    if next_division is None:
        # Did the experiment end, or did the cell really stop dividing?
        if particle.time_point_number() + experiment.division_lookahead_time_points > max_time_point_number:
            return CellFate.UNKNOWN
        return CellFate.NON_DIVIDING
    return CellFate.WILL_DIVIDE
