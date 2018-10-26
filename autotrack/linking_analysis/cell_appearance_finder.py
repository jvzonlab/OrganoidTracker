from typing import Optional, Iterable

from networkx import Graph

from autotrack.core.particles import Particle
from autotrack.linking import existing_connections


def find_appeared_cells(graph: Graph, time_point_number_to_ignore: Optional[int] = None) -> Iterable[Particle]:
    """This method gets all particles that "popped up out of nothing": that have no links to the past. You can give this
    method a time point number to ignore. Usually, this would be the first time point number of the experiment, as
    cells that have no links to the past in the first time point are not that interesting."""
    for particle in graph.nodes():
        if len(existing_connections.find_past_particles(graph, particle)) == 0:
            if time_point_number_to_ignore is None or time_point_number_to_ignore != particle.time_point_number():
                yield particle
