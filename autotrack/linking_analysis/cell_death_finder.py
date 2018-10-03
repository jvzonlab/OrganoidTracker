from typing import Set

from networkx import Graph

from autotrack.core.experiment import Experiment
from autotrack.core.particles import Particle


def find_cell_deaths(graph: Graph, last_time_point_number: int) -> Set[Particle]:
    """Returns a set of all cells that appear as dead in the experiment."""
    dead_cells = set()

    for particle in graph.nodes():
        time_point_number = particle.time_point_number()
        if time_point_number >= last_time_point_number - 1:
            continue

        links = graph[particle]
        links_to_future_count = 0
        for link in links:
            if link.time_point_number() > time_point_number:
                links_to_future_count += 1

        if links_to_future_count == 0:
            dead_cells.add(particle)

    return dead_cells
