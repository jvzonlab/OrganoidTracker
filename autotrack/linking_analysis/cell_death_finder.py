from typing import Set

from networkx import Graph

from autotrack.core.particles import Particle
from autotrack.linking import existing_connections


def find_ended_tracks(graph: Graph, last_time_point_number: int) -> Set[Particle]:
    """Returns a set of all cells that have no links to the future."""
    ended_lineages = set()

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
            ended_lineages.add(particle)

    return ended_lineages

