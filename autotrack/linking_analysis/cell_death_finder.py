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


def is_actual_death(graph: Graph, particle: Particle) -> bool:
    """Checks if the particle at the end of the lineage has been marked as a cell death. Returns False if the cell is
    not at the end of the lineage. Also returns False if the cell is at the end of the lineage, but has not been marked
    as a cell death (so it could just be a cell that goes out of the view.)"""
    node_data = graph.nodes.get(particle)
    if node_data is None:
        return False
    return bool(node_data.get("dead"))


def mark_actual_death(graph: Graph, particle: Particle, dead: bool = True):
    """Marks an end-of-lineage as an actual cell death (and not just as a cell moving out of the view). Can also be used
    to remove such a marker. Throws ValueError if the cell is not in the linking graph, or if dead=True and the cell has
    links to the future."""
    if particle not in graph.nodes:
        raise ValueError(f"{particle} is not in the linking graph")
    if dead and len(existing_connections.find_future_particles(graph, particle)) > 0:
        raise ValueError(f"{particle} has links to the future, so it cannot be a cell death")

    if dead:
        graph.nodes[particle]["dead"] = True
    else:
        del graph.nodes[particle]["dead"]
