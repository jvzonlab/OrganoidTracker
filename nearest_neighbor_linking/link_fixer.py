# The following rules can be used:
# * Every cell must have one or two cells in the next image
# * Every cell must have exactly one cell in the previous image

import imaging
from networkx import Graph
from typing import Iterable
from imaging import Particle


def prune_links(graph: Graph) -> Graph:
    for particle in graph.nodes():
        _fix_no_future_particle(graph, particle)

    return _with_only_the_preferred_edges(graph)


def _fix_no_future_particle(graph: Graph, particle: Particle):
    """This fixes the case where a particle has no future particle lined up"""
    future_particles = _find_future_particles(graph, particle)
    future_preferred_particles = _find_preferred_links(graph, particle, future_particles)

    if len(future_preferred_particles) > 0:
        return

    # Oops, found dead end. Choose a best match from the future_particles list
    newly_matched_future_particle = imaging.get_closest_particle(future_particles, particle)
    if newly_matched_future_particle is None:
        print("Found no future cell for " + str(particle) + ", dead cell?")
        return

    # Downgrade existing edges of new best match, as it is getting a new best match
    for old_connection_of_new_match in _find_past_particles(graph, newly_matched_future_particle):
        graph.add_edge(old_connection_of_new_match, newly_matched_future_particle, pref=False)

    graph.add_edge(particle, newly_matched_future_particle, pref=True)
    print("Connected cell " + str(particle) + " to a new match")

# Helper functions below

def _find_preferred_links(graph: Graph, particle: Particle, linked_particles: Iterable[Particle]):
    return [linked_particle for linked_particle in linked_particles
            if graph[particle][linked_particle]["pref"] == True]


def _find_past_particles(graph: Graph, particle: Particle):
    # all possible connections one step in the past
    linked_particles = graph[particle]
    return [linked_particle for linked_particle in linked_particles
            if linked_particle.frame_number() < particle.frame_number()]


def _find_future_particles(graph: Graph, particle: Particle):
    # All possible connections one step in the future
    linked_particles = graph[particle]
    return [linked_particle for linked_particle in linked_particles
            if linked_particle.frame_number() > particle.frame_number()]


def _with_only_the_preferred_edges(old_graph: Graph):
    graph = Graph()
    for node in old_graph.nodes():
        if not isinstance(node, Particle):
            raise ValueError("Found a node that was not a particle: " + str(node))
        graph.add_node(node)

    for particle_1, particle_2, data in old_graph.edges(data=True):
        if data["pref"]:
            graph.add_edge(particle_1, particle_2)
    return graph