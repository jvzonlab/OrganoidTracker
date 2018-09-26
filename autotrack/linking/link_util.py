from typing import Iterable, Set, Optional

import numpy
from networkx import Graph

from autotrack import core
from autotrack.core import Particle, Experiment, Family
from autotrack.imaging import normalized_image
from autotrack.imaging.normalized_image import ImageEdgeError
from autotrack.linking import mother_finder, errors
from autotrack.linking.scoring_system import MotherScoringSystem


def find_preferred_links(graph: Graph, particle: Particle, linked_particles: Iterable[Particle]):
    preferred_particles = set()
    for linked_particle in linked_particles:
        link = graph[particle][linked_particle]
        if "pref" not in link or link["pref"] is True:
            preferred_particles.add(linked_particle)
    return preferred_particles


def find_past_particles(graph: Graph, particle: Particle):
    # all possible connections one step in the past
    linked_particles = graph[particle]
    return {linked_particle for linked_particle in linked_particles
            if linked_particle.time_point_number() < particle.time_point_number()}


def find_preferred_past_particle(graph: Graph, particle: Particle):
    # the one most likely connection one step in the past
    previous_positions = find_preferred_links(graph, particle, find_past_particles(graph, particle))
    if len(previous_positions) == 0:
        print("Error at " + str(particle) + ": cell popped up out of nothing")
        return None
    if len(previous_positions) > 1:
        print("Error at " + str(particle) + ": cell originated from two different cells")
        return None
    return previous_positions.pop()


def find_future_particles(graph: Graph, particle: Particle) -> Set[Particle]:
    # All possible connections one step in the future
    linked_particles = graph[particle]
    return {linked_particle for linked_particle in linked_particles
            if linked_particle.time_point_number() > particle.time_point_number()}


def find_preferred_future_particles(graph: Graph, particle: Particle) -> Set[Particle]:
    return find_preferred_links(graph, particle, find_future_particles(graph, particle))


def with_only_the_preferred_edges(old_graph: Graph):
    graph = Graph()
    for node, data in old_graph.nodes(data=True):
        if not isinstance(node, Particle):
            raise ValueError("Found a node that was not a particle: " + str(node))
        graph.add_node(node, **data)

    for particle_1, particle_2, data in old_graph.edges(data=True):
        if data["pref"]:
            graph.add_edge(particle_1, particle_2)
    return graph
