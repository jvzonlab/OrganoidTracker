from typing import Iterable, Set

from networkx import Graph

from autotrack.core.particles import Particle


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
        # No previous position
        return None
    if len(previous_positions) > 1:
        # Multiple previous positions
        return None
    return previous_positions.pop()


def find_future_particles(graph: Graph, particle: Particle) -> Set[Particle]:
    # All possible connections one step in the future
    linked_particles = graph[particle]
    return {linked_particle for linked_particle in linked_particles
            if linked_particle.time_point_number() > particle.time_point_number()}


def find_preferred_future_particles(graph: Graph, particle: Particle) -> Set[Particle]:
    return find_preferred_links(graph, particle, find_future_particles(graph, particle))
