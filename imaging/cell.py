from imaging import Experiment, Particle
from typing import Optional, Iterable
from networkx import Graph


def get_age(experiment: Experiment, graph: Graph, particle: Particle) -> Optional[int]:
    """Gets how many time steps ago this cell was born"""
    timesteps_ago = 0
    min_frame_number = experiment.first_frame_number()

    while particle.frame_number() > min_frame_number + 1:
        timesteps_ago += 1

        particle = _find_past_particle(graph, particle)
        if particle is None:
            return None # Cell first appeared here, don't know age for sure
        daughters = _find_preferred_links(graph, particle, _find_future_particles(graph, particle))
        if len(daughters) > 1:
            return timesteps_ago
    return None # Has never divided since start of measurement


def _find_preferred_links(graph: Graph, particle: Particle, linked_particles: Iterable[Particle]):
    return {linked_particle for linked_particle in linked_particles
            if "pref" not in graph[particle][linked_particle] or graph[particle][linked_particle]["pref"] == True}


def _find_past_particles(graph: Graph, particle: Particle):
    # all possible connections one step in the past
    linked_particles = graph[particle]
    return {linked_particle for linked_particle in linked_particles
            if linked_particle.frame_number() < particle.frame_number()}


def _find_past_particle(graph: Graph, particle: Particle):
    # the one most likely connection one step in the past
    previous_positions = _find_preferred_links(graph, particle, _find_past_particles(graph, particle))
    if len(previous_positions) == 0:
        print("Error at " + str(particle) + ": cell popped up out of nothing")
        return None
    if len(previous_positions) > 1:
        print("Error at " + str(particle) + ": cell originated from two different cells")
        return None
    return previous_positions.pop()


def _find_future_particles(graph: Graph, particle: Particle):
    # All possible connections one step in the future
    linked_particles = graph[particle]
    return {linked_particle for linked_particle in linked_particles
            if linked_particle.frame_number() > particle.frame_number()}