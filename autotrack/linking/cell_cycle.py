from typing import Optional, Iterable, Set

from networkx import Graph

from autotrack.core.particles import Particle
from autotrack.core.score import Family
from autotrack.linking.existing_connections import find_future_particles, find_preferred_links, find_past_particles, \
    find_preferred_past_particle


def get_age(graph: Graph, particle: Particle) -> Optional[int]:
    """Gets how many time steps ago this cell was born"""
    timesteps_ago = 0

    while True:
        timesteps_ago += 1

        particle = find_preferred_past_particle(graph, particle)
        if particle is None:
            return None # Cell first appeared here, don't know age for sure
        daughters = find_preferred_links(graph, particle, find_future_particles(graph, particle))
        if len(daughters) > 1:
            return timesteps_ago


def get_next_division(graph: Graph, particle: Particle) -> Optional[Family]:
    """Gets the next division for the given particle. Returns None if there is no such division. Raises ValueError if a
    cell with more than two daughters is found in this lineage."""
    while True:
        next_particles = find_future_particles(graph, particle)
        if len(next_particles) == 0:
            # Cell death or end of experiment
            return None
        if len(next_particles) == 1:
            # Go to next time point
            particle = next_particles.pop()
            continue
        if len(next_particles) == 2:
            # Found the next division
            return Family(particle, *next_particles)
        raise ValueError("Cell " + str(particle) + " has multiple daughters: " + str(next_particles))
