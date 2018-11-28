from typing import Optional

from autotrack.core.links import ParticleLinks
from autotrack.core.particles import Particle
from autotrack.core.score import Family


def get_age(links: ParticleLinks, particle: Particle) -> Optional[int]:
    """Gets how many time steps ago this cell was born"""
    timesteps_ago = 0

    while True:
        timesteps_ago += 1

        particles = links.find_pasts(particle)
        if len(particles) != 1:
            return None  # Cell first appeared here (or we have a cell merge), don't know age for sure
        particle = particles.pop()
        daughters = links.find_futures(particle)
        if len(daughters) > 1:
            return timesteps_ago


def get_next_division(links: ParticleLinks, particle: Particle) -> Optional[Family]:
    """Gets the next division for the given particle. Returns None if there is no such division. Raises ValueError if a
    cell with more than two daughters is found in this lineage."""
    while True:
        next_particles = links.find_futures(particle)
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
