from typing import List, Optional

from autotrack.core.links import ParticleLinks
from autotrack.core.particles import Particle


def find_previous_positions(particle: Particle, links: ParticleLinks, steps_back: int) -> Optional[List[Particle]]:
    """Gets a list consisting of the given particle and steps_back particles in previous time points. Returns None if
    we can't get back that many time points. Index 0 will be the given particle, index 1 the particle one time step back,
    etc."""
    particle_list = [particle]
    while particle_list[0].time_point_number() - particle.time_point_number() < steps_back:
        previous_set = links.find_pasts(particle)
        if len(previous_set) == 0:
            return None
        particle = previous_set.pop()
        particle_list.append(particle)
    return particle_list
