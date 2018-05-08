from typing import Iterable, Set, Dict

from core import Particle
from numpy import ndarray


def find_undetected_particles(labeled_image: ndarray, particles: Iterable[Particle]) -> Dict[Particle, str]:
    """Returns a dict of particle->error code for all particles that were undetected."""
    used_ids = dict()
    found_errors = dict()
    for particle in particles:
        id = labeled_image[int(particle.z), int(particle.y), int(particle.x)]
        if id == 0:
            found_errors[particle] = "Missed"
            continue
        if id in used_ids:
            found_errors[particle] = "Merged with " + str(used_ids[id])
            continue
        used_ids[id] = particle
    return found_errors