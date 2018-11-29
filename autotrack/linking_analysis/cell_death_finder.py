from typing import Set

from autotrack.core.links import ParticleLinks
from autotrack.core.particles import Particle


def find_ended_tracks(links: ParticleLinks, last_time_point_number: int) -> Set[Particle]:
    """Returns a set of all cells that have no links to the future."""
    ended_lineages = set()

    for particle in links.find_all_particles():
        time_point_number = particle.time_point_number()
        if time_point_number >= last_time_point_number - 1:
            continue

        links_to_future = links.find_futures(particle)

        if len(links_to_future) == 0:
            ended_lineages.add(particle)

    return ended_lineages

