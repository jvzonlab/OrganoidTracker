"""Small wrapper method to get the age of a particle. The age is the time since appearance or (only for cells) the last
division.."""

from typing import Optional

from autotrack.core.links import ParticleLinks
from autotrack.core.particles import Particle


def get_age(links: ParticleLinks, particle: Particle) -> Optional[int]:
    """Gets how many time steps ago this cell was born"""
    track = links.get_track(particle)
    if track is None:
        return 0
    return track.get_age(particle)
