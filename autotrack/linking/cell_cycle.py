from typing import Optional

from autotrack.core.links import ParticleLinks
from autotrack.core.particles import Particle
from autotrack.core.score import Family


def get_age(links: ParticleLinks, particle: Particle) -> Optional[int]:
    """Gets how many time steps ago this cell was born"""
    track = links.get_track(particle)
    if track is None:
        return 0
    return track.get_age(particle)


def get_next_division(links: ParticleLinks, particle: Particle) -> Optional[Family]:
    """Gets the next division for the given particle. Returns None if there is no such division. Raises ValueError if a
    cell with more than two daughters is found in this lineage."""
    track = links.get_track(particle)
    if track is None:
        return None

    next_tracks = track.get_next_tracks()
    if len(next_tracks) == 0:
        return None

    next_daughters = [next_track.find_first_particle() for next_track in next_tracks]
    if len(next_daughters) != 2:
        raise ValueError("Cell " + str(track.find_last_particle()) + " has multiple daughters: " + str(next_daughters))

    return Family(track.find_last_particle(), *next_daughters)
