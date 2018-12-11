"""Finds particles which have no link to the future."""

from typing import Set

from autotrack.core.links import ParticleLinks
from autotrack.core.particles import Particle


def find_ended_tracks(links: ParticleLinks, last_time_point_number: int) -> Set[Particle]:
    """Returns a set of all cells that have no links to the future."""
    ended_lineages = set()

    for track in links.find_all_tracks():
        time_point_number = track.max_time_point_number()
        if time_point_number >= last_time_point_number - 1:
            continue

        next_track = track.get_next_tracks()

        if len(next_track) == 0:
            ended_lineages.add(track.find_last_particle())

    return ended_lineages

