"""Finds positions which have no link to the future."""

from typing import Set

from autotrack.core.links import PositionLinks
from autotrack.core.positions import Position


def find_ended_tracks(links: PositionLinks, last_time_point_number: int) -> Set[Position]:
    """Returns a set of all cells that have no links to the future."""
    ended_lineages = set()

    for track in links.find_all_tracks():
        time_point_number = track.max_time_point_number()
        if time_point_number >= last_time_point_number - 1:
            continue

        next_track = track.get_next_tracks()

        if len(next_track) == 0:
            ended_lineages.add(track.find_last_position())

    return ended_lineages
