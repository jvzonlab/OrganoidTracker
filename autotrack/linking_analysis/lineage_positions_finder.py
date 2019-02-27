from typing import Iterable

from autotrack.core.links import Links
from autotrack.core.position import Position


def find_all_positions_in_lineage_of(links: Links, position: Position) -> Iterable[Position]:
    """Gets all positions in the track of the given position, and all tracks after it."""
    track = links.get_track(position)
    if track is None:
        yield position
        return

    for future_track in track.find_all_descending_tracks(include_self=True):
        yield from future_track.positions()
