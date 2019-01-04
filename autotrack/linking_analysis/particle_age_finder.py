"""Small wrapper method to get the age of a position. The age is the time since appearance or (only for cells) the last
division.."""

from typing import Optional

from autotrack.core.links import Links
from autotrack.core.position import Position


def get_age(links: Links, position: Position) -> Optional[int]:
    """Gets how many time steps ago this cell was born"""
    track = links.get_track(position)
    if track is None:
        return 0
    return track.get_age(position)
