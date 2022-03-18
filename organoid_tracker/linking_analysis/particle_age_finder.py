"""Small wrapper method to get the age of a position. The age is the time since appearance or (only for cells) the last
division.."""

from typing import Optional

from organoid_tracker.core.links import Links, LinkingTrack
from organoid_tracker.core.position import Position


def get_age(links: Links, position: Position) -> Optional[int]:
    """Gets how many time steps ago this cell was born"""
    track = links.get_track(position)
    if track is None:
        return None  # Not in a track, cannot calculate
    if len(track.get_previous_tracks()) != 1:
        return None  # Don't know what happened before, so return None
    return track.get_age(position)


def get_age_at_end_of_track(track: LinkingTrack) -> Optional[int]:
    """Gets the length of the cell cycle."""
    if len(track.get_previous_tracks()) != 1:
        return None  # Don't know what happened before, so return None
    return len(track) - 1
