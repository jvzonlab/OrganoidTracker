"""Small wrapper method to get the age of a position. The age is the time since appearance or (only for cells) the last
division.."""
import warnings
from typing import Optional

from organoid_tracker.core.links import Links, LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageTimings


def get_age(links: Links, position: Position) -> Optional[int]:
    """Gets how many time steps ago this cell was born"""
    track = links.get_track(position)
    if track is None:
        return None  # Not in a track, cannot calculate
    if len(track.get_previous_tracks()) != 1:
        return None  # Don't know what happened before, so return None
    return track.get_age(position)


def get_age_at_end_of_track(track: LinkingTrack) -> Optional[int]:
    """Gets the length of the cell cycle in time points."""
    warnings.warn("Use particle_age_finder.get_age_at_end_of_track_h(timings, track) instead", DeprecationWarning)
    if len(track.get_previous_tracks()) != 1:
        return None  # Don't know what happened before, so return None
    return len(track) - 1

def get_age_at_end_of_track_h(timings: ImageTimings, track: LinkingTrack) -> Optional[float]:
    """Gets the age of the cell at the end of the track in hours. The cell needs to have divided beforehand, as
    otherwise we don't know how old it was at the start of the track.

    At the first time point the cell appeared, it is assumed to be 1 time point old (instead of 0). So essentially,
    we assume the division to have happened moments after the last time point of the mother track.
    """
    if len(track.get_previous_tracks()) != 1:
        return None  # Don't know what happened before (either no track or a cell merge), so cannot calculate age
    time_at_start_h = timings.get_time_h_since_start(track.first_time_point_number() - 1)
    time_at_end_h = timings.get_time_h_since_start(track.last_time_point_number())
    return time_at_end_h - time_at_start_h

