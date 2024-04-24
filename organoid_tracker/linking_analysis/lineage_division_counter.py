"""For counting how many times a cell (and its offspring) will divide."""

from typing import Optional

from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.linking_analysis import linking_markers
from organoid_tracker.linking_analysis.linking_markers import EndMarker


def get_division_count_in_lineage(starting_track: LinkingTrack, position_data: PositionData,
                                  last_time_point_number: int) -> Optional[int]:
    """Gets how many divisions there are in the lineage starting at the given cell. If the cell does not divide, then
    this method will return 0. If a lineage ended before the end of the experiment (as defined by
    last_time_point_number) and it was not because of an actual death, then it is assumed that the cell went out of
    view, and that no reliable clonal size can be calculated. In that case, this function returns None.

    Any divisions occurring after last_time_point_number are ignored. This is useful if you want to look at only a
    limited time point window.

    If a single cell goes out of the view, we don't know if it would have divided. If this happens, and require_accurate
    is set to True, then this method returns None. Otherwise, it is assumed the cell doesn't divide.
    """
    division_count = 0
    for track in starting_track.find_all_descending_tracks(include_self=True):
        if track.first_time_point_number() > last_time_point_number:
            # Ignore this track, it is past the end of the time point window
            continue
        if not track.get_next_tracks() \
                and linking_markers.get_track_end_marker(position_data, track.find_last_position()) != EndMarker.DEAD\
                and track.last_time_point_number() < last_time_point_number:
            return None  # Don't know why this track ended, division count in lineage is uncertain
        if track.last_time_point_number() < last_time_point_number and len(track.get_next_tracks()) > 1:
            division_count += 1
    return division_count


def get_min_division_count_in_lineage(starting_track: LinkingTrack, last_time_point_number: int = 999999999):
    """Gets the number of divisions we could see up until last_time_point_number. This method returns a result even
    for lineages that are not fully tracked. In that case, the actual number of divisions may be higher, as we might be
    ignoring divisions that were happening outside the view."""
    division_count = 0
    for track in starting_track.find_all_descending_tracks(include_self=True):
        if track.last_time_point_number() < last_time_point_number and len(track.get_next_tracks()) > 1:
            division_count += 1
    return division_count

def get_number_of_cells_at_end(starting_track: LinkingTrack, last_time_point_number: int = 999999999):
    """Gets the number of divisions we could see up until last_time_point_number. This method returns a result even
    for lineages that are not fully tracked. In that case, the actual number of divisions may be higher, as we might be
    ignoring divisions that were happening outside the view."""
    count = 0
    for track in starting_track.find_all_descending_tracks(include_self=True):
        if (track.max_time_point_number() == last_time_point_number):
           count += 1
    return count



