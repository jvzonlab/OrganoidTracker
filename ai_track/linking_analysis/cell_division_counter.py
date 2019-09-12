"""For calculating how many times a cell has divided."""

from ai_track.core.links import Links
from ai_track.core.position import Position


def find_times_divided(links: Links, position: Position, first_time_point_number: int) -> int:
    """Calculates the number of times this position has divided in the past (since the first time point number). Returns
    -1 if we don't have the full lineage until that time point."""
    track = links.get_track(position)
    if track is None:
        return -1

    parent_tracks = track.get_previous_tracks()
    division_count = 0
    while len(parent_tracks) == 1:
        division_count += 1

        track = parent_tracks.pop()
        if track.min_time_point_number() > first_time_point_number:
            # Go back further in time
            parent_tracks = track.get_previous_tracks()
        else:
            parent_tracks = []

    if track.min_time_point_number() > first_time_point_number:
        # Could not look back far enough
        return -1

    return division_count

