from typing import Iterable, List

from autotrack.core import TimePoint
from autotrack.core.links import Links, LinkingTrack
from autotrack.core.position import Position


def find_future_positions_at(links: Links, position: Position, time_point: TimePoint) -> List[Position]:
    """Finds the positions the given particle will have in the given time point, which must be in the future. (Or in the
    present, but then you'll get the original position back.) Returns an empty array if time_point is in the past.
    Returns multiple positions if the position is a cell and it divided."""
    if position.time_point_number() == time_point.time_point_number():
        return [position]
    if position.time_point_number() > time_point.time_point_number():
        return []

    track = links.get_track(position)
    if track is None:
        return []

    return list(_find_future_position_in_track(track, time_point))


def _find_future_position_in_track(track: LinkingTrack, time_point: TimePoint) -> Iterable[Position]:
    """Gets the position from a given track, or one of the next tracks."""
    if time_point.time_point_number() <= track.max_time_point_number():
        yield track.get_by_time_point(time_point.time_point_number())
    else:
        for next_track in track.get_next_tracks():
            yield from _find_future_position_in_track(next_track, time_point)
