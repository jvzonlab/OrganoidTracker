"""Used to get a summary of what happens to a single cell lineage. How many divisions, deaths, ends and errors are
there?"""

from organoid_tracker.core.links import Links, LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.linking_analysis import linking_markers
from organoid_tracker.linking_analysis.linking_markers import EndMarker


class LineageFate:
    """Calculates the number of occurences of certain events in the lineage."""

    tracks: int = 0  # How many tracks this lineage tree consists of. 0 if the position has no links.
    divisions: int = 0  # How many divisions are there in the lineage?
    deaths: int = 0  # How many cell deaths are there in the lineage?
    sheds: int = 0  # How many cell shedding events are there in the lineage?
    ends: int = 0  # How many lineage ends (including cell deaths and sheddings) are still in the lineage?


def get_lineage_fate(position: Position, links: Links, position_data: PositionData, last_time_point_number: int) -> LineageFate:
    """Calculates the fate of the lineage. The last time point number is used to ignore lineage ends that occur in that
    time point."""
    lineage_fate = LineageFate()
    starting_track = links.get_track(position)
    if starting_track is not None:
        _get_sub_cell_fate(starting_track, position_data, lineage_fate, last_time_point_number)
    return lineage_fate


def _get_sub_cell_fate(track: LinkingTrack, position_data: PositionData, lineage_fate: LineageFate,
                       last_time_point_number: int):
    lineage_fate.tracks += 1
    next_tracks = track.get_next_tracks()
    if len(next_tracks) > 1:
        lineage_fate.divisions += 1
        for next_track in next_tracks:
            _get_sub_cell_fate(next_track, position_data, lineage_fate, last_time_point_number)
    elif len(next_tracks) == 0:
        if track.last_time_point_number() < last_time_point_number:
            lineage_fate.ends += 1  # Ignores lineage ends in the last time point
        end_marker = linking_markers.get_track_end_marker(position_data, track.find_last_position())
        if end_marker == EndMarker.DEAD:
            lineage_fate.deaths += 1
        if EndMarker.is_shed(end_marker):
            lineage_fate.sheds += 1
