"""Used to get a summary of what happens to a single cell lineage. How many divisions, deaths, ends and errors are
there?"""

from autotrack.core.links import PositionLinks
from autotrack.core.positions import Position
from autotrack.linking_analysis import linking_markers
from autotrack.linking_analysis.linking_markers import EndMarker


class LineageFate:
    """Calculates the number of occurences of certain events in the lineage."""

    divisions: int = 0  # How many divisions are there in the lineage?
    deaths: int = 0  # How many cell deaths are there in the lineage?
    errors: int = 0  # How many warnings are still in the lineage?
    ends: int = 0  # How many lineage ends (including cell deaths) are still in the lineage?


def get_lineage_fate(position: Position, links: PositionLinks, last_time_point_number: int) -> LineageFate:
    """Calculates the fate of the lineage. The last time point number is used to ignore lineage ends that occur in that
    time point."""
    lineage_fate = LineageFate()
    _get_sub_cell_fate(position, links, lineage_fate, last_time_point_number)
    return lineage_fate


def _get_sub_cell_fate(position: Position, links: PositionLinks, lineage_fate: LineageFate, last_time_point_number: int):
    while True:
        error = linking_markers.get_error_marker(links, position)
        if error is not None:
            lineage_fate.errors += 1

        next_positions = links.find_futures(position)
        if len(next_positions) > 1:
            lineage_fate.divisions += 1
            for next_position in next_positions:
                _get_sub_cell_fate(next_position, links, lineage_fate, last_time_point_number)
            return
        elif len(next_positions) == 0:
            if position.time_point_number() < last_time_point_number:
                lineage_fate.ends += 1  # Ignore lineage ends in the last time point
            if linking_markers.get_track_end_marker(links, position) == EndMarker.DEAD:
                lineage_fate.deaths += 1
            return
        else:
            position = next_positions.pop()
