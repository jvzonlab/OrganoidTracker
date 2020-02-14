"""Extra markers used to describe the linking data. For example, you can mark the end of a lineage as a cell death."""

from enum import Enum
from typing import Optional, Iterable, Dict, Set, Union

from ai_track.core import links, TimePoint
from ai_track.core.links import Links
from ai_track.core.position import Position
from ai_track.core.position_data import PositionData
from ai_track.linking_analysis.errors import Error


class EndMarker(Enum):
    DEAD = 1
    OUT_OF_VIEW = 2
    SHED = 3

    def get_display_name(self):
        """Gets a user-friendly display name."""
        return self.name.lower().replace("_", " ")


class StartMarker(Enum):
    GOES_INTO_VIEW = 1
    UNSURE = 2

    def get_display_name(self):
        """Gets a user-friendly display name."""
        return self.name.lower().replace("_", " ")


def get_track_end_marker(links: Links, position: Position) -> Optional[EndMarker]:
    """Gets a death marker, which provides a reason why the cell lineage ended."""
    ending_str = links.get_position_data(position, "ending")
    if ending_str is None:
        return None

    return EndMarker[ending_str.upper()]


def is_live(links: Links, position: Position) -> bool:
    """Returns true if the position is a live cell, i.e. it does not have a shed or death marker."""
    end_marker = get_track_end_marker(links, position)
    return end_marker != EndMarker.DEAD and end_marker != EndMarker.SHED


def set_track_end_marker(links: Links, position: Position, end_marker: Optional[EndMarker]):
    """Sets a reason why the track ended at the given point."""
    if end_marker is None:
        links.set_position_data(position, "ending", None)
    else:
        links.set_position_data(position, "ending", end_marker.name.lower())


def find_death_and_shed_positions(links: Links) -> Iterable[Position]:
    """Gets all positions that were marked as a cell death or a cell shedding event."""
    death_marker = EndMarker.DEAD.name.lower()
    shed_marker = EndMarker.SHED.name.lower()
    for position, ending_marker in links.find_all_positions_with_data("ending"):
        if len(links.find_futures(position)) > 0:
            continue  # Not actually ending, ending marker is useless

        if ending_marker == death_marker or ending_marker == shed_marker:
            yield position


def find_shed_positions(links: Links) -> Iterable[Position]:
    """Gets all positions that were marked as a cell shedding event."""
    shed_marker = EndMarker.SHED.name.lower()
    for position, ending_marker in links.find_all_positions_with_data("ending"):
        if len(links.find_futures(position)) > 0:
            continue  # Not actually ending, ending marker is useless

        if ending_marker == shed_marker:
            yield position


def find_death_positions(links: Links) -> Iterable[Position]:
    """Gets all positions that were marked as a cell death event."""
    dead_marker = EndMarker.DEAD.name.lower()
    for position, ending_marker in links.find_all_positions_with_data("ending"):
        if len(links.find_futures(position)) > 0:
            continue  # Not actually ending, ending marker is useless

        if ending_marker == dead_marker:
            yield position


def get_track_start_marker(links: Links, position: Position) -> Optional[StartMarker]:
    """Gets the appearance marker. This is used to explain why a cell appeared out of thin air."""
    starting_str = links.get_position_data(position, "starting")
    if starting_str is None:
        return None

    return StartMarker[starting_str.upper()]


def set_track_start_marker(links: Links, position: Position, start_marker: Optional[StartMarker]):
    """Sets a reason why the track ended at the given point."""
    if start_marker is None:
        links.set_position_data(position, "starting", None)
    else:
        links.set_position_data(position, "starting", start_marker.name.lower())


def find_errored_positions(links: Links, *, min_time_point: Optional[TimePoint] = None,
                           max_time_point: Optional[TimePoint] = None) -> Iterable[Position]:
    """Gets all positions that have a (non suppressed) error in the given time point range."""
    min_time_point_number = min_time_point.time_point_number() if min_time_point is not None else float("-inf")
    max_time_point_number = max_time_point.time_point_number() if max_time_point is not None else float("inf")

    with_error_marker = links.find_all_positions_with_data("error")
    for position, error_number in with_error_marker:
        if links.get_position_data(position, "suppressed_error") == error_number:
            continue  # Error was suppressed

        if min_time_point_number <= position.time_point_number() <= max_time_point_number:
            yield position


def get_error_marker(links: Links, position: Position) -> Optional[Error]:
    """Gets the error marker for the given link, if any. Returns None if the error has been suppressed using
    suppress_error_marker."""
    error_number = links.get_position_data(position, "error")
    if error_number is None:
        return None

    if links.get_position_data(position, "suppressed_error") == error_number:
        return None  # Error was suppressed
    return Error(error_number)


def suppress_error_marker(links: Links, position: Position, error: Error):
    """Suppresses an error. Even if set_error_marker is called afterwards, the error will not show up in
    get_error_marker."""
    links.set_position_data(position, "suppressed_error", error.value)


def is_error_suppressed(links: Links, position: Position, error: Error) -> bool:
    """Returns True if the given error is suppressed. If another type of error is suppressed, this method returns
    False."""
    return links.get_position_data(position, "suppressed_error") == error.value


def set_error_marker(links: Links, position: Position, error: Optional[Error]):
    """Sets an error marker for the given position."""
    if error is None:
        links.set_position_data(position, "error", None)
    else:
        links.set_position_data(position, "error", error.value)


def get_position_type(links: Links, position: Position) -> Optional[str]:
    """Gets the type of the cell in UPPERCASE, interpreted as the intestinal organoid cell type."""
    type = links.get_position_data(position, "type")
    if type is None:
        return None
    return type.upper()


def set_position_type(links: Links, position: Position, type: Optional[str]):
    """Sets the type of the cell. Set to None to delete the cell type."""
    type_str = type.upper() if type is not None else None
    links.set_position_data(position, "type", type_str)


def get_position_types(links: Links, positions: Set[Position]) -> Dict[Position, Optional[str]]:
    """Gets all known cell types of the given positions, with the names in UPPERCASE."""
    types = dict()
    for position in positions:
        types[position] = get_position_type(links, position)
    return types


def set_mother_score(position_data: PositionData, position: Position, value: float):
    """Sets the mother score to the given value."""
    if value == 0:
        position_data.set_position_data(position, "mother_score", None)
    else:
        position_data.set_position_data(position, "mother_score", value)


def get_mother_score(position_data: Union[Links, PositionData], position: Position) -> float:
    """Gets the mother score of the given position, or 0 if absent."""
    value = position_data.get_position_data(position, "mother_score")
    if value is not None:
        return float(value)
    return 0


def has_mother_scores(position_data: PositionData) -> bool:
    """Gets if there is at least a single mother score recorded."""
    return position_data.has_position_data_with_name("mother_score")


def is_uncertain(position_data: PositionData, position: Position) -> bool:
    """Returns True if the given position is marked as uncertain. A person can mark a position as uncertain if it is not
    clear whether there is actually a position marker there. The error checker will warm for such cases."""
    uncertain = position_data.get_position_data(position, "uncertain")
    if uncertain is None:
        return False
    return uncertain


def set_uncertain(position_data: PositionData, position: Position, uncertain: bool):
    """Marks the given position as certain/uncertain. See is_uncertain for more info."""
    if uncertain:
        position_data.set_position_data(position, "uncertain", True)
    else:
        position_data.set_position_data(position, "uncertain", None)  # This removes the marker
