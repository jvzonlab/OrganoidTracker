"""Extra markers used to describe the linking data. For example, you can mark the end of a lineage as a cell death."""
import warnings
from enum import Enum
from typing import Optional, Iterable

from organoid_tracker.core.links import Links
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.linking_analysis.errors import Error
from organoid_tracker.position_analysis.position_markers import UNCERTAIN_MARKER


class EndMarker(Enum):
    DEAD = 1
    OUT_OF_VIEW = 2
    SHED = 3
    SHED_OUTSIDE = 4
    STIMULATED_SHED = 5

    @staticmethod
    def is_shed(marker: Optional["EndMarker"]) -> bool:
        return marker == EndMarker.SHED or marker == EndMarker.SHED_OUTSIDE or marker == EndMarker.STIMULATED_SHED

    def get_display_name(self):
        """Gets a user-friendly display name."""
        return self.name.lower().replace("_", " ")


class StartMarker(Enum):
    GOES_INTO_VIEW = 1
    UNSURE = 2

    def get_display_name(self):
        """Gets a user-friendly display name."""
        return self.name.lower().replace("_", " ")


def get_track_end_marker(positions: PositionCollection, position: Position) -> Optional[EndMarker]:
    """Gets a death marker, which provides a reason why the cell lineage ended."""
    ending_str = positions.get_position_data(position, "ending")
    if ending_str is None:
        return None

    return EndMarker[ending_str.upper()]


def is_live(positions: PositionCollection, position: Position) -> bool:
    """Returns true if the position is a live cell, i.e. it does not have a shed or death marker."""
    end_marker = get_track_end_marker(positions, position)
    return end_marker != EndMarker.DEAD and end_marker != EndMarker.SHED and end_marker != EndMarker.SHED_OUTSIDE and end_marker != EndMarker.STIMULATED_SHED


def set_track_end_marker(positions: PositionCollection, position: Position, end_marker: Optional[EndMarker]):
    """Sets a reason why the track ended at the given point."""
    if end_marker is None:
        positions.set_position_data(position, "ending", None)
    else:
        positions.set_position_data(position, "ending", end_marker.name.lower())


def find_death_and_shed_positions(links: Links, positions: PositionCollection) -> Iterable[Position]:
    """Gets all positions that were marked as a cell death or a cell shedding event."""
    death_marker = EndMarker.DEAD.name.lower()
    shed_marker = EndMarker.SHED.name.lower()
    shed_outside_marker = EndMarker.SHED_OUTSIDE.name.lower()
    stimulated_shed_marker = EndMarker.STIMULATED_SHED.name.lower()
    for position, ending_marker in positions.find_all_positions_with_data("ending"):
        if len(links.find_futures(position)) > 0:
            continue  # Not actually ending, ending marker is useless

        if ending_marker == death_marker or ending_marker == shed_marker or ending_marker == shed_outside_marker or ending_marker == stimulated_shed_marker:
            yield position


def find_shed_positions(links: Links, positions: PositionCollection) -> Iterable[Position]:
    """Gets all positions that were marked as a cell shedding event."""
    shed_marker = EndMarker.SHED.name.lower()
    shed_outside_marker = EndMarker.SHED_OUTSIDE.name.lower()
    for position, ending_marker in positions.find_all_positions_with_data("ending"):
        if len(links.find_futures(position)) > 0:
            continue  # Not actually ending, ending marker is useless

        if ending_marker == shed_marker or ending_marker == shed_outside_marker:
            yield position


def find_death_positions(links: Links, positions: PositionCollection) -> Iterable[Position]:
    """Gets all positions that were marked as a cell death event."""
    dead_marker = EndMarker.DEAD.name.lower()
    for position, ending_marker in positions.find_all_positions_with_data("ending"):
        if len(links.find_futures(position)) > 0:
            continue  # Not actually ending, ending marker is useless

        if ending_marker == dead_marker:
            yield position

def find_stimulated_shed_positions(links: Links, positions: PositionCollection) -> Iterable[Position]:
    """Gets all positions that were marked as a cell death event."""
    stimulated_shed_marker = EndMarker.STIMULATED_SHED.name.lower()
    for position, ending_marker in positions.find_all_positions_with_data("ending"):
        if len(links.find_futures(position)) > 0:
            continue  # Not actually ending, ending marker is useless

        if ending_marker == stimulated_shed_marker:
            yield position


def get_track_start_marker(positions: PositionCollection, position: Position) -> Optional[StartMarker]:
    """Gets the appearance marker. This is used to explain why a cell appeared out of thin air."""
    starting_str = positions.get_position_data(position, "starting")
    if starting_str is None:
        return None

    return StartMarker[starting_str.upper()]


def set_track_start_marker(positions: PositionCollection, position: Position, start_marker: Optional[StartMarker]):
    """Sets a reason why the track ended at the given point."""
    if start_marker is None:
        positions.set_position_data(position, "starting", None)
    else:
        positions.set_position_data(position, "starting", start_marker.name.lower())


def find_errored_positions(positions: PositionCollection) -> Iterable[Position]:
    """Gets all positions that have a (non-suppressed) error in the given time point range."""
    with_error_marker = positions.find_all_positions_with_data("error")
    for position, error_number in with_error_marker:
        if positions.get_position_data(position, "suppressed_error") == error_number:
            continue  # Error was suppressed

        yield position


def get_error_marker(positions: PositionCollection, position: Position) -> Optional[Error]:
    """Gets the error marker for the given link, if any. Returns None if the error has been suppressed using
    suppress_error_marker."""
    error_number = positions.get_position_data(position, "error")
    if error_number is None:
        return None

    if positions.get_position_data(position, "suppressed_error") == error_number:
        return None  # Error was suppressed
    return Error(error_number)


def suppress_error_marker(positions: PositionCollection, position: Position, error: Error):
    """Suppresses an error. Even if set_error_marker is called afterwards, the error will not show up in
    get_error_marker."""
    positions.set_position_data(position, "suppressed_error", error.value)


def is_error_suppressed(positions: PositionCollection, position: Position, error: Error) -> bool:
    """Returns True if the given error is suppressed. If another type of error is suppressed, this method returns
    False."""
    return positions.get_position_data(position, "suppressed_error") == error.value


def unsuppress_error_marker(positions: PositionCollection, position: Position):
    """Unsuppresses an error. If the error was not suppressed, or there was no error, this method does nothing."""
    positions.set_position_data(position, "suppressed_error", None)


def set_error_marker(positions: PositionCollection, position: Position, error: Optional[Error]):
    """Sets an error marker for the given position."""
    if error is None:
        positions.set_position_data(position, "error", None)
    else:
        positions.set_position_data(position, "error", error.value)


def remove_error_marker(positions: PositionCollection, position: Position, error: Error):
    """Removes an error marker if (and only if) it's currently set to the given error.

    See also:
        - set_error_marker(positions, position, None) - removes any error marker, independent of the current one
        - suppress_error_marker(positions, position, error) - makes this position immune of the given type of error,
          even if set_error_marker is called, it will not show up in get_error_marker. Normally used by the end-user
          to suppress false positives.
    """
    error_number = positions.get_position_data(position, "error")
    if error_number is None or error_number != error.value:
        return  # Error is not set to the one we want removed

    positions.set_position_data(position, "error", None)
    positions.set_position_data(position, "suppressed_error", None)


def get_position_type(positions: PositionCollection, position: Position) -> Optional[str]:
    """Deprecated - use position_markers.get_position_type"""
    warnings.warn("Use position_markers.get_position_type instead of linking_markers.get_position_type", DeprecationWarning)
    position_type = positions.get_position_data(position, "type")
    if position_type is None:
        return None
    return position_type.upper()


def has_shapes(positions: PositionCollection) -> bool:
    """Gets whether there is shape information stored for at least one position in the experiment."""
    return positions.has_position_data_with_name("shape")


def set_mother_score(positions: PositionCollection, position: Position, value: float):
    """Sets the mother score to the given value."""
    if value == 0:
        positions.set_position_data(position, "mother_score", None)
    else:
        positions.set_position_data(position, "mother_score", value)


def get_mother_score(positions: PositionCollection, position: Position) -> float:
    """Gets the mother score of the given position, or 0 if absent."""
    value = positions.get_position_data(position, "mother_score")
    if value is not None:
        return float(value)
    return 0


def has_mother_scores(positions: PositionCollection) -> bool:
    """Gets if there is at least a single mother score recorded."""
    return positions.has_position_data_with_name("mother_score")


def is_uncertain(positions: PositionCollection, position: Position) -> bool:
    """Returns True if the given position is marked as uncertain. A person can mark a position as uncertain if it is not
    clear whether there is actually a position marker there. The error checker will warm for such cases."""
    uncertain = positions.get_position_data(position, UNCERTAIN_MARKER)
    if uncertain is None:
        return False
    return uncertain


def set_uncertain(positions: PositionCollection, position: Position, uncertain: bool):
    """Marks the given position as certain/uncertain. See is_uncertain for more info."""
    if uncertain:
        positions.set_position_data(position, UNCERTAIN_MARKER, True)
    else:
        positions.set_position_data(position, UNCERTAIN_MARKER, None)  # This removes the marker
