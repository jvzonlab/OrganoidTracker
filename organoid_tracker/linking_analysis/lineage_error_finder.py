"""Used to find lineage trees with (potential) errors."""

from typing import List, Optional, Set, AbstractSet, Dict, Iterable

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import Links, LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.gui.window import DisplaySettings
from organoid_tracker.linking_analysis import linking_markers


class LineageWithErrors:
    """Represents all (potential) errors in a  complete lineage tree."""

    start: LinkingTrack  # Start of the lineage tree
    errored_positions: List[Position]  # All positions with (potential) errors
    crumbs: Set[Position]  # Some of the positions in the lineage tree.

    def __init__(self, start: LinkingTrack):
        self.start = start
        self.errored_positions = []
        self.crumbs = set()

    def _add_errors(self, errors: Optional[Iterable[Position]]):
        if errors is not None:
            self.errored_positions += errors

    def _add_crumbs(self, crumbs: Optional[Iterable[Position]]):
        if crumbs is not None:
            self.crumbs |= set(crumbs)


def _group_by_track(links: Links, positions: Iterable[Position]) -> Dict[LinkingTrack, List[Position]]:
    track_to_positions = dict()
    for position in positions:
        track = links.get_track(position)
        if track in track_to_positions:
            track_to_positions[track].append(position)
        else:
            track_to_positions[track] = [position]
    return track_to_positions


def delete_problematic_lineages(experiment: Experiment):
    """This deletes all positions in a lineage with errors. What remains should be a clean experiment with just the
    corrected data."""

    # We cannot remove positions during iteration, so remember the positions to remove in a list
    positions_to_remove = list()

    # Find all positions to remove
    lineages_with_errors = get_problematic_lineages(experiment.links, experiment.position_data, set())
    for lineage in lineages_with_errors:
        for track in lineage.start.find_all_descending_tracks(include_self=True):
            for position in track.positions():
                positions_to_remove.append(position)

    # Actually remove them
    for position in positions_to_remove:
        experiment.remove_position(position)


def get_problematic_lineages(links: Links, position_data: PositionData, crumbs: AbstractSet[Position],
                             *, min_time_point: Optional[TimePoint] = None,
                             max_time_point: Optional[TimePoint] = None, min_divisions: int = 0
                             ) -> List[LineageWithErrors]:
    """Gets a list of all lineages with warnings in the experiment. The provided "crumbs" are placed in the right
    lineages, so that you can see to what lineages those cells belong."""
    positions_with_errors = linking_markers.find_errored_positions(position_data, min_time_point=min_time_point,
                                                                   max_time_point=max_time_point)
    track_to_errors = _group_by_track(links, positions_with_errors)
    track_to_crumbs = _group_by_track(links, crumbs)

    lineages_with_errors = list()
    for starting_track in links.find_starting_tracks():
        divisions_in_lineage = 0
        lineage = LineageWithErrors(starting_track)

        for track in starting_track.find_all_descending_tracks(include_self=True):
            if track.will_divide():
                divisions_in_lineage += 1

            lineage._add_errors(track_to_errors.get(track))
            lineage._add_crumbs(track_to_crumbs.get(track))

        if len(lineage.errored_positions) > 0 and divisions_in_lineage >= min_divisions:
            lineages_with_errors.append(lineage)

    return lineages_with_errors


def _find_errors_in_lineage(links: Links, position_data: PositionData, lineage: LineageWithErrors, position: Position, crumbs: AbstractSet[Position]):
    while True:
        if position in crumbs:
            lineage.crumbs.add(position)

        error = linking_markers.get_error_marker(position_data, position)
        if error is not None:
            lineage.errored_positions.append(position)
        future_positions = links.find_futures(position)

        if len(future_positions) > 1:
            # Branch out
            for future_position in future_positions:
                _find_errors_in_lineage(links, position_data, lineage, future_position, crumbs)
            return
        if len(future_positions) < 1:
            # Stop
            return
        # Continue
        position = future_positions.pop()


def find_lineage_index_with_crumb(lineages: List[LineageWithErrors], crumb: Optional[Position]) -> Optional[int]:
    """Attempts to find the given position in the lineages. Returns None if the position is None or if the position is
    in none of the lineages."""
    if crumb is None:
        return None
    for index, lineage in enumerate(lineages):
        if crumb in lineage.crumbs:
            return index
    return None
