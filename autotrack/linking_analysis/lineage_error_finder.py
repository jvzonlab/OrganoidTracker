"""Used to find lineage trees with (potential) errors."""

from typing import List, Optional, Set, AbstractSet, Dict, Iterable

from autotrack.core.links import PositionLinks, LinkingTrack
from autotrack.core.positions import Position
from autotrack.linking_analysis import linking_markers


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


def _group_by_track(links: PositionLinks, positions: Iterable[Position]) -> Dict[LinkingTrack, List[Position]]:
    track_to_positions = dict()
    for position in positions:
        track = links.get_track(position)
        if track in track_to_positions:
            track_to_positions[track].append(position)
        else:
            track_to_positions[track] = [position]
    return track_to_positions


def get_problematic_lineages(links: PositionLinks, crumbs: AbstractSet[Position]) -> List[LineageWithErrors]:
    """Gets a list of all lineages with warnings in the experiment. The provided "crumbs" are placed in the right
    lineages, so that you can see to what lineages those cells belong."""
    positions_with_errors = linking_markers.find_errored_positions(links)
    track_to_errors = _group_by_track(links, positions_with_errors)
    track_to_crumbs = _group_by_track(links, crumbs)

    lineages_with_errors = list()
    for track in links.find_starting_tracks():
        lineage = LineageWithErrors(track)
        lineage._add_errors(track_to_errors.get(track))
        lineage._add_crumbs(track_to_crumbs.get(track))

        for next_track in track.find_all_descending_tracks():
            lineage._add_errors(track_to_errors.get(next_track))
            lineage._add_crumbs(track_to_crumbs.get(next_track))

        if len(lineage.errored_positions) > 0:
            lineages_with_errors.append(lineage)

    return lineages_with_errors


def _find_errors_in_lineage(links: PositionLinks, lineage: LineageWithErrors, position: Position, crumbs: AbstractSet[Position]):
    while True:
        if position in crumbs:
            lineage.crumbs.add(position)

        error = linking_markers.get_error_marker(links, position)
        if error is not None:
            lineage.errored_positions.append(position)
        future_positions = links.find_futures(position)

        if len(future_positions) > 1:
            # Branch out
            for future_position in future_positions:
                _find_errors_in_lineage(links, lineage, future_position, crumbs)
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