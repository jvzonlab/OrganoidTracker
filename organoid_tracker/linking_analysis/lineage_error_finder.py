"""Used to find lineage trees with (potential) errors."""

from typing import List, Optional, Set, AbstractSet, Dict, Iterable

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import Links, LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.linking_analysis import linking_markers

# When at least one position has this position data marker, the error checker will only check in the lineages containing
# positions with this marker. This is useful for focusing the error checker on a specific part of the experiment.
ERROR_FOCUS_POINT_MARKER = "error_focus_point"

# This is set in experiment.global_data, to exclude all lineages with too few divisions from error correction
ERROR_FOCUS_MIN_DIVISIONS = "error_focus_min_divisions"


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
    lineages_with_errors = get_problematic_lineages(experiment, set())
    for lineage in lineages_with_errors:
        for track in lineage.start.find_all_descending_tracks(include_self=True):
            for position in track.positions():
                positions_to_remove.append(position)

    # Actually remove them
    for position in positions_to_remove:
        experiment.remove_position(position)


def find_error_focus_min_divisions(experiment: Experiment) -> int:
    """Gets the minimum number of divisions a lineage must have to be in focus for error correction. If no value is
    defined, it returns 0."""
    min_divisions_value = experiment.global_data.get_data(ERROR_FOCUS_MIN_DIVISIONS)
    if isinstance(min_divisions_value, int) or isinstance(min_divisions_value, float):
        return int(min_divisions_value)
    return 0


def find_error_focus_tracks(experiment: Experiment) -> Optional[Set[LinkingTrack]]:
    """Finds all tracks that are in focus for error correction. If no focus positions have been defined, returns None.

    Note that this method may also return an empty set instead of None, which happens when all defined focus points do
    not have a track. This is useful to signal that the track focus system is in use, but no tracks are in focus.
    """
    min_divisions = find_error_focus_min_divisions(experiment)
    if min_divisions > 0:
        # Find all tracks that have a division
        focus_tracks = set()
        for starting_track in experiment.links.find_starting_tracks():
            if starting_track.will_divide():
                for track in starting_track.find_all_descending_tracks(include_self=True):
                    focus_tracks.add(track)
        return focus_tracks

    links = experiment.links
    focus_tracks = set()
    has_focus_points = False
    for focus_point, value in experiment.positions.find_all_positions_with_data(ERROR_FOCUS_POINT_MARKER):
        if value <= 0:
            continue

        has_focus_points = True  # We have at least one focus point (it may not have a track, but we still register it)

        track = links.get_track(focus_point)
        if track is not None:
            for related_track in track.find_all_tracks_in_same_lineage(include_self=True):
                focus_tracks.add(related_track)
    if not has_focus_points:
        return None  # This signals that the track focus system is not in use
    return focus_tracks


def get_problematic_lineages(experiment: Experiment, crumbs: AbstractSet[Position],
                             min_time_point=None,
                             max_time_point=None,
                             excluded_errors=None) -> List[LineageWithErrors]:
    """Gets a list of all lineages with warnings in the experiment. The provided "crumbs" are placed in the right
    lineages, so that you can see to what lineages those cells belong."""
    positions_with_errors = linking_markers.find_errored_positions(experiment.positions)
    track_to_errors = _group_by_track(experiment.links, positions_with_errors)
    track_to_crumbs = _group_by_track(experiment.links, crumbs)
    tracks_to_focus_on = find_error_focus_tracks(experiment)

    lineages_with_errors = list()
    for starting_track in experiment.links.find_starting_tracks():
        lineage = LineageWithErrors(starting_track)

        for track in starting_track.find_all_descending_tracks(include_self=True):
            if tracks_to_focus_on is not None and track not in tracks_to_focus_on:
                continue

            lineage._add_errors(track_to_errors.get(track))
            lineage._add_crumbs(track_to_crumbs.get(track))

        if len(lineage.errored_positions) > 0:
            lineages_with_errors.append(lineage)

    return lineages_with_errors


def _find_errors_in_lineage(links: Links, positions: PositionCollection, lineage: LineageWithErrors, position: Position,
                            crumbs: AbstractSet[Position]):
    while True:
        if position in crumbs:
            lineage.crumbs.add(position)

        error = linking_markers.get_error_marker(positions, position)
        if error is not None:
            lineage.errored_positions.append(position)
        future_positions = links.find_futures(position)

        if len(future_positions) > 1:
            # Branch out
            for future_position in future_positions:
                _find_errors_in_lineage(links, positions, lineage, future_position, crumbs)
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
