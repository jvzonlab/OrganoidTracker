from typing import List, Tuple

from ai_track.core import TimePoint, Color
from ai_track.core.links import LinkingTrack, Links
from ai_track.linking_analysis import lineage_id_creator


class LineageTree:
    """Represents a lineage tree, starting from a single cell."""

    _starting_track: LinkingTrack
    _plotting_size: int

    def __init__(self, starting_track: LinkingTrack):
        self._starting_track = starting_track
        self._plotting_size = self._count_size(starting_track)

    @property
    def starting_track(self) -> LinkingTrack:
        """Gets the first track that starts this lineage."""
        return self._starting_track

    @property
    def plotting_size(self) -> int:
        """Gets the total width the lineage tree will end up with."""
        return self._plotting_size

    def _count_size(self, track: LinkingTrack) -> int:
        count = 1
        for next_track in track.get_next_tracks():
            count += self._count_size(next_track)
        return count

    def get_tracks_at_time_point(self, time_point: TimePoint) -> List[LinkingTrack]:
        """Gets a list of all tracks in this lineage tree that reach the given time point."""
        return self.get_tracks_at_time_point_number(time_point.time_point_number())

    def get_tracks_at_time_point_number(self, time_point_number: int) -> List[LinkingTrack]:
        """Gets a list of all tracks in this lineage tree that reach the given time point."""
        if time_point_number < self._starting_track.min_time_point_number():
            return []

        the_list = []
        self._add_to_list(self._starting_track, time_point_number, the_list)
        return the_list

    def _add_to_list(self, track: LinkingTrack, time_point_number: int, the_list: List[LinkingTrack]):
        """Recursive function that adds all (daughter) tracks that cross the given time point to the given list. This
        method assumes that the given time point is during or after the given track."""
        if time_point_number > track.max_time_point_number():
            for next_track in track.get_next_tracks():
                self._add_to_list(next_track, time_point_number, the_list)
        else:
            the_list.append(track)


