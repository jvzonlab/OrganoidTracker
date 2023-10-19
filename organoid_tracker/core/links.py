from pprint import pprint
from typing import Optional, Dict, Iterable, List, Set, Union, Tuple, Any, ItemsView, Callable

from organoid_tracker.core import TimePoint, Color
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.typing import DataType


class LinkingTrack:
    _min_time_point_number: int  # Equal to _positions_by_time_point[0].time_point_number()

    # Positions by time point. Position 0 contains the position at min_time_point, position 1 min_time_point + 1, etc.
    _positions_by_time_point: List[Position]

    # Links to other tracks that follow or precede this track
    _next_tracks: List["LinkingTrack"]
    _previous_tracks: List["LinkingTrack"]

    _lineage_data: Dict[str, DataType]  # Only has contents if there are no previous tracks

    def __init__(self, positions_by_time_point: List[Position]):
        self._min_time_point_number = positions_by_time_point[0].time_point_number()
        self._positions_by_time_point = positions_by_time_point
        self._next_tracks = list()
        self._previous_tracks = list()
        self._lineage_data = dict()

    def find_position_at_time_point_number(self, time_point_number: int) -> Position:
        if time_point_number < self._min_time_point_number \
                or time_point_number >= self._min_time_point_number + len(self._positions_by_time_point):
            raise IndexError(f"Time point {time_point_number} outside track from {self._min_time_point_number} to"
                             f" {self.max_time_point_number()}")
        return self._positions_by_time_point[time_point_number - self._min_time_point_number]

    def _find_pasts(self, time_point_number: int) -> Set[Position]:
        """Returns all positions directly linked to the position at the given time point."""
        search_index = (time_point_number - 1) - self._min_time_point_number # -1 is to look one time point in the past
        if search_index >= 0:
            return {self._positions_by_time_point[search_index]}

        # We ended up at the first time point of this track, continue search in previous tracks
        return {track.find_last_position() for track in self._previous_tracks}

    def _find_futures(self, time_point_number: int) -> Set[Position]:
        """Returns all positions directly linked to the position at the given time point."""
        search_index = (time_point_number + 1) - self._min_time_point_number
        if search_index < len(self._positions_by_time_point):
            return {self._positions_by_time_point[search_index]}

        # We ended up at the last time point of this track, continue search in next tracks
        return {track.find_first_position() for track in self._next_tracks}

    def find_first_position(self) -> Position:
        """Returns the first position in this track."""
        return self._positions_by_time_point[0]

    def find_last_position(self) -> Position:
        """Returns the last position in this track."""
        return self._positions_by_time_point[-1]

    def find_all_descending_tracks(self, include_self: bool = False) -> Iterable["LinkingTrack"]:
        """Iterates over all tracks that will follow this one, and the one after that, etc."""
        if include_self:
            yield self
        for next_track in self._next_tracks:
            yield next_track
            yield from next_track.find_all_descending_tracks()

    def find_all_previous_tracks(self, include_self: bool = False) -> Iterable["LinkingTrack"]:
        """Iterates over all tracks that precede this one, and the one before that, etc."""
        if include_self:
            yield self
        for previous_track in self._previous_tracks:
            yield previous_track
            yield from previous_track.find_all_previous_tracks()

    def positions(self, connect_to_previous_track: bool = False) -> Iterable[Position]:
        """Returns all positions in this track, in order.

        If connect_to_previous_track is True, then it also returns the last position of the previous track, if that exists. This is useful if you
        are drawing lines in between positions."""
        if connect_to_previous_track:
            if len(self._previous_tracks) >= 1:
                yield next(iter(self._previous_tracks)).find_last_position()

        yield from self._positions_by_time_point

    def _update_link_to_previous(self, was: "LinkingTrack", will_be: "LinkingTrack"):
        """Replaces a value in the _previous_tracks list. Make sure that old track is in the list."""
        self._previous_tracks.remove(was)
        self._previous_tracks.append(will_be)

    def _update_link_to_next(self, was: "LinkingTrack", will_be: "LinkingTrack"):
        """Replaces a value in the _next_tracks list. Make sure that old track is in the list."""
        self._next_tracks.remove(was)
        self._next_tracks.append(will_be)

    def max_time_point_number(self) -> int:
        """Gets the highest time point number where this track still contains a position ."""
        return self._min_time_point_number + len(self._positions_by_time_point) - 1

    def get_age(self, position: Position) -> int:
        """Gets the age of this position. This will be 0 if its the first track position, 1 on the position after that,
        etcetera."""
        return position.time_point_number() - self._min_time_point_number

    def __repr__(self):
        return f"<LinkingTrack t={self._min_time_point_number}-{self.max_time_point_number()}>"

    def get_next_tracks(self) -> Set["LinkingTrack"]:
        """Gets a set of the tracks that will directly follow this track. If empty, the lineage end. If the length is 2,
        it is a cell division. Lengths of 1 will never occur. Lengths of 3 can occur, but make no biological sense.

        See find_all_descending_tracks if you're also interested in the tracks that come after that."""
        return set(self._next_tracks)

    def get_previous_tracks(self) -> Set["LinkingTrack"]:
        """Gets a set of the tracks before this track. If empty, the lineage started. Normally, the set will have a
        size of 1. Larger sizes indicate a cell merge, which makes no biological sense."""
        return set(self._previous_tracks)

    def min_time_point_number(self) -> int:
        """Gets the first time point number of this track."""
        return self._min_time_point_number

    def __len__(self):
        """Gets the time length of the track, in number of time points."""
        return len(self._positions_by_time_point)

    def __hash__(self) -> int:
        return hash(self._positions_by_time_point[0])

    def __eq__(self, other: Any) -> bool:
        if other is self:
            return True
        if not isinstance(other, LinkingTrack):
            return False
        if other._min_time_point_number != self._min_time_point_number:
            return False
        return other._positions_by_time_point[0] == self._positions_by_time_point[0]

    def find_all_previous_and_descending_tracks(self, *, include_self: bool = False) -> Iterable["LinkingTrack"]:
        """Finds all tracks in the lineage of this track, including siblings, cousins, etc."""
        yield from self.find_all_previous_tracks(include_self=include_self)
        yield from self.find_all_descending_tracks(include_self=False)  # Avoid including self twice

    def get_duration_in_time_points(self) -> int:
        """Gets the time this track takes in time points. This is simply the number of recorded positions."""
        return len(self._positions_by_time_point)

    def will_divide(self) -> bool:
        """Checks whether there are at least two next tracks. Faster than calling len(track.get_next_tracks()) > 1,
        since this method doesn't create a set."""
        return len(self._next_tracks) > 1


class Links:
    """Represents all links between positions at different time points. This is used to follow particles over time. If a
    position is linked to two positions in the next time step, than that is a cell division. If a position is linked to
    no position in the next step, then either the cell died or the cell moved out of the image."""

    _tracks: List[LinkingTrack]
    _position_to_track: Dict[Position, LinkingTrack]

    def __init__(self):
        self._tracks = []
        self._position_to_track = dict()

    def add_links(self, links: "Links"):
        """Adds all links from the graph. Existing link are not removed. Changes may write through in the original
        links."""
        if self.has_links():
            for position1, position2 in links.find_all_links():
                self.add_link(position1, position2)
        else:
            self._tracks = links._tracks
            self._position_to_track = links._position_to_track

    def remove_all_links(self):
        """Removes all links in the experiment."""
        for track in self._tracks:  # Help the garbage collector by removing all the cyclic dependencies
            track._next_tracks.clear()
            track._previous_tracks.clear()
        self._tracks.clear()
        self._position_to_track.clear()

    def remove_links_of_position(self, position: Position):
        """Removes all links from and to the position."""
        track = self._position_to_track.get(position)
        if track is None:
            return

        age = track.get_age(position)
        if len(track._positions_by_time_point) == 1:
            # This was the only position in the track, remove track
            for previous_track in track._previous_tracks:
                self._decouple_next_track(previous_track, next_track=track)
            for next_track in track._next_tracks:
                self._decouple_previous_track(next_track, previous_track=track)
            self._tracks.remove(track)
        elif age == 0:
            # Position is first position of the track
            # Remove links with previous tracks
            for previous_track in track._previous_tracks:
                self._decouple_next_track(previous_track, next_track=track)
            track._previous_tracks = []

            # Remove actual position
            track._positions_by_time_point[0] = None
            while track._positions_by_time_point[0] is None:  # Remove all Nones at the beginning
                track._min_time_point_number += 1
                track._positions_by_time_point = track._positions_by_time_point[1:]
        else:
            # Position is further in the track
            if position.time_point_number() < track.max_time_point_number():
                # Need to split so that position is the last position of the track
                _ = self._split_track(track, age + 1)

            # Decouple from next tracks
            for next_track in track._next_tracks:
                self._decouple_previous_track(next_track, previous_track=track)
            track._next_tracks = []

            # Delete last position in the track
            del track._positions_by_time_point[-1]

            # Check if track needs to remain alive
            self._try_remove_if_one_length_track(track)

        # Remove from index
        del self._position_to_track[position]

    def replace_position(self, old_position: Position, position_new: Position):
        """Replaces one position with another. The old position is removed from the graph, the new one is added. All
        links will be moved over to the new position"""
        if old_position.time_point_number() != position_new.time_point_number():
            raise ValueError("Cannot replace with position at another time point")

        # Update in track
        track = self._position_to_track.get(old_position)
        if track is not None:
            track._positions_by_time_point[position_new.time_point_number() - track._min_time_point_number] = position_new

            # Update reference to track
            del self._position_to_track[old_position]
            self._position_to_track[position_new] = track

    def has_links(self) -> bool:
        """Returns True if at least one link is present."""
        return len(self._position_to_track) > 0

    def find_futures(self, position: Position) -> Set[Position]:
        """Returns the positions linked to this position in the next time point. Normally, this will be one position.
        However, if the cell divides between now and the next time point, two positions are returned. And if the cell
        track ends, zero positions are returned."""
        track = self._position_to_track.get(position)
        if track is None:
            return set()
        return track._find_futures(position.time_point_number())

    def find_single_future(self, position: Position) -> Optional[Position]:
        """If the given position is linked to exactly one other position, then that position is returned."""
        futures = self.find_futures(position)
        if len(futures) == 1:
            return futures.pop()
        return None

    def find_pasts(self, position: Position) -> Set[Position]:
        """Returns the positions linked to this position in the previous time point. Normally, this will be one
        position. However, the cell track just started, zero positions are returned. In the case of a cell merge,
        multiple positions are returned."""
        track = self._position_to_track.get(position)
        if track is None:
            return set()
        return track._find_pasts(position.time_point_number())

    def find_single_past(self, position: Position) -> Optional[Position]:
        """If the position has a single link to the previous time point, then this method returns that linked position.
        Otherwise, it returns None."""
        pasts = self.find_pasts(position)
        if len(pasts) == 1:
            return pasts.pop()
        return None

    def find_appeared_positions(self, time_point_number_to_ignore: Optional[int] = None) -> Iterable[Position]:
        """This method gets all positions that "popped up out of nothing": that have no links to the past. You can give
        this method a time point number to ignore. Usually, this would be the first time point number of the experiment,
        as cells that have no links to the past in the first time point are not that interesting."""
        for track in self._tracks:
            if time_point_number_to_ignore is None or time_point_number_to_ignore != track._min_time_point_number:
                if len(track.get_previous_tracks()) == 0:
                    yield track.find_first_position()

    def find_disappeared_positions(self, time_point_number_to_ignore: Optional[int] = None) -> Iterable[Position]:
        """This method gets all positions that "disappear into nothing": that have no links to the future. You can give
        this method a time point number to ignore. Usually, this would be the last time point number of the experiment,
        as cells that have no links to the future in the last time point are not that interesting."""
        for track in self._tracks:
            if time_point_number_to_ignore is None or time_point_number_to_ignore != track._positions_by_time_point[-1].time_point_number():
                if len(track.get_next_tracks()) == 0:
                    yield track.find_last_position()

    def add_link(self, position1: Position, position2: Position):
        """Adds a link between the positions. The linking network will be initialized if necessary."""
        dt = position1.time_point_number() - position2.time_point_number()
        if dt == 0:
            raise ValueError(f"Positions are in the same time point: {position1} cannot be linked to {position2}")
        if dt > 0:
            # Make sure position1 comes first in time
            position1, position2 = position2, position1
            dt = -dt
        if dt < -1:
            raise ValueError(f"Link skipped a time point: {position1} cannot be linked to {position2}")


        track1 = self._position_to_track.get(position1)
        track2 = self._position_to_track.get(position2)

        if track1 is not None and track2 is not None and self.contains_link(position1, position2):
            return  # Already has that link, don't add a second link (this will corrupt the data structure)

        if track1 is not None and track2 is None:
            if track1.max_time_point_number() == position1.time_point_number() \
                    and not track1._next_tracks \
                    and position2.time_point_number() == position1.time_point_number() + 1:
                # This very common case of adding a single position to a track is singled out
                # It could be handled just fine by the code below, which will create a new track and then merge the
                # tracks, but this is faster
                track1._positions_by_time_point.append(position2)
                self._position_to_track[position2] = track1
                return

        if track1 is None:  # Create new mini-track
            track1 = LinkingTrack([position1])
            self._tracks.append(track1)
            self._position_to_track[position1] = track1

        if track2 is None:  # Create new mini-track
            track2 = LinkingTrack([position2])
            self._tracks.append(track2)
            self._position_to_track[position2] = track2

        if position1.time_point_number() < track1.max_time_point_number():
            # Need to split track 1 so that position1 is at the end
            _ = self._split_track(track1, track1.get_age(position1) + 1)

        if position2.time_point_number() > track2._min_time_point_number:
            # Need to split track 2 so that position2 is at the start
            part_after_split = self._split_track(track2, track2.get_age(position2))
            track2 = part_after_split

        # Connect the tracks
        track1._next_tracks.append(track2)
        track2._previous_tracks.append(track1)
        self._try_merge(track1, track2)

    def get_lineage_data(self, track: LinkingTrack, data_name: str) -> Optional[DataType]:
        """Gets the attribute of the lineage tree. Returns None if not found."""
        # Find earliest track
        previous_tracks = track._previous_tracks
        while len(previous_tracks) > 0:
            track = track.get_previous_tracks().pop()
            previous_tracks = track._previous_tracks

        return track._lineage_data.get(data_name)

    def set_lineage_data(self, track: LinkingTrack, data_name: str, value: Optional[DataType]):
        """Adds or overwrites the given attribute for the given lineage (not the individual track!). Set the value to
        None to delete the attribute.
        """
        if data_name == "id":
            raise ValueError("The data_name 'id' is reserved for internal use.")
        if data_name.startswith("__"):
            raise ValueError(f"The data name {data_name} is not allowed: data names must not start with '__'.")

        # Find earliest track
        previous_tracks = track._previous_tracks
        while len(previous_tracks) > 0:
            track = track.get_previous_tracks().pop()
            previous_tracks = track._previous_tracks

        # Store or remove meta data
        if value is None:
            # Remove value
            if data_name in track._lineage_data:
                del track._lineage_data[data_name]
        else:
            # Store value
            track._lineage_data[data_name] = value

    def find_all_data_of_lineage(self, track: LinkingTrack) -> Iterable[Tuple[str, DataType]]:
        """Finds all lineage data of the given track."""
        # Find earliest track
        previous_tracks = track._previous_tracks
        while len(previous_tracks) > 0:
            track = track.get_previous_tracks().pop()
            previous_tracks = track._previous_tracks

        yield from track._lineage_data.items()

    def find_links_of(self, position: Position) -> Set[Position]:
        """Gets all links of a position, both to the past and the future."""
        track = self._position_to_track.get(position)
        if track is None:
            return set()
        return track._find_futures(position.time_point_number()) | track._find_pasts(position.time_point_number())

    def find_all_positions(self) -> Iterable[Position]:
        """Gets all positions in the linking graph. Note that positions without links are not included here."""
        return self._position_to_track.keys()

    def remove_link(self, position1: Position, position2: Position):
        """Removes the link between the given positions. Does nothing if there is no link between the positions."""
        if position1.time_point_number() > position2.time_point_number():
            position2, position1 = position1, position2

        if position1.time_point_number() == position2.time_point_number():
            return  # No link can possibly exist

        track1 = self._position_to_track.get(position1)
        track2 = self._position_to_track.get(position2)
        if track1 is None or track2 is None:
            return  # No link exists
        if track1 == track2:
            # So positions are in the same track

            # Check if there is nothing in between
            for time_point_number in range(position1.time_point_number() + 1, position2.time_point_number()):
                if track1.find_position_at_time_point_number(time_point_number) is not None:
                    return  # There's a position in between, so the specified link doesn't exist

            # Split directly after position1
            new_track = self._split_track(track1, position1.time_point_number() + 1 - track1._min_time_point_number)
            track1._next_tracks = []
            self._try_remove_if_one_length_track(track1)
            new_track._previous_tracks = []
            self._try_remove_if_one_length_track(new_track)
        else:
            # Check if the tracks connect
            if not track1.max_time_point_number() == position1.time_point_number():
                # Position 1 is not the last position in its track, so it cannot be connected to another track
                return
            if not track2._min_time_point_number == position2.time_point_number():
                # Position 2 is not the first position in its track, so it cannot be connected to another track
                return

            # The tracks may be connected. Remove the connection, if any
            if track2 in track1._next_tracks:
                self._decouple_next_track(track1, next_track=track2)
                self._decouple_previous_track(track2, previous_track=track1)


    def _decouple_next_track(self, track: LinkingTrack, *, next_track: LinkingTrack):
        """Removes a next track from the current track. If only one next track remains, a merge with the remaining next
        track is performed. If the track ends up with only a single time point and no links to other tracks, it will be
        removed. Raises ValueError if next_track is not in track.get_next_tracks().

        Note: this is a low-level function. If you plan on keeping the next_track object alive (so it's still in the
        linking network), then you'll also need to remove `track` as a previous track from `next_track`. Call
        self.debug_sanity_check() if you're unsure that the links are still consistent with each other."""
        track._next_tracks.remove(next_track)
        if len(track._next_tracks) == 1:  # Used to have two next tracks, now only one - try a merge
            self._try_merge(track, next(iter(track._next_tracks)))
        else:
            self._try_remove_if_one_length_track(track)

    def _decouple_previous_track(self, track: LinkingTrack, *, previous_track: LinkingTrack):
        """Removes a previous track from the current track. If only one previous track remains, a merge with the
        remaining previous track is performed.  If the track ends up with only a single time point and no links to other
        tracks, it will be removed. Raises ValueError if previous_track is not in track.get_previous_tracks().

        Note: this is a low-level function. If you plan on keeping the previous_track object alive (so it's still in the
        linking network), then you'll also need to remove `track` as a next track from `previous_track`. Call
        self.debug_sanity_check() if you're unsure that the links are still consistent with each other."""
        track._previous_tracks.remove(previous_track)
        if len(track._previous_tracks) == 1:  # Used to have two next tracks, now only one - try a merge
            self._try_merge(track, next(iter(track._previous_tracks)))
        else:
            self._try_remove_if_one_length_track(track)

    def _try_remove_if_one_length_track(self, track: LinkingTrack):
        """Removes the track if it is just a single time point and has no links or lineage data."""
        if len(track) != 1:
            return  # Don't remove tracks of multiple time points
        if len(track._previous_tracks) != 0 or len(track._next_tracks) != 0:
            return  # Not safe to remove this track
        if len(track._lineage_data) != 0:
            return  # Has metadata, don't delete

        # Safe to delete
        del self._position_to_track[track.find_first_position()]
        self._tracks.remove(track)

    def contains_link(self, position1: Position, position2: Position) -> bool:
        """Returns True if the two given positions are linked to each other."""
        if position1.time_point_number() == position2.time_point_number():
            return False  # Impossible to have a link

        if position1.time_point_number() < position2.time_point_number():
            return position1 in self.find_pasts(position2)
        else:
            return position1 in self.find_futures(position2)

    def contains_position(self, position: Position) -> bool:
        """Returns True if the given position is part of this linking network."""
        return position in self._position_to_track

    def find_all_links(self) -> Iterable[Tuple[Position, Position]]:
        """Gets all available links. The first position is always the earliest in time."""
        for track in self._tracks:
            # Return inter-track links
            previous_position = None
            for position in track.positions():
                if previous_position is not None:
                    yield previous_position, position
                previous_position = position

            # Return links to next track
            for next_track in track._next_tracks:
                yield previous_position, next_track.find_first_position()

            # (links to previous track are NOT returned as those links will already be included by that previous track
            # as links to the next track)

    def __len__(self) -> int:
        """Returns the total number of links."""
        total = 0
        for track in self._tracks:
            total += len(track) - 1  # A track of three cells contains two links
            total += len(track._next_tracks)  # Links to next tracks are also links
            # (links to previous track are NOT counted as those links will already be included by that previous track
            # as links to the next track)
        return total

    def copy(self) -> "Links":
        """Returns a copy of all the links, so that you can modify that data set without affecting this one."""
        copy = Links()

        # Copy over tracks
        for track in self._tracks:
            copied_track = LinkingTrack(track._positions_by_time_point.copy())
            copied_track._lineage_data = track._lineage_data.copy()
            copy._tracks.append(copied_track)
            for position in track.positions():
                copy._position_to_track[position] = copied_track

        # We can now re-establish the links between all tracks
        for track in self._tracks:
            track_copy = copy._position_to_track[track.find_first_position()]
            for next_track in track._next_tracks:
                next_track_copy = copy._position_to_track[next_track.find_first_position()]
                track_copy._next_tracks.append(next_track_copy)
                next_track_copy._previous_tracks.append(track_copy)

        return copy

    def _split_track(self, old_track: LinkingTrack, split_index: int) -> LinkingTrack:
        """Modifies the given track so that all positions after a certain time points are removed, and placed in a new
        track. So positions[0:split_index] will remain in this track, positions[split_index:] will be moved."""
        positions_after_split = old_track._positions_by_time_point[split_index:]

        # Remove None at front (this is safe, as the last position in the track may never be None)
        while positions_after_split[0] is None:
            positions_after_split = positions_after_split[1:]

        # Delete positions from after the split from original track
        del old_track._positions_by_time_point[split_index:]

        # Create a new track, add all connections
        track_after_split = LinkingTrack(positions_after_split)
        track_after_split._next_tracks = old_track._next_tracks
        for new_next_track in track_after_split._next_tracks:
            new_next_track._update_link_to_previous(was=old_track, will_be=track_after_split)
        track_after_split._previous_tracks = [old_track]
        old_track._next_tracks = [track_after_split]

        # Update indices for changed tracks
        self._tracks.insert(self._tracks.index(old_track) + 1, track_after_split)
        for position_after_split in positions_after_split:
            self._position_to_track[position_after_split] = track_after_split

        return track_after_split

    def _try_merge(self, track1: LinkingTrack, track2: LinkingTrack):
        """Call this method with two tracks that are ALREADY connected to each other. If possible, they will be merged
        into one big track."""
        if track1._min_time_point_number > track2._min_time_point_number:
            first_track, second_track = track2, track1
        else:
            first_track, second_track = track1, track2

        if len(second_track._previous_tracks) != 1 or len(first_track._next_tracks) != 1:
            return  # Cannot be merged
        # Ok, they can be merged into just first_track. Move all positions over to first_track.
        gap_length = second_track._min_time_point_number -\
                     (first_track._min_time_point_number + len(first_track._positions_by_time_point))
        if gap_length != 0:
            raise ValueError("Skipping a time point")
        first_track._positions_by_time_point += second_track._positions_by_time_point

        # Update registries
        first_track._lineage_data.update(second_track._lineage_data)
        self._tracks.remove(second_track)
        for moved_position in second_track.positions():
            self._position_to_track[moved_position] = first_track
        first_track._next_tracks = second_track._next_tracks
        for new_next_track in first_track._next_tracks:  # Notify all next tracks that they have a new predecessor
            new_next_track._update_link_to_previous(second_track, first_track)

    def debug_sanity_check(self):
        """Checks if the data structure still has a valid structure. If not, this method throws ValueError. This should
        never happen if you only use the public methods (those without a _ at the start), and don't poke around in
        internal code.

        This method is very useful to debug the data structure if you get some weird results."""
        for position, track in self._position_to_track.items():
            if track not in self._tracks:
                raise ValueError(f"{track} is not in the track list, but is in the index for position {position}")

        for track in self._tracks:
            if len(track._positions_by_time_point) == 0:
                raise ValueError(f"Empty track at t={track._min_time_point_number}")
            if len(track._positions_by_time_point) == 1 and len(track._previous_tracks) == 0\
                    and len(track._next_tracks) == 0 and len(track._lineage_data) == 0:
                raise ValueError(f"Length=1 track at t={track._min_time_point_number}")
            if track.find_first_position() is None:
                raise ValueError(f"{track} has no first position")
            if track.find_last_position() is None:
                raise ValueError(f"{track} has no last position")
            if len(track._previous_tracks) > 0 and len(track._lineage_data) > 0:
                raise ValueError(f"{track} has lineage meta data, even though it is not the start of a lineage")
            for position in track.positions():
                if position not in self._position_to_track:
                    raise ValueError(f"{position} of {track} is not indexed")
                elif self._position_to_track[position] != track:
                    raise ValueError(f"{position} in track {track} is indexed as being in track"
                                     f" {self._position_to_track[position]}")
            for previous_track in track._previous_tracks:
                if previous_track.max_time_point_number() >= track._min_time_point_number:
                    raise ValueError(f"Previous track {previous_track} is not in the past compared to {track}")
                if track not in previous_track._next_tracks:
                    raise ValueError(f"Current track {track} is connected to previous track {previous_track}, but that"
                                     f" track is not connected to the current track.")
            if len(track._next_tracks) == 1 and len(track._next_tracks[0]._previous_tracks) == 1:
                raise ValueError(f"Track {track} and {track._next_tracks[0]} could have been merged into"
                                 f" a single track")

    def find_starting_tracks(self) -> Iterable[LinkingTrack]:
        """Gets all starting tracks, which are all tracks that have no links to the past."""
        for track in self._tracks:
            if not track._previous_tracks:
                yield track

    def find_ending_tracks(self) -> Iterable[LinkingTrack]:
        """Gets all ending tracks, which are all tracks that have no links to the future."""
        for track in self._tracks:
            if not track._next_tracks:
                yield track

    def get_track_by_id(self, track_id: int) -> Optional[LinkingTrack]:
        """Gets the track with the given id, or None if no such track exists."""
        if track_id < 0 or track_id >= len(self._tracks):
            return None
        return self._tracks[track_id]

    def get_track(self, position: Position) -> Optional[LinkingTrack]:
        """Gets the track the given position belong in."""
        return self._position_to_track.get(position)

    def sort_tracks_by_x(self):
        """Sorts the tracks, which affects the order in which most find_ functions return data (like
        find_starting_tracks)."""
        self._tracks.sort(key=lambda track: track.find_first_position().x)

    def find_all_tracks_in_time_point(self, time_point_number: int) -> Iterable[LinkingTrack]:
        """This method finds all tracks that run trough the given time point."""
        for track in self._tracks:
            if track._min_time_point_number > time_point_number:
                continue
            if track.max_time_point_number() < time_point_number:
                continue
            yield track

    def find_all_tracks(self) -> Iterable[LinkingTrack]:
        """Gets all tracks, even tracks that have another track before them."""
        yield from self._tracks

    def get_highest_track_id(self) -> int:
        """Gets the highest track id currently in use. Returns -1 if there are no tracks."""
        return len(self._tracks) - 1

    def find_all_tracks_and_ids(self) -> Iterable[Tuple[int, LinkingTrack]]:
        """Gets all tracks and their id. Just like get_all_tracks, this method returns tracks that have another track
        before them in time."""
        yield from enumerate(self._tracks)

    def get_position_near_time_point(self, position: Position, time_point: TimePoint) -> Position:
        """Follows the position backwards or forwards in time through the linking network, until a position as close as
        possible to the specified time has been reached. If the given position has no links, the same position will just
        be returned. If a cell divides, an arbitrary daughter cell will be picked."""
        track = self.get_track(position)
        if track is None:
            return position  # Position has no links
        if position.time_point() == time_point:
            return position  # Trivial case - we're already at the right time point

        time_point_number = time_point.time_point_number()
        if position.time_point_number() > time_point_number:
            # Run back in time
            while track.min_time_point_number() > time_point_number:
                next_tracks = track.get_previous_tracks()
                if len(next_tracks) == 0:
                    return track.find_first_position()  # Cannot go back further
                else:
                    track = next_tracks.pop()
            return track.find_position_at_time_point_number(time_point_number)
        else:
            # Run forwards in time
            while track.max_time_point_number() < time_point_number:
                next_tracks = track.get_next_tracks()
                if len(next_tracks) == 0:
                    return track.find_last_position()  # Cannot go forward further
                else:
                    track = next_tracks.pop()
            return track.find_position_at_time_point_number(time_point_number)

    def get_position_at_time_point(self, position: Position, time_point: TimePoint) -> Optional[Position]:
        """Follows the position backwards or forwards in time through the linking network, until a position at the
        given time point has been found. If a cell divides, an arbitrary daughter cell will be picked. Returns None if
        we couldn't track the position until the requested time point."""
        position_near_time_point = self.get_position_near_time_point(position, time_point)
        if position_near_time_point.time_point() != time_point:
            return None  # Failed
        return position_near_time_point

    def of_time_point(self, time_point: TimePoint) -> Iterable[Tuple[Position, Position]]:
        """Returns all links where one of the two positions is in that time point. The first position in each tuple is
        in the given time point, the second one is one time point earlier or later."""
        time_point_number = time_point.time_point_number()
        for track in self._tracks:
            track_min_time_point_number = track._min_time_point_number
            if track_min_time_point_number > time_point_number:
                continue
            track_max_time_point_number = track.max_time_point_number()
            if track_max_time_point_number < time_point_number:
                continue
            # Track crosses this time point
            position = track.find_position_at_time_point_number(time_point_number)
            for past_position in track._find_pasts(time_point_number):
                yield position, past_position
            for future_position in track._find_futures(time_point_number):
                yield position, future_position

    def get_track_id(self, track: LinkingTrack) -> Optional[int]:
        """Gets the track id of the given track. Returns None if the track is not stored in the linking data here."""
        try:
            return self._tracks.index(track)
        except ValueError:
            return None

    def iterate_to_past(self, position: Position) -> Iterable[Position]:
        """Iterates towards the past, yielding this position, the previous position, the position before that, ect.
         Stops at cell merges or at the first detection."""
        track = self.get_track(position)
        if track is None:
            yield position  # Only yield position itself
            return

        time_point_number = position.time_point_number()
        while True:
            if time_point_number < track.min_time_point_number():
                previous_tracks = track.get_previous_tracks()
                if len(previous_tracks) == 1:
                    track = previous_tracks.pop()
                else:
                    return  # No more or multiple previous positions, stop

            yield track.find_position_at_time_point_number(time_point_number)
            time_point_number -= 1

    def iterate_to_future(self, position: Position) -> Iterable[Position]:
        """Iterates towards the future, yielding this position, the next position, the position after that, ect.
         Stops at cell divisions or at the last detection."""
        track = self.get_track(position)
        if track is None:
            yield position  # Only yield position itself
            return

        time_point_number = position.time_point_number()
        while True:
            if time_point_number > track.max_time_point_number():
                next_tracks = track.get_next_tracks()
                if len(next_tracks) == 1:
                    track = next_tracks.pop()
                else:
                    return  # No more or multiple next positions, stop

            yield track.find_position_at_time_point_number(time_point_number)
            time_point_number += 1

    def move_in_time(self, time_point_delta: int):
        """Moves all data with the given time point delta."""
        # We need to update self._tracks and rebuild self._position_to_track
        self._position_to_track.clear()
        for track in self._tracks:
            track._min_time_point_number += time_point_delta
            for i, position in enumerate(track._positions_by_time_point):
                moved_position = position.with_time_point_number(position.time_point_number() + time_point_delta)
                track._positions_by_time_point[i] = moved_position
                self._position_to_track[moved_position] = track
