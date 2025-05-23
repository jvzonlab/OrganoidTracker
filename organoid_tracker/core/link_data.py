from typing import Dict, Tuple, Optional, Iterable, ItemsView, Set

from organoid_tracker.core import TimePoint
from organoid_tracker.core.position import Position
from organoid_tracker.core.typing import DataType


def _create_link_tuple(position1: Position, position2: Position) -> Tuple[Position, Position]:
    """Returns a tuple with the position that's first in time on position 0. Raises ValueError if the positions are
    not in consecutive time points."""
    if position1.time_point_number() + 1 == position2.time_point_number():
        return position1, position2
    if position1.time_point_number() - 1 == position2.time_point_number():
        return position2, position1
    raise ValueError(f"Not in consecutive time points, so no link can exist: {position1}---{position2}")


class _LinkDataOfTimePoint:
    """All link metadata from this time point to the next one."""
    _time_point_first: TimePoint
    _link_data: Dict[str, Dict[Tuple[Position, Position], DataType]]

    def __init__(self, time_point_first: TimePoint):
        self._link_data = dict()
        self._time_point_first = time_point_first

    def has_link_data(self) -> bool:
        """Gets whether there is any link data stored here."""
        return len(self._link_data) > 0

    def get_link_data(self, link_tuple: Tuple[Position, Position], data_name: str) -> Optional[DataType]:
        """Gets the attribute of the link with the given name. Returns None if not found. Raises ValueError if the two
        positions are not in consecutive time points."""
        data_of_links = self._link_data.get(data_name)
        if data_of_links is None:
            return None
        return data_of_links.get(link_tuple)

    def set_link_data(self, link_tuple: Tuple[Position, Position], data_name: str, value: Optional[DataType]):
        """Adds or overwrites the given attribute for the given position. Set value to None to delete the attribute.
        Raises ValueError if the two positions are not in consecutive time points.

        For compatibility with the old file format, some names ("source", "target", or anything starting with "__") are
        reserved and cannot be used as data names.
        """
        if data_name.startswith("__"):
            # Reserved for future/internal use
            raise ValueError(f"The data name {data_name} is not allowed: data names must not start with '__'.")
        if data_name == "source" or data_name == "target":
            # Would go wrong when saving to JSON
            raise ValueError(f"The data name {data_name} is not allowed: this is a reserved word.")

        data_of_links = self._link_data.get(data_name)
        if data_of_links is None:
            if value is None:
                return  # No value was stored already, so no need to change anything

            # Initialize dict for this data type
            data_of_links = dict()
            self._link_data[data_name] = data_of_links

        if value is None:
            # Delete
            if link_tuple in data_of_links:
                del data_of_links[link_tuple]
                if len(data_of_links) == 0:
                    # Remove dict for this data type
                    del self._link_data[data_name]
        else:
            # Store
            data_of_links[link_tuple] = value


    def copy(self) -> "_LinkDataOfTimePoint":
        """Creates a copy of this linking dataset. Changes to the copy will not affect this object, and vice versa."""
        copy = _LinkDataOfTimePoint(self._time_point_first)
        for data_name, data_value in self._link_data.items():
            copy._link_data[data_name] = data_value.copy()
        return copy

    def merge_data(self, other: "_LinkDataOfTimePoint"):
        """Merges all data from the given dataset into this one. Changes to the other dataset made afterwards may
        "write-through" into this dataset."""
        for data_name, values in other._link_data.items():
            if data_name not in self._link_data:
                self._link_data[data_name] = values
            else:
                self._link_data[data_name].update(values)

    def remove_link(self, link_tuple: Tuple[Position, Position]):
        """Removes all data of the given link. Raises ValueError if the two positions are not in consecutive time
        points."""
        for data_name, data_of_links in list(self._link_data.items()):
            # ^ The list(...) makes a defensive copy, so we can delete things while iterating over it
            if link_tuple in data_of_links:
                # Delete data for given link
                del data_of_links[link_tuple]

                if len(data_of_links) == 0:
                    # No remaining data of this type, delete it
                    del self._link_data[data_name]

    def replace_link(self, link_tuple_old: Tuple[Position, Position], link_tuple_new: Tuple[Position, Position]):
        """Replaces a link, for example if the position moved. Raises ValueError if any of the two links are not between
        consecutive time points. Raises ValueError if the time points of the links are changed."""
        # Actually replace
        for data_name, data_of_links in self._link_data.items():
            if link_tuple_old in data_of_links:
                data_old = data_of_links[link_tuple_old]
                del data_of_links[link_tuple_old]
                data_of_links[link_tuple_new] = data_old

    def find_all_links_with_data(self, data_name: str) -> ItemsView[Tuple[Position, Position], DataType]:
        """Gets a dictionary of all positions with the given data marker. Do not modify the returned dictionary."""
        data_set = self._link_data.get(data_name)
        if data_set is None:
            return dict().items()
        return data_set.items()

    def find_all_data_of_link(self, link_tuple: Tuple[Position, Position]) -> Iterable[Tuple[str, DataType]]:
        """Finds all data associated with the given link. Raises ValueError if the two positions are not in
        consecutive time points."""
        for data_name, data_of_links in self._link_data.items():
            if link_tuple in data_of_links:
                yield data_name, data_of_links[link_tuple]

    def find_all_data_names(self):
        """Finds all data_names"""
        return self._link_data.keys()

    def move_in_time(self, time_point_delta: int):
        """Moves all data with the given time point delta."""
        for data_key in list(self._link_data.keys()):
            values_new = dict()
            values_old = self._link_data[data_key]
            for (position_a, position_b), value in values_old.items():
                values_new[(position_a.with_time_point_number(position_a.time_point_number() + time_point_delta),
                            position_b.with_time_point_number(position_b.time_point_number() + time_point_delta))] = value
            self._link_data[data_key] = values_new
        self._time_point_first = TimePoint(self._time_point_first.time_point_number() + time_point_delta)


class LinkData:
    """Used to supply additional metadata to links."""

    _data_by_first_time_point: Dict[int, _LinkDataOfTimePoint]

    def __init__(self):
        self._data_by_first_time_point = dict()

    def has_link_data(self) -> bool:
        """Gets whether there is any link data stored here."""
        return len(self._data_by_first_time_point) > 0

    def get_link_data(self, position1: Position, position2: Position, data_name: str) -> Optional[DataType]:
        """Gets the attribute of the link with the given name. Returns None if not found. Raises ValueError if the two
        positions are not in consecutive time points."""
        link_tuple = _create_link_tuple(position1, position2)
        data_of_time_point = self._data_by_first_time_point.get(link_tuple[0].time_point_number())
        if data_of_time_point is None:
            return None
        return data_of_time_point.get_link_data(link_tuple, data_name)

    def set_link_data(self, position1: Position, position2: Position, data_name: str, value: Optional[DataType]):
        """Adds or overwrites the given attribute for the given position. Set value to None to delete the attribute.
        Raises ValueError if the two positions are not in consecutive time points.

        Note: this is a low-level API. See the linking_markers module for more high-level methods, for example for how
        to read end markers, error markers, etc.
        """
        link_tuple = _create_link_tuple(position1, position2)
        data_of_time_point = self._data_by_first_time_point.get(link_tuple[0].time_point_number())

        if data_of_time_point is None:
            if value is None:
                return  # No value was stored already, so no need to change anything

            # Need to create a new data_of_time_point
            data_of_time_point = _LinkDataOfTimePoint(link_tuple[0].time_point())
            self._data_by_first_time_point[link_tuple[0].time_point_number()] = data_of_time_point
        data_of_time_point.set_link_data(link_tuple, data_name, value)

        if value is None and not data_of_time_point.has_link_data():
            # Deleted the last data of this time point, so remove the time point
            del self._data_by_first_time_point[link_tuple[0].time_point_number()]

    def copy(self) -> "LinkData":
        """Creates a copy of this linking dataset. Changes to the copy will not affect this object, and vice versa."""
        copy = LinkData()
        for time_point_number, data_of_time_point in self._data_by_first_time_point.items():
            copy._data_by_first_time_point[time_point_number] = data_of_time_point.copy()
        return copy

    def merge_data(self, other: "LinkData"):
        """Merges all data from the given dataset into this one. Changes to the other dataset made afterwards may
        "write-through" into this dataset."""
        for time_point_number, other_data_of_time_point in other._data_by_first_time_point.items():
            our_data_of_time_point = self._data_by_first_time_point.get(time_point_number)
            if our_data_of_time_point is None:
                # Just reference the other data
                self._data_by_first_time_point[time_point_number] = other_data_of_time_point
            else:
                # Need to merge
                our_data_of_time_point.merge_data(other_data_of_time_point)

    def remove_link(self, position1: Position, position2: Position):
        """Removes all data of the given link. Raises ValueError if the two positions are not in consecutive time
        points."""
        link_tuple = _create_link_tuple(position1, position2)
        data_of_time_point = self._data_by_first_time_point.get(link_tuple[0].time_point_number())
        if data_of_time_point is not None:
            data_of_time_point.remove_link(link_tuple)
            if not data_of_time_point.has_link_data():
                del self._data_by_first_time_point[link_tuple[0].time_point_number()]

    def replace_link(self, position_old1: Position, position_old2: Position, position_new1: Position, position_new2: Position):
        """Replaces a link, for example if the position moved. Raises ValueError if any of the two links are not between
        consecutive time points. Raises ValueError if the time points of the links are changed."""
        link_tuple_old = _create_link_tuple(position_old1, position_old2)
        link_tuple_new = _create_link_tuple(position_new1, position_new2)
        if link_tuple_old[0].time_point_number() != link_tuple_new[0].time_point_number():
            raise ValueError(f"Links cannot change time points. Old: {link_tuple_old}. New: {link_tuple_new}.")

        data_of_time_point = self._data_by_first_time_point.get(link_tuple_old[0].time_point_number())
        if data_of_time_point is not None:
            data_of_time_point.replace_link(link_tuple_old, link_tuple_new)

    def find_all_links_with_data(self, data_name: str) -> ItemsView[Tuple[Position, Position], DataType]:
        """Gets a dictionary of all positions with the given data marker. Do not modify the returned dictionary."""
        all_links_with_data = dict()
        for data_of_time_point in self._data_by_first_time_point.values():
            all_links_with_data.update(data_of_time_point.find_all_links_with_data(data_name))
        return all_links_with_data.items()

    def find_all_data_of_link(self, position1: Position, position2: Position) -> Iterable[Tuple[str, DataType]]:
        """Finds all data associated with the given link. Raises ValueError if the two positions are not in
        consecutive time points."""
        link_tuple = _create_link_tuple(position1, position2)
        data_of_time_point = self._data_by_first_time_point.get(link_tuple[0].time_point_number())
        if data_of_time_point is None:
            return
        yield from data_of_time_point.find_all_data_of_link(link_tuple)

    def find_all_data_names(self) -> Set[str]:
        """Finds all data_names"""
        data_names = set()
        for data_of_time_point in self._data_by_first_time_point.values():
            data_names.update(data_of_time_point.find_all_data_names())
        return data_names

    def move_in_time(self, time_point_delta: int):
        """Moves all data with the given time point delta."""
        new_dictionary = dict()
        for time_point_number, data_of_time_point in self._data_by_first_time_point.items():
            data_of_time_point.move_in_time(time_point_delta)
            new_dictionary[time_point_number + time_point_delta] = data_of_time_point
        self._data_by_first_time_point = new_dictionary
