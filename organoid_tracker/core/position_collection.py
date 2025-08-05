import warnings
from collections import defaultdict
from typing import Dict, AbstractSet, Optional, Iterable, List, Any, Tuple, Union, Type, Set

from organoid_tracker.core import TimePoint, min_none, max_none
from organoid_tracker.core.position import Position
from organoid_tracker.core.typing import DataType


class _PositionsAtTimePoint:
    """Holds the positions of a single point in time.

    For every time point, we store for every position a list of the metadata values. These metadata values
    correspond to the data names that are stored in the same time point (this object)."""

    _positions: Dict[Position, List[Any]]  # Position -> metadata
    _metadata_names: Dict[str, int]  # Metadata name -> index in metadata list in self._positions
    _metadata_counts: Dict[str, int]  # Metadata name -> number of times it is used

    def __init__(self):
        self._positions = dict()
        self._metadata_names = dict()
        self._metadata_counts = dict()

    def copy(self, ) -> "_PositionsAtTimePoint":
        """Gets a deep copy of this object. Changes to the returned object will not affect this object, and vice versa.
        """
        copy = _PositionsAtTimePoint()
        for position, metadata in self._positions.items():
            copy._positions[position] = metadata.copy()
        copy._metadata_names = self._metadata_names.copy()
        copy._metadata_counts = self._metadata_counts.copy()
        return copy

    def move_in_time(self, time_point_offset: int):
        """Must only be called from PositionCollection, otherwise the indexing is wrong."""
        new_dict = dict()
        for position, metadata in self._positions.items():
            new_dict[position.with_time_point_number(position.time_point_number() + time_point_offset)] = metadata
        self._positions = new_dict

    def replace_position(self, old_position: Position, new_position: Position):
        """Moves a position if it exists, keeping its metadata. Does nothing if the position is not in this collection.
        Does not check whether both positions have the same time point."""
        if new_position in self._positions:
            raise ValueError("New position already exists")
        if old_position == new_position:
            return
        old_data = self._positions.pop(old_position, None)
        if old_data is not None:
            self._positions[new_position] = old_data

    def delete_data_with_name(self, data_name: str):
        """Deletes the data with the given key, for all positions in the time point. Does nothing if the data name is
        not found in this time point."""
        index_of_data_name = self._metadata_names.get(data_name)
        if index_of_data_name is None:
            return  # Nothing to delete

        # Remove the data name from the metadata names
        for data in self._positions.values():
            if index_of_data_name < len(data):
                del data[index_of_data_name]

        # Update the indices of metadata names
        new_metadata_names = dict()
        for name, index in self._metadata_names.items():
            if index > index_of_data_name:
                new_metadata_names[name] = index - 1
            elif index == index_of_data_name:
                # Skip this index, we're deleting it
                continue
            else:
                new_metadata_names[name] = index
        self._metadata_names = new_metadata_names

        # Update the counts
        del self._metadata_counts[data_name]

    def find_all_positions_with_data(self, data_name: str) -> Iterable[Tuple[Position, DataType]]:
        index = self._metadata_names.get(data_name)
        if index is None:
            return

        for position, metadata in self._positions.items():
            if index < len(metadata):
                value = metadata[index]
                if value is not None:
                    yield position, value

    def add_position(self, position: Position):
        """Adds a position to this time point. If the position already exists, it is not added again."""
        if position in self._positions:
            return
        # Add the position with an empty metadata list
        self._positions[position] = []

    def set_position_data_required(self, position: Position, data_name: str, value_required: DataType):
        """Sets the data for a position. If the data already exists, it is overwritten. Note that the position data
        is *required* here, None is not allowed. To delete data, use delete_position_data_and_check_if_last."""
        if value_required is None:
            raise ValueError("Use delete_position_data_and_check_if_last to delete data")

        # Get existing position data
        data_of_position = self._positions.get(position)
        if data_of_position is None:
            return False  # Position does not exist, so we don't set the data

        # Look up where the data name is stored
        data_index = self._metadata_names.get(data_name)
        if data_index is None:
            # Need to add the data name
            data_index = len(self._metadata_names)
            self._metadata_names[data_name] = data_index

        # Modify the data list to insert the data value at the correct index
        while len(data_of_position) <= data_index:
            data_of_position.append(None)
        if data_of_position[data_index] is None:
            # New data value, increment count
            self._metadata_counts[data_name] = self._metadata_counts.get(data_name, 0) + 1

        data_of_position[data_index] = value_required
        return True

    def set_position_data_required_multiple(self, data_name: str, values_required: Dict[Position, DataType]):
        """Sets the data for a position. If the data already exists, it is overwritten. Note that the position data
        is *required* here, None is not allowed. To delete data, use delete_position_data_and_check_if_last."""

        # Look up where the data name is stored
        data_index = self._metadata_names.get(data_name)
        if data_index is None:
            # Need to add the data name
            data_index = len(self._metadata_names)
            self._metadata_names[data_name] = data_index

        # Add the data values
        for position, value_required in values_required.items():
            if value_required is None:
                raise ValueError("Found None in values_required")

            # Get existing position data
            data_of_position = self._positions.get(position)
            if data_of_position is None:
                data_of_position = []
                self._positions[position] = data_of_position

            # Modify the data list to insert the data value at the correct index
            while len(data_of_position) <= data_index:
                data_of_position.append(None)
            if data_of_position[data_index] is None:
                # New data value, increment count
                self._metadata_counts[data_name] = self._metadata_counts.get(data_name, 0) + 1

            data_of_position[data_index] = value_required

    def delete_position_data_and_check_if_last(self, position: Position, data_name: str) -> bool:
        # Look up where the data name is stored
        index_of_data_name = self._metadata_names.get(data_name)
        if index_of_data_name is None:
            return False  # Nothing to delete

        # Get existing position data
        data_of_position = self._positions.get(position)
        if data_of_position is None:
            return False  # Nothing to delete

        # Delete the metadata value if it exists
        if index_of_data_name >= len(data_of_position):
            return False  # Nothing to delete

        if data_of_position[index_of_data_name] is None:
            return False  # Nothing to delete

        # Ok, now we're actually deleting something
        data_of_position[index_of_data_name] = None
        self._metadata_counts[data_name] -= 1
        if self._metadata_counts[data_name] == 0:
            # We deleted the last data of this type, so we can remove the data type from our index
            self.delete_data_with_name(data_name)

            return True  # Signal that the last data of this type was deleted
        return False

    def is_empty(self) -> bool:
        return len(self._positions) == 0

    def detach_position(self, position: Position) -> Union[bool, List[str]]:
        """Removes a position from this time point.
        The return value is somewhat complex:
        - If the position was not found, False is returned.
        - If the position was found and removed, True is returned.
        - However, if one or more metadata values were fully depleted by removing the position, a list of the depleted
          metadata names is returned. The caller can use these to update their own indices.
        """
        existing_data = self._positions.pop(position, None)
        if existing_data is None:
            return False

        metadata_names_to_delete = None
        for metadata_name, metadata_index in self._metadata_names.items():
            if metadata_index < len(existing_data) and existing_data[metadata_index] is not None:
                # We're deleting some metadata, so we need to decrement the count
                self._metadata_counts[metadata_name] -= 1
                if self._metadata_counts[metadata_name] == 0:
                    # Keep track of depleted metadata
                    if metadata_names_to_delete is None:
                        metadata_names_to_delete = []
                    metadata_names_to_delete.append(metadata_name)

        # Remove metadata names that are now depleted
        if metadata_names_to_delete is not None:
            for metadata_name in metadata_names_to_delete:
                self.delete_data_with_name(metadata_name)
            return metadata_names_to_delete

        # No metadata was depleted
        return True

    def merge_data(self, other: "_PositionsAtTimePoint"):
        """Merges the metadata of another instance into this one. The instances must be of the same time point.

        Note: this method is kind of slow, as it has to check every metadata value for every position.
        """

        # Add space for any new metadata names
        for other_metadata_name, other_metadata_index in other._metadata_names.items():
            if other_metadata_name not in self._metadata_names:
                self._metadata_names[other_metadata_name] = len(self._metadata_names)
                self._metadata_counts[other_metadata_name] = 0

        other_metadata_names = other._metadata_names
        for position, other_metadata_values in other._positions.items():
            our_metadata_values = self._positions.get(position)
            if our_metadata_values is None:
                our_metadata_values = []
                self._positions[position] = our_metadata_values

            for other_metadata_name, other_metadata_index in other_metadata_names.items():
                if other_metadata_index >= len(other_metadata_values):
                    continue
                other_metadata_value = other_metadata_values[other_metadata_index]
                if other_metadata_value is None:
                    continue

                # Need to add a value
                our_metadata_index = self._metadata_names[other_metadata_name]
                while len(our_metadata_values) <= our_metadata_index:
                    our_metadata_values.append(None)

                if our_metadata_values[our_metadata_index] is None:
                    # New data value, increment count
                    self._metadata_counts[other_metadata_name] = self._metadata_counts.get(other_metadata_name,
                                                                                           0) + 1
                our_metadata_values[our_metadata_index] = other_metadata_value

    def positions(self) -> Iterable[Position]:
        """View of all positions in this time point. Don't modify the returned positions, this will corrupt the
        internal data structure. Use the accessor methods instead."""
        return self._positions.keys()

    def contains_position(self, position: Position) -> bool:
        """Returns whether the given position is part of this time point."""
        return position in self._positions

    def __len__(self) -> int:
        """Returns the number of positions in this time point."""
        return len(self._positions)

    def lowest_z(self) -> Optional[int]:
        """Returns the lowest z in use for this time point. If there are no positions, returns None."""
        if not self._positions:
            return None
        return min(round(position.z) for position in self._positions.keys())

    def highest_z(self) -> Optional[int]:
        """Returns the highest z in use for this time point. If there are no positions, returns None."""
        if not self._positions:
            return None
        return max(round(position.z) for position in self._positions.keys())


def _guess_data_type(example_value: Any) -> Type:
    if isinstance(example_value, bool):
        return bool
    elif isinstance(example_value, int) or isinstance(example_value, float):
        return float
    elif isinstance(example_value, str):
        return str
    elif isinstance(example_value, list):
        return list
    else:
        return object


# noinspection PyProtectedMember
class PositionCollection:

    _all_positions: Dict[int, _PositionsAtTimePoint]
    _min_time_point_number: Optional[int] = None
    _max_time_point_number: Optional[int] = None
    _data_names_and_types: Dict[str, Type]  # Data name -> type

    def __init__(self, positions: Iterable[Position] = ()):
        """Creates a new positions collection with the given positions already present."""
        self._all_positions = dict()
        self._data_names_and_types = dict()

        for position in positions:
            self.add(position)

    def of_time_point(self, time_point: TimePoint) -> AbstractSet[Position]:
        """Returns all positions for a given time point. Returns an empty set if that time point doesn't exist."""
        positions_at_time_point = self._all_positions.get(time_point.time_point_number())
        if not positions_at_time_point:
            return set()
        return set(positions_at_time_point.positions())

    def detach_all_for_time_point(self, time_point: TimePoint):
        """Removes all positions for a given time point, if any."""
        if time_point.time_point_number() in self._all_positions:
            del self._all_positions[time_point.time_point_number()]
            self._recalculate_min_max_time_points()

    def add(self, position: Position):
        """Adds a position, optionally with the given shape. The position must have a time point specified."""
        time_point_number = position.time_point_number()
        if time_point_number is None:
            raise ValueError("Position does not have a time point, so it cannot be added")

        self._update_min_max_time_points_for_addition(time_point_number)

        positions_at_time_point = self._all_positions.get(time_point_number)
        if positions_at_time_point is None:
            positions_at_time_point = _PositionsAtTimePoint()
            self._all_positions[time_point_number] = positions_at_time_point
        positions_at_time_point.add_position(position)

    def _update_min_max_time_points_for_addition(self, new_time_point_number: int):
        """Bookkeeping: makes sure the min and max time points are updated when a new time point is added"""
        if self._min_time_point_number is None or new_time_point_number < self._min_time_point_number:
            self._min_time_point_number = new_time_point_number
        if self._max_time_point_number is None or new_time_point_number > self._max_time_point_number:
            self._max_time_point_number = new_time_point_number

    def _recalculate_min_max_time_points(self):
        """Bookkeeping: recalculates min and max time point if a time point was removed."""
        # Reset min and max, then repopulate by readding all time points
        self._min_time_point_number = None
        self._max_time_point_number = None
        for time_point_number in self._all_positions.keys():
            self._update_min_max_time_points_for_addition(time_point_number)

    def move_position(self, old_position: Position, new_position: Position):
        """Moves a position, keeping its shape. Does nothing if the position is not in this collection. Raises a value
        error if the time points the provided positions are None or if they do not match."""
        if old_position.time_point_number() != new_position.time_point_number():
            raise ValueError("Time points are different")

        time_point_number = old_position.time_point_number()
        if time_point_number is None:
            raise ValueError("Position does not have a time point, so it cannot be added")

        positions_at_time_point = self._all_positions.get(time_point_number)
        if positions_at_time_point is None:
            return  # Position was not in collection
        positions_at_time_point.replace_position(old_position, new_position)

    def detach_position(self, position: Position):
        """Removes a position from a time point. Does nothing if the position is not in this collection."""
        positions_at_time_point = self._all_positions.get(position.time_point_number())
        if positions_at_time_point is None:
            return

        return_value = positions_at_time_point.detach_position(position)
        if return_value is False:
            return  # Position was not found

        # Remove time point entirely if necessary
        if positions_at_time_point.is_empty():
            del self._all_positions[position.time_point_number()]
            self._recalculate_min_max_time_points()

        if return_value is True:
            return  # Position was found and removed, but no metadata was depleted

        for depleted_metadata_name in return_value:
            is_in_other_time_points = any(data_of_time_point._metadata_names.get(depleted_metadata_name) is not None
                                          for data_of_time_point in self._all_positions.values())
            if not is_in_other_time_points:
                del self._data_names_and_types[depleted_metadata_name]

    def first_time_point_number(self) -> Optional[int]:
        """Gets the first time point that contains positions, or None if there are no positions stored."""
        return self._min_time_point_number

    def last_time_point_number(self) -> Optional[int]:
        """Gets the last time point (inclusive) that contains positions, or None if there are no positions stored."""
        return self._max_time_point_number

    def first_time_point(self) -> Optional[TimePoint]:
        """Gets the first time point that contains positions, or None if there are no positions stored."""
        return TimePoint(self._min_time_point_number) if self._min_time_point_number is not None else None

    def last_time_point(self) -> Optional[TimePoint]:
        """Gets the last time point (inclusive) that contains positions, or None if there are no positions stored."""
        return TimePoint(self._max_time_point_number) if self._max_time_point_number is not None else None

    def contains_position(self, position: Position) -> bool:
        """Returns whether the given position is part of the experiment."""
        positions_at_time_point = self._all_positions.get(position.time_point_number())
        if positions_at_time_point is None:
            return False
        return positions_at_time_point.contains_position(position)

    def __len__(self):
        """Returns the total number of positions across all time points."""
        count = 0
        for positions_at_time_point in self._all_positions.values():
            count += len(positions_at_time_point)
        return count

    def __iter__(self):
        """Iterates over all positions."""
        for positions_at_time_point in self._all_positions.values():
            yield from positions_at_time_point.positions()

    def has_positions(self) -> bool:
        """Returns True if there are any positions stored here."""
        return len(self._all_positions) > 0

    def merge_data(self, other: "PositionCollection"):
        """Adds all positions and metadata of the other collection to this collection."""
        # Update data names and types
        self._data_names_and_types.update(other._data_names_and_types)

        # Merge all position data
        for time_point_number, metadata_at_time_point in other._all_positions.items():
            existing_metadata_at_time_point = self._all_positions.get(time_point_number)
            if existing_metadata_at_time_point is None:
                # Easy case: just copy the metadata
                self._all_positions[time_point_number] = metadata_at_time_point.copy()
            else:
                # Otherwise, do a merge
                existing_metadata_at_time_point.merge_data(metadata_at_time_point)

        # Update min and max time points
        self._min_time_point_number = min_none(self._min_time_point_number, other._min_time_point_number)
        self._max_time_point_number = max_none(self._max_time_point_number, other._max_time_point_number)

    def add_positions(self, other: "PositionCollection"):
        warnings.warn("PositionCollection.add_positions() was renamed to PositionCollection.merge_data()", DeprecationWarning)
        self.merge_data(other)

    def time_points(self) -> Iterable[TimePoint]:
        """Returns all time points from self.first_time_point() to self.last_time_point()."""
        return TimePoint.range(self.first_time_point(), self.last_time_point())

    def copy(self) -> "PositionCollection":
        """Creates a copy of this positions collection. Changes made to the copy will not affect this instance and vice
        versa."""
        the_copy = PositionCollection()
        for key, value in self._all_positions.items():
            the_copy._all_positions[key] = value.copy()

        the_copy._min_time_point_number = self._min_time_point_number
        the_copy._max_time_point_number = self._max_time_point_number
        the_copy._data_names_and_types = self._data_names_and_types.copy()
        return the_copy

    def nearby_z(self, z: int) -> Iterable[Position]:
        """Returns all positions (for any time point) for which round(position.z) == z."""
        # Not very optimized. There used to be a data structure indexed by Z, but in practice that was not needed.
        for positions_at_time_point in self._all_positions.values():
            for position in positions_at_time_point.positions():
                if round(position.z) == z:
                    yield position

    def of_time_point_and_z(self, time_point: TimePoint, z_min: Optional[int] = None, z_max: Optional[int] = None
                            ) -> Iterable[Position]:
        """Gets all positions that are nearby the given min to max z, inclusive."""
        of_time_point = self._all_positions.get(time_point.time_point_number())
        if of_time_point is None:
            return

        if z_min is None and z_max is None:
            # Just return all positions
            yield from of_time_point.positions()
            return

        for position in of_time_point.positions():
            if z_min is not None and round(position.z) < z_min:
                continue
            if z_max is not None and round(position.z) > z_max:
                continue
            yield position

    def lowest_z(self) -> Optional[int]:
        """Returns the lowest z in use, or None if there are no positions in this collection."""
        return min_none(positions_at_time_point.lowest_z() for positions_at_time_point in self._all_positions.values())

    def highest_z(self) -> Optional[int]:
        """Returns the lowest z in use, or None if there are no positions in this collection."""
        return max_none(positions_at_time_point.highest_z() for positions_at_time_point in self._all_positions.values())

    def count_positions(self, *, time_point: Optional[TimePoint]):
        """Counts the number of positions at the given time point. If notime point is given,
        positions of all available time points will be counted."""
        if time_point is not None:
            # Specific time point
            at_time_point = self._all_positions.get(time_point.time_point_number())
            if at_time_point is None:
                return 0  # No positions at this time point

            return len(at_time_point)
        else:
            # All time points, all z
            return len(self)

    def move_in_time(self, time_point_delta: int):
        """Moves all data with the given time point delta."""
        new_positions_dict = dict()
        for time_point_number, values_old in self._all_positions.items():
            values_old.move_in_time(time_point_delta)
            new_positions_dict[time_point_number + time_point_delta] = values_old
        self._all_positions = new_positions_dict

    def has_position_data(self) -> bool:
        """Gets whether there is any position metadata stored here."""
        return len(self._data_names_and_types) > 0

    def has_position_data_with_name(self, data_name: str) -> bool:
        """Returns whether there is position metadata stored for the given data name."""
        return data_name in self._data_names_and_types

    def get_position_data(self, position: Position, data_name: str) -> Optional[DataType]:
        """Gets the attribute of the position with the given name. Returns None if not found."""
        data_of_time_point = self._all_positions.get(position.time_point_number())
        if data_of_time_point is None:
            return None
        data_of_position = data_of_time_point._positions.get(position)
        if data_of_position is None:
            return None
        data_index = data_of_time_point._metadata_names.get(data_name)
        if data_index is None or data_index >= len(data_of_position):
            return None
        return data_of_position[data_index]

    def set_position_data(self, position: Position, data_name: str, value: Optional[DataType]):
        """Adds or overwrites the given attribute for the given position. Set value to None to delete the attribute.

        Does nothing if the position does not exist. Use `self.add(...)` to add a position.

        If the data_name is 'id' or starts with "__", a ValueError is raised. This requirement was necessary for the
        old save system, so as long as OrganoidTracker still supports writing to the old format, this restriction
        remains in place.
        """
        if data_name == "id":
            raise ValueError("The data_name 'id' is used to store the position itself.")
        if data_name.startswith("__"):
            raise ValueError(f"The data name {data_name} is not allowed: data names must not start with '__'.")

        data_of_time_point = self._all_positions.get(position.time_point_number())
        if data_of_time_point is None:
            data_of_time_point = _PositionsAtTimePoint()
            self._all_positions[position.time_point_number()] = data_of_time_point

        if value is None:
            deleted_last = data_of_time_point.delete_position_data_and_check_if_last(position, data_name)
            if deleted_last:
                # If the last data of this type was deleted, we can remove the data type from our index
                # if it is also not used in any other time point
                is_in_other_time_points = any(data_of_time_point._metadata_names.get(data_name) is not None
                                              for data_of_time_point in self._all_positions.values())
                if not is_in_other_time_points:
                    del self._data_names_and_types[data_name]
        else:
            if data_of_time_point.set_position_data_required(position, data_name, value):

                # Update our data type index
                if data_name not in self._data_names_and_types:
                    self._data_names_and_types[data_name] = _guess_data_type(value)

    def find_all_positions_with_data(self, data_name: str) -> Iterable[Tuple[Position, DataType]]:
        """Gets a dictionary of all positions with the given data marker. Do not modify the returned dictionary."""
        for data_of_time_point in self._all_positions.values():
            yield from data_of_time_point.find_all_positions_with_data(data_name)

    def find_all_data_of_position(self, position: Position) -> Iterable[Tuple[str, DataType]]:
        """Finds all stored data of a given position."""
        data_of_time_point = self._all_positions.get(position.time_point_number())
        if data_of_time_point is None:
            return
        data_of_position = data_of_time_point._positions.get(position)
        if data_of_position is None:
            return
        data_names = data_of_time_point._metadata_names
        for name, value in zip(data_names.keys(), data_of_position):
            if value is not None:
                yield name, value

    def add_positions_data(self, data_name: str, data_set: Dict[Position, DataType]):
        """Bulk-addition of position data. Should be faster that adding everything individually."""
        if len(data_set) == 0:
            return

        # Split the data by time point
        by_time_point = defaultdict(dict)
        for position, value in data_set.items():
            by_time_point[position.time_point_number()][position] = value

        # Add the data to the time points
        for time_point_number, data_set_for_time_point in by_time_point.items():
            data_of_time_point = self._all_positions.get(time_point_number)
            if data_of_time_point is None:
                data_of_time_point = _PositionsAtTimePoint()
                self._all_positions[time_point_number] = data_of_time_point

            data_of_time_point.set_position_data_required_multiple(data_name, data_set_for_time_point)

        # Update our data type index
        if data_name not in self._data_names_and_types:
            first_value = next(iter(data_set.values()))
            self._data_names_and_types[data_name] = _guess_data_type(first_value)

    def delete_data_with_name(self, data_name: str):
        """Deletes the data with the given key, for all positions in the experiment."""
        for positions_at_time_point in self._all_positions.values():
            positions_at_time_point.delete_data_with_name(data_name)

    def find_all_data_names(self) -> Set[str]:
        """Finds all data_names"""
        return set(self._data_names_and_types.keys())

    def get_data_names_and_types(self) -> Dict[str, Type]:
        """Gets all data names that are currently in use, along with their type. The type will be str, float, bool,
        list, or object. (The type int is never returned, for ints float is returned instead. This is done
        so that users don't have to worry about storing their numbers with the correct type.)"""
        return self._data_names_and_types.copy()

    def add_data_from_time_point_dict(self, time_point: TimePoint, positions: List[Position], metadata_dict: Dict[str, List[Optional[DataType]]]):
        """Adds a time point with positions and metadata. The metadata dictionary must contain lists of the same length
        as the positions list. The position and metadata lists must be in the same order, and the position list must
        not contain any duplicates.

        This method is kind of low-level, and is mostly used for loading data from files. It's faster than just calling
        set_position_data for every position. However, note that for speed reasons, this method does not check the
        positions list for duplicates.
        """

        positions_at_time_point = _PositionsAtTimePoint()

        metadata_names = dict()
        metadata_counts = dict()
        for index, (metadata_name, metadata_values) in enumerate(metadata_dict.items()):
            metadata_names[metadata_name] = index
            metadata_count = 0
            for value in metadata_values:
                if value is not None:
                    metadata_count += 1
            metadata_counts[metadata_name] = metadata_count
        positions_at_time_point._metadata_names = metadata_names
        positions_at_time_point._metadata_counts = metadata_counts

        metadata_values_all = list(metadata_dict.values())
        for meta_index in range(1, len(metadata_values_all)):
            if len(metadata_values_all[meta_index]) != len(positions):
                print(f"All metadata lists must have the same length. However, we have {len(positions)} positions and {metadata_values_all[meta_index]} has length {len(metadata_values_all[meta_index])}")

        for position_index in range(len(positions)):
            metadata_values_position = [metadata_values_all[meta_index][position_index] for meta_index in range(len(metadata_values_all))]
            positions_at_time_point._positions[positions[position_index]] = metadata_values_position

        existing_positions_at_time_point = self._all_positions.get(time_point.time_point_number())
        if existing_positions_at_time_point is not None:
            # Merge the data (slow, unfortunately)
            existing_positions_at_time_point.merge_data(positions_at_time_point)
        else:
            # Add as new time point
            self._all_positions[time_point.time_point_number()] = positions_at_time_point
            self._min_time_point_number = min_none(self._min_time_point_number, time_point.time_point_number())
            self._max_time_point_number = max_none(self._max_time_point_number, time_point.time_point_number())

        # Update our data type index using the first non-None value of each metadata
        for data_name, data_values in metadata_dict.items():
            if data_name in self._data_names_and_types:
                continue  # Already known

            # Not known yet, so we need to guess the data type based on the first non-None value
            for some_value in data_values:
                if some_value is not None:
                    self._data_names_and_types[data_name] = _guess_data_type(some_value)
                    break

    def create_time_point_dict(self, time_point: TimePoint, positions: List[Position]) -> Dict[str, List[Optional[DataType]]]:
        """Creates a dictionary of metadata lists for a given time point. The metadata lists are empty. This is useful
        for creating a new time point with the same positions as an existing time point, but with no metadata."""
        positions_at_time_point = self._all_positions.get(time_point.time_point_number())
        if positions_at_time_point is None:
            return dict()

        # Build empty metadata table
        metadata_dict = dict()
        for metadata_name in positions_at_time_point._metadata_names.keys():
            metadata_dict[metadata_name] = [None] * len(positions)

        # Build plain metadata names list, to quickly go from index -> name
        metadata_names_ordered: List[Optional[str]] = [None for _ in range(len(positions_at_time_point._metadata_names))]
        for metadata_name, metadata_index in positions_at_time_point._metadata_names.items():
            metadata_names_ordered[metadata_index] = metadata_name
        if None in metadata_names_ordered:
            raise ValueError(f"Metadata values did not have consistent 1 to N indexing: {positions_at_time_point._metadata_names}")

        # Fill the dictionary with the metadata values
        for i, position in enumerate(positions):
            metadata_values = positions_at_time_point._positions.get(position)
            if metadata_values is None:
                continue
            for metadata_name, metadata_value in zip(metadata_names_ordered, metadata_values):
                metadata_dict[metadata_name][i] = metadata_value

        return metadata_dict
