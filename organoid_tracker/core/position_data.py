import warnings
from collections import defaultdict
from typing import Dict, AbstractSet, Optional, Iterable, Set, List, Any, Type, ItemsView, Tuple, Union

from organoid_tracker.core import TimePoint, min_none, max_none
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.typing import DataType


class PositionData:
    """All metadata used to be stored in this class. Now it is stored in PositionCollection, so you can access it
    using `experiment.positions` (instead of `experiment.position_data`). This class only exists for backwards
    compatibility, and acts as a thin wrapper of PositionCollection."""

    _forward: PositionCollection

    def __init__(self, positions: PositionCollection):
        """Deprecated wrapper object for PositionCollection. Nowadays, all position metadata is stored in
        PositionCollection, so create an instance of PositionCollection instead."""
        self._forward = positions

    def first_time_point_number(self) -> Optional[int]:
        """Gets the first time point that contains positions, or None if there are no positions stored."""
        return self._forward.first_time_point_number()

    def last_time_point_number(self) -> Optional[int]:
        """Gets the last time point (inclusive) that contains positions, or None if there are no positions stored."""
        return self._forward.last_time_point_number()

    def first_time_point(self) -> Optional[TimePoint]:
        """Gets the first time point that contains positions, or None if there are no positions stored."""
        return self._forward.first_time_point()

    def last_time_point(self) -> Optional[TimePoint]:
        """Gets the last time point (inclusive) that contains positions, or None if there are no positions stored."""
        return self._forward.last_time_point()

    def merge_data(self, position_data: "PositionData"):
        """Merges all position data"""
        self._forward.merge_data(position_data._forward)

    def has_position_data(self) -> bool:
        """Gets whether there is any position data stored here."""
        return self._forward.has_position_data()

    def has_position_data_with_name(self, data_name: str) -> bool:
        """Returns whether there is position data stored for the given type."""
        return self._forward.has_position_data_with_name(data_name)

    def get_position_data(self, position: Position, data_name: str) -> Optional[DataType]:
        """Gets the attribute of the position with the given name. Returns None if not found."""
        return self._forward.get_position_data(position, data_name)

    def set_position_data(self, position: Position, data_name: str, value: Optional[DataType]):
        """Adds or overwrites the given attribute for the given position. Set value to None to delete the attribute.

        If the data_name is 'id' or starts with "__", a ValueError is raised. This requirement was necessary for the
        old save system, so as long as OrganoidTracker still supports writing to the old format, this restriction
        remains in place.
        """
        self._forward.set_position_data(position, data_name, value)

    def copy(self) -> "PositionData":
        """Creates a copy of this position metadata collection. Changes made to the copy will not affect this instance
        and vice versa."""
        return PositionData(self._forward.copy())

    def find_all_positions_with_data(self, data_name: str) -> Iterable[Tuple[Position, DataType]]:
        """Gets a dictionary of all positions with the given data marker. Do not modify the returned dictionary."""
        return self._forward.find_all_positions_with_data(data_name)

    def find_all_data_of_position(self, position: Position) -> Iterable[Tuple[str, DataType]]:
        """Finds all stored data of a given position."""
        return self._forward.find_all_data_of_position(position)

    def add_positions_data(self, data_name: str, data_set: Dict[Position, DataType]):
        """Bulk-addition of position data. Should be faster that adding everything individually."""
        self._forward.add_positions_data(data_name, data_set)

    def delete_data_with_name(self, data_name: str):
        """Deletes the data with the given key, for all positions in the experiment."""
        self._forward.delete_data_with_name(data_name)

    def find_all_data_names(self) -> Set[str]:
        """Finds all data_names"""
        return self._forward.find_all_data_names()

    def get_data_names_and_types(self) -> Dict[str, Type]:
        """Gets all data names that are currently in use, along with their type. The type will be str, float, bool,
        list, or object. (The type int is never returned, for ints float is returned instead. This is done
        so that users don't have to worry about storing their numbers with the correct type.)"""
        return self._forward.get_data_names_and_types()

    def add_data_from_time_point_dict(self, time_point: TimePoint, positions: List[Position], metadata_dict: Dict[str, List[Optional[DataType]]]):
        """Adds a time point with positions and metadata. The metadata dictionary must contain lists of the same length
        as the positions list. The position and metadata lists must be in the same order, and the position list must
        not contain any duplicates.

        This method is kind of low-level, and is mostly used for loading data from files. It's faster than just calling
        set_position_data for every position. However, note that for speed reasons, this method does not check the
        positions list for duplicates.

        You can easily create the metadata dictionary
        """
        self._forward.add_data_from_time_point_dict(time_point, positions, metadata_dict)

    def create_time_point_dict(self, time_point: TimePoint, positions: List[Position]) -> Dict[str, List[Optional[DataType]]]:
        """Creates a dictionary of metadata lists for a given time point. The metadata lists are empty. This is useful
        for creating a new time point with the same positions as an existing time point, but with no metadata."""
        return self._forward.create_time_point_dict(time_point, positions)

    def move_in_time(self, time_point_delta: int):
        """Moves all data with the given time point delta."""
        self._forward.move_in_time(time_point_delta)
