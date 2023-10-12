from typing import Dict, Optional, ItemsView, Iterable, Tuple, Union, Set, Type

from organoid_tracker.core.position import Position
from organoid_tracker.core.typing import DataType


class PositionData:
    _position_data: Dict[str, Dict[Position, DataType]]

    def __init__(self):
        self._position_data = dict()

    def merge_data(self, position_data: "PositionData"):
        # Merge data
        for data_name, values in position_data._position_data.items():
            if data_name not in self._position_data:
                self._position_data[data_name] = values
            else:
                self._position_data[data_name].update(values)

    def remove_position(self, position: Position):
        """Removes all data for the given position."""
        for data_set in self._position_data.values():
            if position in data_set:
                del data_set[position]

    def replace_position(self, old_position: Position, new_position: Position):
        """Replaces one position with another, such that all data associated with the old position becomes associated
         with tne nemw."""
        for data_name, data_dict in self._position_data.items():
            if old_position in data_dict:
                old_value = data_dict[old_position]
                del data_dict[old_position]
                data_dict[new_position] = old_value

    def has_position_data(self) -> bool:
        """Gets whether there is any position data stored here."""
        return len(self._position_data) > 0

    def has_position_data_with_name(self, data_name: str) -> bool:
        """Returns whether there is position data stored for the given type."""
        return data_name in self._position_data

    def get_position_data(self, position: Position, data_name: str) -> Optional[DataType]:
        """Gets the attribute of the position with the given name. Returns None if not found."""
        data_of_positions = self._position_data.get(data_name)
        if data_of_positions is None:
            return None
        return data_of_positions.get(position)

    def set_position_data(self, position: Position, data_name: str, value: Optional[DataType]):
        """Adds or overwrites the given attribute for the given position. Set value to None to delete the attribute.

        Note: this is a low-level API. See the linking_markers module for more high-level methods, for example for how
        to read end markers, error markers, etc.
        """
        if data_name == "id":
            raise ValueError("The data_name 'id' is used to store the position itself.")
        if data_name.startswith("__"):
            raise ValueError(f"The data name {data_name} is not allowed: data names must not start with '__'.")
        data_of_positions = self._position_data.get(data_name)
        if data_of_positions is None:
            if value is None:
                return  # No value was stored already, so no need to change anything

            # Intialize dict for this data type
            data_of_positions = dict()
            self._position_data[data_name] = data_of_positions

        if value is None:
            # Delete
            if position in data_of_positions:
                del data_of_positions[position]
                if len(data_of_positions) == 0:
                    # Remove dict for this data type
                    del self._position_data[data_name]
        else:
            # Store
            data_of_positions[position] = value

    def copy(self) -> "PositionData":
        copy = PositionData()
        for data_name, data_value in self._position_data.items():
            copy._position_data[data_name] = data_value.copy()
        return copy

    def find_all_positions_with_data(self, data_name: str) -> ItemsView[Position, DataType]:
        """Gets a dictionary of all positions with the given data marker. Do not modify the returned dictionary."""
        data_set = self._position_data.get(data_name)
        if data_set is None:
            return dict().items()
        return data_set.items()

    def find_all_data_of_position(self, position: Position) -> Iterable[Tuple[str, DataType]]:
        """Finds all stored data of a given position."""
        for data_name, data_values in self._position_data.items():
            data_value = data_values.get(position)
            if data_value is not None:
                yield data_name, data_value

    def add_positions_data(self, data_name: str, data_set: Dict[Position, DataType]):
        """Bulk-addition of position data. Should be much faster that adding everything individually."""
        existing_data_set = self._position_data.get(data_name)
        if existing_data_set is None:
            self._position_data[data_name] = data_set
        else:
            existing_data_set.update(data_set)

    def delete_data_with_name(self, data_name: str):
        """Deletes the data with the given key, for all positions in the experiment."""
        if data_name in self._position_data:
            del self._position_data[data_name]

    def find_all_data_names(self):
        """Finds all data_names"""
        return self._position_data.keys()

    def get_data_names_and_types(self) -> Dict[str, Type]:
        """Gets all data names that are currently in use, along with their type. The type will be str, float, bool,
        list, or object. (The type int is never returned, for ints float is returned instead. This is done
        so that users don't have to worry about storing their numbers with the correct type.)"""
        return_dict = dict()
        for key, values_by_position in self._position_data.items():
            if len(values_by_position) == 0:
                continue
            example_value = next(iter(values_by_position.values()))
            if isinstance(example_value, bool):
                return_dict[key] = bool
            elif isinstance(example_value, int) or isinstance(example_value, float):
                return_dict[key] = float
            elif isinstance(example_value, str):
                return_dict[key] = str
            elif isinstance(example_value, list):
                return_dict[key] = list
            else:
                return_dict[key] = object  # Don't know the type

        return return_dict

    def move_in_time(self, time_point_delta: int):
        """Moves all data with the given time point delta."""
        for data_key in list(self._position_data.keys()):
            values_new = dict()
            values_old = self._position_data[data_key]
            for position, value in values_old.items():
                values_new[position.with_time_point_number(position.time_point_number() + time_point_delta)] = value
            self._position_data[data_key] = values_new
