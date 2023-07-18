from typing import Optional, Dict

from organoid_tracker.core.typing import DataType


class GlobalData:
    """Miscellaneous data that applies to the entire experiment, such """

    _global_data: Dict[str, DataType]

    def __init__(self, start_value: Optional[Dict[str, DataType]] = None):
        if start_value is None:
            self._global_data = dict()
        else:
            self._global_data = start_value

    def merge_data(self, other: "GlobalData"):
        self._global_data.update(other._global_data)

    def has_global_data(self):
        return len(self._global_data) > 0

    def get_all_data(self) -> Dict[str, DataType]:
        """Gets a copy of all stored data."""
        return self._global_data.copy()

    def get_data(self, data_name: str) -> Optional[DataType]:
        """Gets the attribute with the given name (global data only). Returns None if not found."""
        return self._global_data.get(data_name)

    def set_data(self, data_name: str, data_value: Optional[DataType]):
        """Sets global data."""
        if data_value is None:
            if data_name in self._global_data:
                del self._global_data[data_name]
        else:
            self._global_data[data_name] = data_value

    def copy(self) -> "GlobalData":
        return GlobalData(self._global_data.copy())
