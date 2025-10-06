from typing import Dict, Tuple, Optional, Iterable, ItemsView, Set

from organoid_tracker.core import TimePoint
from organoid_tracker.core.links import Links
from organoid_tracker.core.position import Position
from organoid_tracker.core.typing import DataType



class LinkData:
    """All link metadata used to be stored in this class. Now it is stored directly in Links, so you can access it
    using `experiment.links` (instead of `experiment.link_data`). This class only exists for backwards
    compatibility, and acts as a thin wrapper of the Links class.
    """

    _forward: Links

    def __init__(self, forward: Links):
        self._forward = forward

    def has_link_data(self) -> bool:
        """Gets whether there is any link data stored here."""
        return self._forward.has_link_data()

    def get_link_data(self, position1: Position, position2: Position, data_name: str) -> Optional[DataType]:
        """Gets the attribute of the link with the given name. Returns None if not found. Raises ValueError if the two
                positions are not in consecutive time points."""
        return self._forward.get_link_data(position1, position2, data_name)

    def set_link_data(self, position1: Position, position2: Position, data_name: str, value: Optional[DataType]):
        """Adds or overwrites the given attribute for the given position. Set value to None to delete the attribute.
        Raises ValueError if the two positions are not in consecutive time points.
        """
        self._forward.set_link_data(position1, position2, data_name, value)

    def copy(self) -> "LinkData":
        """Creates a copy of this linking dataset. Changes to the copy will not affect this object, and vice versa."""
        return LinkData(self._forward.copy())

    def replace_link(self, position_old1: Position, position_old2: Position, position_new1: Position, position_new2: Position):
        """Replaces a link, for example if the position moved. Raises ValueError if any of the two links are not between
        consecutive time points. Raises ValueError if the time points of the links are changed."""
        self._forward.replace_position(position_old1, position_new1)
        self._forward.replace_position(position_old2, position_new2)

    def find_all_links_with_data(self, data_name: str) -> ItemsView[Tuple[Position, Position], DataType]:
        """Gets a dictionary of all positions with the given data marker. Do not modify the returned dictionary."""
        return self._forward.find_all_links_with_data(data_name)

    def find_all_data_of_link(self, position1: Position, position2: Position) -> Iterable[Tuple[str, DataType]]:
        """Finds all data associated with the given link. Raises ValueError if the two positions are not in
                consecutive time points."""
        return self._forward.find_all_data_of_link(position1, position2)

    def find_all_data_names(self) -> Set[str]:
        """Finds all data_names"""
        return self._forward.find_all_data_names()

