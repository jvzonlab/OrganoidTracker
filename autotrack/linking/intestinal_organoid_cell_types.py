from enum import Enum
from typing import Optional

from autotrack.core.links import Links
from autotrack.core.position import Position


class IntestinalOrganoidCellType(Enum):
    # The biological cell type
    PANETH = 1


def get_cell_type(links: Links, position: Position) -> Optional[IntestinalOrganoidCellType]:
    """Gets the type of the cell, interpreted as the intestinal organoid cell type."""
    name = links.get_track_data(position, "cell_type")
    if name is None:
        return None
    return IntestinalOrganoidCellType[name.upper()]


def set_cell_type(links: Links, position: Position, type: Optional[IntestinalOrganoidCellType]):
    """Sets the type of the cell. Set to None to delete the cell type."""
    type_str = type.name.lower() if type is not None else None
    links.set_track_data(position, "cell_type", type_str)
