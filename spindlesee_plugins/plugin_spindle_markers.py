from typing import List, Iterable

from autotrack.core.links import Links, LinkingTrack
from autotrack.core.position import Position, PositionType
from autotrack.gui.window import Window
from autotrack.linking_analysis import linking_markers

SPINDLE = PositionType("SPINDLE", "mitotic spindle", (255, 255, 0))


def init(window: Window):
    gui_experiment = window.get_gui_experiment()
    gui_experiment.register_position_type(SPINDLE)


def is_part_of_spindle(links: Links, position: Position) -> bool:
    """Return True if the given position is part of a spindle."""
    return linking_markers.get_position_type(links, position) == "SPINDLE"
