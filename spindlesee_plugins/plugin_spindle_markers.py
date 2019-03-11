from autotrack.core.links import Links, LinkingTrack
from autotrack.core.position import Position, PositionType
from autotrack.gui.window import Window
from autotrack.linking_analysis import linking_markers

SPINDLE = PositionType("SPINDLE", "mitotic spindle", (255, 255, 0))
LUMEN = PositionType("LUMEN", "lumen", (30, 30, 30))
MIDBODY = PositionType("MIDBODY", "mitotic midbody", (255, 0, 0))


def init(window: Window):
    gui_experiment = window.get_gui_experiment()
    gui_experiment.register_position_type(SPINDLE)
    gui_experiment.register_position_type(LUMEN)
    gui_experiment.register_position_type(MIDBODY)


def is_part_of_spindle(links: Links, position: Position) -> bool:
    """Return True if the given position is part of a spindle."""
    return linking_markers.get_position_type(links, position) == "SPINDLE"


def is_lumen(links: Links, position: Position) -> bool:
    """Return True if the given position is part of a lumen."""
    return linking_markers.get_position_type(links, position) == "LUMEN"


def is_part_of_midbody(links, position: Position) -> bool:
    """Returns True if the given position is part of the mitotic midbody."""
    return linking_markers.get_position_type(links, position) == "MIDBODY"
