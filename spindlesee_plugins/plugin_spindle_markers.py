from typing import List, Optional, Iterable

from autotrack.core.connections import Connections
from autotrack.core.links import Links, LinkingTrack
from autotrack.core.position import Position, PositionType
from autotrack.gui.window import Window
from autotrack.imaging import angles
from autotrack.linking_analysis import linking_markers

SPINDLE = PositionType("SPINDLE", "mitotic spindle", (200, 200, 0))
LUMEN = PositionType("LUMEN", "lumen", (30, 30, 30))
MIDBODY = PositionType("MIDBODY", "mitotic midbody", (255, 0, 0))
EDGE = PositionType("EDGE", "organoid edge", (255, 255, 255))


def init(window: Window):
    gui_experiment = window.get_gui_experiment()
    gui_experiment.register_position_type(SPINDLE)
    gui_experiment.register_position_type(LUMEN)
    gui_experiment.register_position_type(MIDBODY)
    gui_experiment.register_position_type(EDGE)


def is_part_of_spindle(links: Links, position: Position) -> bool:
    """Return True if the given position is part of a spindle."""
    name = linking_markers.get_position_type(links, position)
    return name == "SPINDLE" or name == "SPINDLE_LUMEN"


def is_lumen(links: Links, position: Position) -> bool:
    """Return True if the given position is part of a lumen."""
    return linking_markers.get_position_type(links, position) == "LUMEN"


def is_part_of_midbody(links: Links, position: Position) -> bool:
    """Returns True if the given position is part of the mitotic midbody."""
    return linking_markers.get_position_type(links, position) == "MIDBODY"

def is_part_of_organoid_edge(links: Links, position: Position) -> bool:
    """Returns True if the given position is part of the edge of the organoid."""
    return linking_markers.get_position_type(links, position) == "EDGE"


class Spindle:
    """Represents all information on a mitotic spindle."""
    positions1: List[Position]
    positions2: List[Position]
    midbody: List[Position]
    midbody_edge: List[Optional[Position]]  # midbody_edge[i] points to the nearest edge of the organoid from midbody[i]
    lumen: Optional[Position]

    def __init__(self):
        self.positions1 = list()
        self.positions2 = list()
        self.midbody = list()
        self.midbody_edge = list()
        self.lumen = None

    def __str__(self):
        return f"{len(self.positions1)} spindle positions, {len(self.midbody)} midbody positions"

    def get_orientation_change(self) -> float:
        """Gets how many degrees the orientation of the spindle changed."""
        first_orientation = angles.direction_2d(self.positions1[0], self.positions2[0])
        last_orientation = angles.direction_2d(self.positions1[-1], self.positions2[-1])
        return angles.direction_change_of_line(first_orientation, last_orientation)


def _find_spindle(links: Links, connections: Connections, track: LinkingTrack) -> Optional[Spindle]:
    first_position = track.find_first_position()
    if not is_part_of_spindle(links, first_position):
        return None
    for connection in connections.find_connections_starting_at(first_position):
        if not is_part_of_spindle(links, connection):
            continue
        spindle = Spindle()

        # Follow division
        position1, position2 = first_position, connection
        while position1 is not None and position2 is not None \
                and connections.contains_connection(position1, position2) \
                and is_part_of_spindle(links, position1) \
                and is_part_of_spindle(links, position2):
            spindle.positions1.append(position1)
            spindle.positions2.append(position2)

            for connection in connections.find_connections(position1):  # Search for lumen
                if is_lumen(links, connection):
                    spindle.lumen = connection

            position1 = links.find_single_future(position1)
            position2 = links.find_single_future(position2)

        # Continue following midbody after division
        if is_part_of_midbody(links, position2):
            position1, position2 = position2, position1  # Swap

        while is_part_of_midbody(links, position1):
            spindle.midbody.append(position1)

            # Find edge of organoid (using annotation)
            edge_position = None
            for connection in connections.find_connections(position1):
                if is_part_of_organoid_edge(links, connection):
                    edge_position = connection
            spindle.midbody_edge.append(edge_position)

            position1 = links.find_single_future(position1)

        return spindle


def find_all_spindles(links: Links, connections: Connections) -> Iterable[Spindle]:
    """Finds all spindles + metadata in the experiment."""
    for track in links.find_all_tracks():
        spindle = _find_spindle(links, connections, track)
        if spindle is not None:
            yield spindle
