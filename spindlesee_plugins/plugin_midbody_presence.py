from typing import Dict, Any, List, Optional, Iterable

from autotrack.core.connections import Connections
from autotrack.core.links import LinkingTrack, Links
from autotrack.core.position import Position
from autotrack.gui.window import Window
from . import plugin_spindle_markers


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Midbody-Midbody presence": lambda: _show_midbody_presence(window)
    }


class _Spindle:
    positions1: List[Position]
    positions2: List[Position]
    midbody: List[Position]

    def __init__(self):
        self.positions1 = list()
        self.positions2 = list()
        self.midbody = list()

    def __str__(self):
        return f"{len(self.positions1)} spindle positions, {len(self.midbody)} midbody positions"


def _find_spindle(links: Links, connections: Connections, track: LinkingTrack) -> Optional[_Spindle]:
    first_position = track.find_first_position()
    if not plugin_spindle_markers.is_part_of_spindle(links, first_position):
        return None
    for connection in connections.find_connections_starting_at(first_position):
        if not plugin_spindle_markers.is_part_of_spindle(links, connection):
            continue
        spindle = _Spindle()

        # Follow division
        position1, position2 = first_position, connection
        while position1 != None and position2 != None and connections.contains_connection(position1, position2) \
                and plugin_spindle_markers.is_part_of_spindle(links, position1)\
                and plugin_spindle_markers.is_part_of_spindle(links, position2):
            spindle.positions1.append(position1)
            spindle.positions2.append(position2)

            position1 = links.find_single_future(position1)
            position2 = links.find_single_future(position2)

        # Continue following midbody after division
        if plugin_spindle_markers.is_part_of_midbody(links, position2):
            position1, position2 = position2, position1  # Swap

        while plugin_spindle_markers.is_part_of_midbody(links, position1):
            spindle.midbody.append(position1)
            position1 = links.find_single_future(position1)

        return spindle


def _find_all_spindles(links: Links, connections: Connections) -> Iterable[_Spindle]:
    for track in links.find_all_tracks():
        spindle = _find_spindle(links, connections, track)
        if spindle is not None:
            yield spindle


def _show_midbody_presence(window: Window):
    experiment = window.get_experiment()
    print("---")
    for spindle in _find_all_spindles(experiment.links, experiment.connections):
        print(spindle)
