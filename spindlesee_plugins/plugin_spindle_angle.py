from typing import Dict, Any, List, Tuple

from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from autotrack.core.connections import Connections
from autotrack.core.links import Links
from autotrack.core.position import Position
from autotrack.gui import dialog
from autotrack.gui.window import Window
from autotrack.imaging import angles


_Line = List[Tuple[float, float]]  # Typedef of a line going from point to point


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Spindle-Angle of spindle over time...": lambda: _view_spindle_angle(window)
    }


def _view_spindle_angle(window: Window):
    experiment = window.get_experiment()
    links = experiment.links
    connections = experiment.connections
    minutes_per_time_point = experiment.images.resolution().time_point_interval_m
    angle_lists = []

    for track in experiment.links.find_all_tracks():
        first_position = track.find_first_position()
        for connected_position in connections.find_connections_starting_at(first_position):
            angle_list = _create_angles_list(links, connections,
                                             first_position, connected_position, minutes_per_time_point)
            angle_lists.append(angle_list)

    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _show_figure(figure, angle_lists))


def _create_angles_list(links: Links, connections: Connections, position1: Position, position2: Position,
                        minutes_per_time_point: float) -> _Line:
    """Gets a list of (minute, angle) points for the mitotic spindle."""
    angle_list = []
    time_point = 0
    while connections.exists(position1, position2):
        angle = angles.direction_2d(position1, position2)
        angle_list.append((time_point * minutes_per_time_point, angle))

        # Find positions in next time point
        futures1 = links.find_futures(position1)
        futures2 = links.find_futures(position2)
        if len(futures1) != 1 or len(futures2) != 1:
            break

        # Advance to next time point
        position1 = futures1.pop()
        position2 = futures2.pop()
        time_point += 1
    return angle_list


def _get_highest_time(angle_lists: List[_Line]) -> float:
    highest_time = 0
    for line in angle_lists:
        highest_line_time = line[-1][0]  # Highest time will be the x coord of the last entry
        if highest_line_time > highest_time:
            highest_time = highest_line_time
    return highest_time


def _show_figure(figure: Figure, angle_lists: List[_Line]):
    highest_time = _get_highest_time(angle_lists)

    axes = figure.gca()
    axes.set_xlim(0, highest_time)
    axes.set_ylim(-5, 365)
    axes.set_xlabel("Time since spindle appeared")
    axes.set_ylabel("Angle of spindle (degrees)")
    axes.add_collection(LineCollection(angle_lists))
