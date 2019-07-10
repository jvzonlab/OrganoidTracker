"""This plugin compares the orientation of spindles to the lumen. The spindle by two positions of type "spindle" on each
time point, each of them marking one end of a spindle. In the last time point where the spindle is visible, the lumen is
indicated by a position of type "lumen". The two spindle positions must be connected, and a connection from both spindle
positions to the lumen position must also be drawn.
"""

from typing import Dict, Any, List, Tuple, Optional

from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from ai_track.core.connections import Connections
from ai_track.core.experiment import Experiment
from ai_track.core.links import Links
from ai_track.core.position import Position
from ai_track.gui import dialog
from ai_track.gui.window import Window
from ai_track.imaging import angles
from ai_track.util.mpl_helper import SANDER_APPROVED_COLORS

from . import plugin_spindle_markers

_DIVIDER = 50


class _Line:
    angles: List[Tuple[float, float]]  # List of time and angle tuples. First element is last time point
    positions: List[Tuple[Position, Position]]

    def __init__(self, positions: List[Tuple[Position, Position]], angles: List[Tuple[float, float]]):
        self.positions = positions
        self.angles = angles

    def is_rotating(self) -> bool:
        return abs(self.angles[0][1]) > _DIVIDER


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Spindle-Angle of spindle compared to lumen...": lambda: _view_spindle_angle(window),
    }


def _view_spindle_angle(window: Window):
    experiment = window.get_experiment()
    angle_lists = _get_spindle_angles_list(experiment)

    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _show_figure(figure, angle_lists), size_cm=(8,5))


def _get_spindle_angles_list(experiment: Experiment) -> List[_Line]:
    links = experiment.links
    connections = experiment.connections
    minutes_per_time_point = experiment.images.resolution().time_point_interval_m
    angle_lists = []
    for track in experiment.links.find_all_tracks():
        first_position = track.find_first_position()
        if not plugin_spindle_markers.is_part_of_spindle(links, first_position):
            continue
        for connected_position in connections.find_connections_starting_at(first_position):
            if not plugin_spindle_markers.is_part_of_spindle(links, connected_position):
                continue
            angle_list = _create_angles_list(links, connections,
                                             first_position, connected_position, minutes_per_time_point)
            if angle_list is not None:
                angle_lists.append(angle_list)
    return angle_lists


def _mean(value1: float, value2: float) -> float:
    return (value1 + value2) / 2


def _create_angles_list(links: Links, connections: Connections, position1: Position, position2: Position,
                        minutes_per_time_point: float) -> Optional[_Line]:
    """Gets a list of (minute, angle) points for the mitotic spindle."""
    position_list = []
    angle_list = []
    time_point = 0
    while connections.contains_connection(position1, position2)\
            and plugin_spindle_markers.is_part_of_spindle(links, position1) \
            and plugin_spindle_markers.is_part_of_spindle(links, position2):
        angle = angles.direction_2d(position1, position2)
        angle_list.append((time_point * minutes_per_time_point, angle))
        position_list.append((position1, position2))

        # Find positions in next time point
        futures1 = links.find_futures(position1)
        futures2 = links.find_futures(position2)
        if len(futures1) != 1 or len(futures2) != 1:
            break

        # Advance to next time point
        position1 = futures1.pop()
        position2 = futures2.pop()
        time_point += 1

    # Find lumen
    lumen = None
    for connection in connections.find_connections(position1):
        if plugin_spindle_markers.is_lumen(links, connection):
            lumen = connection
    if lumen is not None and not connections.contains_connection(position2, lumen):
        print("For spindle at", position1, "only one of the positions has a connection drawn to the lumen")
        lumen = None
    if lumen is None:
        return

    spindle_average_pos = (position1 + position2) / 2
    lumen_angle = (angles.direction_2d(spindle_average_pos, lumen) + 90) % 360

    # Make angles relative to final angle
    if len(angle_list) > 0:
        final_time, final_angle = angle_list[-1]
        angle_list = [(final_time - time, angles.direction_change_of_line(lumen_angle, angle)) for time, angle in
                      angle_list]

    return _Line(position_list, angle_list)


def _get_highest_time(angle_lists: List[_Line]) -> float:
    highest_time = 0
    for line in angle_lists:
        highest_line_time = line.angles[0][0]  # Highest time will be the x coord of the first entry
        if highest_line_time > highest_time:
            highest_time = highest_line_time
    return highest_time


def _show_figure(figure: Figure, angle_lists: List[_Line]):
    highest_time = _get_highest_time(angle_lists)
    raw_angle_list = [angle_list.angles for angle_list in angle_lists]
    last_angle_list = [angle_list.angles[0] for angle_list in angle_lists]
    last_angle_list_x, last_angle_list_y = [x_y[0] for x_y in last_angle_list], [x_y[1] for x_y in last_angle_list]
    first_angle_list = [angle_list.angles[-1] for angle_list in angle_lists]
    first_angle_list_x, first_angle_list_y = [x_y[0] for x_y in first_angle_list], [x_y[1] for x_y in first_angle_list]

    axes = figure.gca()
    axes.set_xlim(highest_time + 3, -3)
    axes.set_ylim(-5, 95)
    axes.tick_params(direction="in", bottom=True, top=True, left=True, right=True, which="both")
    axes.set_yticks([0, 45, 90], minor=False)
    axes.set_yticks([15, 30, 60, 75], minor=True)
    axes.set_title('Rotation of spindle compared to lumen')
    axes.plot([min(axes.get_xlim()), max(axes.get_xlim())], [45, 45], color="lightgray")
    axes.scatter(last_angle_list_x, last_angle_list_y, 81, color=SANDER_APPROVED_COLORS)
    axes.scatter(first_angle_list_x, first_angle_list_y, 81, marker="X", color=SANDER_APPROVED_COLORS)
    axes.add_collection(LineCollection(raw_angle_list, colors=SANDER_APPROVED_COLORS, linewidths=[2]))
    axes.set_ylabel(f"Rotation (degrees)")
    axes.set_xlabel("Time until spindle disappears (minutes)")

