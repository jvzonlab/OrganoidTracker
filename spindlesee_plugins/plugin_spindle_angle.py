from typing import Dict, Any, List, Tuple

from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from autotrack.core.connections import Connections
from autotrack.core.experiment import Experiment
from autotrack.core.links import Links
from autotrack.core.position import Position
from autotrack.gui import dialog
from autotrack.gui.window import Window
from autotrack.imaging import angles
from matplotlib import colors, cm

from autotrack.imaging.grapher import colorline

from . import plugin_spindle_markers

_DIVIDER = 50

class _Line:
    angles: List[Tuple[float, float]]
    positions: List[Tuple[Position, Position]]

    def __init__(self, positions: List[Tuple[Position, Position]], angles: List[Tuple[float, float]]):
        self.positions = positions
        self.angles = angles

    def is_rotating(self) -> bool:
        return abs(self.angles[-1][1]) > _DIVIDER


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Spindle-Angle of spindle over time...": lambda: _view_spindle_angle(window),
        "View//Spindle-Locations of rotating spindles...": lambda: _view_spindle_locations(window)
    }


def _view_spindle_angle(window: Window):
    experiment = window.get_experiment()
    angle_lists = _get_spindle_angles_list(experiment)

    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _show_figure(figure, angle_lists))


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
            angle_lists.append(angle_list)
    return angle_lists


def _mean(value1: float, value2: float) -> float:
    return (value1 + value2) / 2


def _view_spindle_locations(window: Window):
    experiment = window.get_experiment()
    angle_lists = _get_spindle_angles_list(experiment)

    figure = window.get_figure()
    axes = figure.gca()
    for angle_list in angle_lists:
        axes.plot([start.x for start in angle_list.positions[0]], [start.y for start in angle_list.positions[0]], color="lightgray", linewidth=3)
        axes.plot([end.x for end in angle_list.positions[-1]], [end.y for end in angle_list.positions[-1]], color="red", linewidth=3)
        axes.add_collection(colorline([_mean(pos[0].x, pos[1].x) for pos in angle_list.positions],
                            [_mean(pos[0].y, pos[1].y) for pos in angle_list.positions], cmap=cm.get_cmap('Reds'),
                                      linewidth=1))
    figure.canvas.draw()


def _create_angles_list(links: Links, connections: Connections, position1: Position, position2: Position,
                        minutes_per_time_point: float) -> _Line:
    """Gets a list of (minute, angle) points for the mitotic spindle."""
    position_list = []
    angle_list = []
    original_angle = angles.direction_2d(position1, position2)
    time_point = 0
    while connections.contains_connection(position1, position2) and plugin_spindle_markers.is_part_of_spindle(links, position1)\
            and plugin_spindle_markers.is_part_of_spindle(links, position2):
        angle = angles.direction_2d(position1, position2)
        relative_angle = angles.direction_change_of_line(original_angle, angle)
        angle_list.append((time_point * minutes_per_time_point, relative_angle))
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
    return _Line(position_list, angle_list)


def _get_highest_time(angle_lists: List[_Line]) -> float:
    highest_time = 0
    for line in angle_lists:
        highest_line_time = line.angles[-1][0]  # Highest time will be the x coord of the last entry
        if highest_line_time > highest_time:
            highest_time = highest_line_time
    return highest_time


def _show_figure(figure: Figure, angle_lists: List[_Line]):
    highest_time = _get_highest_time(angle_lists)
    rotating_list = [angle_list.angles for angle_list in angle_lists if angle_list.is_rotating()]
    not_rotating_list = [angle_list.angles for angle_list in angle_lists if not angle_list.is_rotating()]
    color_names = colors.TABLEAU_COLORS
    color_codes = [colors.to_rgba(name, 1) for name in color_names]

    axes = figure.subplots(2, sharex=True)
    axes[0].set_xlim(0, highest_time)
    axes[0].set_ylim(-5, 95)
    axes[0].set_title('Rotation of spindle since start of mitosis')
    axes[0].add_collection(LineCollection(rotating_list, colors=color_codes))
    axes[0].set_ylabel(f"More than {_DIVIDER} degrees")

    axes[1].set_ylim(-5, 95)
    axes[1].add_collection(LineCollection(not_rotating_list, colors=color_codes))
    axes[1].set_xlabel("Time since spindle appeared (minutes)")
    axes[1].set_ylabel(f"Less than {_DIVIDER} degrees")
