from typing import Dict, Any, List, Tuple

import numpy
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from autotrack.core import UserError
from autotrack.core.experiment import Experiment
from autotrack.gui import dialog
from autotrack.gui.window import Window
from autotrack.imaging import angles
from autotrack.imaging.grapher import colorline
from autotrack.util.mpl_helper import SANDER_APPROVED_COLORS
from . import plugin_spindle_markers
from .plugin_spindle_markers import Spindle

_DIVIDER = 45


class _Line:
    angles: List[Tuple[float, float]]  # List of (time, angle) tuples. First element is last time point
    spindle: Spindle

    def __init__(self, spindle: Spindle, angles: List[Tuple[float, float]]):
        self.spindle = spindle
        self.angles = angles

    def is_rotating(self) -> bool:
        return abs(self.angles[0][1]) > _DIVIDER

    def get_rotation(self) -> float:
        """Gets the change in angle from the first to the last time point"""
        return self.angles[0][1]

    def get_duration(self) -> float:
        """Gets the time that the spindle was visible."""
        return self.angles[0][0]


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Spindle-Angle of spindle over time...": lambda: _view_spindle_angle(window),
        "View//Spindle-Locations of rotating spindles...": lambda: _view_spindle_locations(window),
        "View//Spindle-Average spindle rotation...": lambda: _view_average_spindle_rotation(window)
    }


def _view_spindle_angle(window: Window):
    experiment = window.get_experiment()
    angle_lists = _get_spindle_angles_list(experiment)

    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _show_figure(figure, angle_lists), size_cm=(8,9))


def _view_average_spindle_rotation(window: Window):
    experiment = window.get_experiment()
    angle_lists = _get_spindle_angles_list(experiment)
    angle_changes = list()

    for line in angle_lists:
        print(line.angles)
        final_angle = line.angles[0][1]
        angle_changes.append(final_angle)

    if len(angle_changes) == 0:
        raise UserError("No spindles found", "No spindles found. Dit you mark the positions as spindles, and did you"
                                             " establish connections between opposing poles?")

    dialog.popup_message("Average rotation", f"There are {len(angle_changes)} spindles recorded. The average spindle"
                         f" rotation is {numpy.mean(angle_changes)} degrees.")


def _get_spindle_angles_list(experiment: Experiment) -> List[_Line]:
    links = experiment.links
    connections = experiment.connections
    minutes_per_time_point = experiment.images.resolution().time_point_interval_m
    angle_lists = []
    for spindle in plugin_spindle_markers.find_all_spindles(links, connections):
        angle_list = _create_angles_list(spindle, minutes_per_time_point)
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
        spindle = angle_list.spindle
        axes.plot([spindle.positions1[0].x, spindle.positions2[0].x],
                  [spindle.positions1[0].y, spindle.positions2[0].y],
                  color="lightgray", linewidth=3)
        axes.plot([spindle.positions1[-1].x, spindle.positions2[-1].x],
                  [spindle.positions1[-1].y, spindle.positions2[-1].y],
                  color="red", linewidth=3)
        axes.add_collection(colorline([_mean(pos[0].x, pos[1].x) for pos in zip(spindle.positions1, spindle.positions2)],
                                      [_mean(pos[0].y, pos[1].y) for pos in zip(spindle.positions1, spindle.positions2)],
                                      cmap=cm.get_cmap('Reds'),
                                      linewidth=1))
    figure.canvas.draw()


def _create_angles_list(spindle: Spindle, minutes_per_time_point: float) -> _Line:
    """Gets a list of (minute, angle) points for the mitotic spindle."""
    angle_list = []
    time_point = 0
    for position1, position2 in zip(spindle.positions1, spindle.positions2):
        angle = angles.direction_2d(position1, position2)
        angle_list.append((time_point * minutes_per_time_point, angle))

        # Advance to next time point
        time_point += 1

    # Make angles relative to final angle
    if len(angle_list) > 0:
        final_time, final_angle = angle_list[-1]
        angle_list = [(final_time - time, angles.direction_change_of_line(final_angle, angle)) for time, angle in
                      angle_list]

    return _Line(spindle, angle_list)


def _get_highest_time(angle_lists: List[_Line]) -> float:
    highest_time = 0
    for line in angle_lists:
        highest_line_time = line.get_duration()  # Highest time will be the x coord of the first entry
        if highest_line_time > highest_time:
            highest_time = highest_line_time
    return highest_time


def _show_figure(figure: Figure, angle_lists: List[_Line]):
    highest_time = _get_highest_time(angle_lists)
    lumen_list = [angle_list for angle_list in angle_lists if angle_list.spindle.lumen is not None]
    not_lumen_list = [angle_list for angle_list in angle_lists if angle_list.spindle.lumen is None]

    angles_of_list = [l.angles for l in lumen_list]
    angles_of_non_list = [l.angles for l in not_lumen_list]
    average_of_list = numpy.mean([angle_list.get_duration() for angle_list in lumen_list])
    average_of_non_list = numpy.mean([angle_list.get_duration() for angle_list in not_lumen_list])

    figure.suptitle("Rotation of spindle over time")
    axes: Tuple[Axes, Axes] = figure.subplots(2, sharex=True)
    axes[0].set_xlim(highest_time, 0)
    axes[0].set_ylim(-5, 95)
    axes[0].add_collection(LineCollection(angles_of_list, colors=SANDER_APPROVED_COLORS))
    axes[0].set_title(f"Clearly next to a lumen", fontdict={"fontsize": "medium"})
    axes[0].text(highest_time - 3, 1, f"Avg. duration: {average_of_list:.1f} min")
    axes[1].set_ylim(-5, 95)
    axes[1].add_collection(LineCollection(angles_of_non_list, colors=SANDER_APPROVED_COLORS))
    axes[1].set_xlabel("Time until spindle disappears (minutes)")
    axes[1].set_title(f"No clear lumen visible", fontdict={"fontsize": "medium"})
    axes[1].text(highest_time - 3, 80, f"Avg. duration: {average_of_non_list:.1f} min")
    for ax in axes:
        ax.tick_params(direction="in", bottom=True, top=True, left=True, right=True, which="both")
        ax.set_yticks([0, 45, 90], minor=False)
        ax.set_yticks([15, 30, 60, 75], minor=True)
        ax.plot([min(ax.get_xlim()), max(ax.get_xlim())], [_DIVIDER, _DIVIDER], color="lightgray")
    figure.text(0.01, 0.5, 'Rotation of spindle (degrees)', va='center', rotation='vertical')
