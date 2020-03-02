from typing import Dict, Any, List, Tuple

import numpy
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.imaging import angles
from organoid_tracker.imaging.grapher import colorline
from organoid_tracker.util.mpl_helper import SANDER_APPROVED_COLORS
from . import plugin_spindle_markers
from .plugin_spindle_markers import Spindle

_DIVIDER = 45
_OSCILLATE_THRESHOLD_DEGREES = 15


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

    def count_oscillations(self) -> int:
        """Counts how many times the spindle changed its oscillation"""
        oscillations = 0

        previous_previous_angle = None
        previous_angle = None
        for time, angle in self.angles:
            if previous_previous_angle is not None:
                previous_change = previous_previous_angle - previous_angle
                current_change = previous_angle - angle
                #if abs(previous_change - current_change) > _OSCILLATE_THRESHOLD_DEGREES:
                if previous_change < 0 and current_change > 0 or previous_change > 0 and current_change < 0:
                    # We're changing direction
                    oscillations += 1

            previous_previous_angle = previous_angle
            previous_angle = angle

        return oscillations


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Spindle-Angle of spindle over time...": lambda: _view_spindle_angle(window),
        "View//Spindle-Locations of rotating spindles...": lambda: _view_spindle_locations(window),
        "View//Spindle-Average spindle rotation...": lambda: _view_average_spindle_rotation(window),
        "Graph//Spindle-Histogram of oscillations...": lambda: _view_spindle_oscillations_histogram(window)
    }


def _get_x_min_avg_max(angle_lists: List[_Line]) -> Tuple[List[float], List[float], List[float], List[float]]:
    # Find longest spindle
    length_of_list = 0
    for angle_list in angle_lists:
        if len(angle_list.angles) > length_of_list:
            length_of_list = len(angle_list.angles)

    # Find x, avg - stdev, avg, and avg + stdev
    t_list = list()
    avg_minus_std_list = list()
    avg_list = list()
    avg_plus_std_list = list()
    for i in range(1, length_of_list + 1):
        angle_values = list()
        time_value = 0
        for angle_list in angle_lists:
            if len(angle_list.angles) >= i:
                time_value = angle_list.angles[-i][0]
                angle_values.append(angle_list.angles[-i][1])

        t_list.append(time_value)
        avg_minus_std_list.append(float(numpy.mean(angle_values) - numpy.std(angle_values)))
        avg_list.append(float(numpy.mean(angle_values)))
        avg_plus_std_list.append(float(numpy.mean(angle_values) + numpy.std(angle_values)))

    return t_list, avg_minus_std_list, avg_list, avg_plus_std_list


def _view_spindle_angle(window: Window):
    experiment = window.get_experiment()
    angle_lists = []
    for experiment in window.get_experiments():
        angle_lists += _get_spindle_angles_list(experiment, limit_duration=True)

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


def _view_spindle_oscillations_histogram(window: Window):
    experiment = window.get_experiment()
    angle_lists = _get_spindle_angles_list(experiment, limit_duration=True)

    def draw_graph(figure: Figure):
        axes: Tuple[Axes, Axes] = figure.subplots(nrows=2, ncols=1, sharex=True, sharey=False)
        ax_less_45, ax_more_45 = axes

        list_more_45 = [angle_list for angle_list in angle_lists if angle_list.is_rotating()]
        list_less_45 = [angle_list for angle_list in angle_lists if not angle_list.is_rotating()]

        oscillation_count_more_45 = [angle_list.count_oscillations() for angle_list in list_more_45]
        oscillation_count_less_45 = [angle_list.count_oscillations() for angle_list in list_less_45]

        max_oscillations = max(max(oscillation_count_less_45), max(oscillation_count_more_45))
        bins = list(range(max_oscillations + 1))
        ax_less_45.hist(oscillation_count_less_45, bins=bins, color="#74b9ff")
        ax_more_45.hist(oscillation_count_more_45, bins=bins, color="#74b9ff")

        ax_less_45.set_title("Less than 45 degrees")
        ax_more_45.set_title("More then 45 degrees")
        ax_more_45.set_xlabel("Number of oscillation changes per spindle")
        ax_more_45.set_yticks([0, 4, 8])
        ax_less_45.set_ylabel("Amount of spindles")
        for ax in axes:
            ax.tick_params(direction="in", bottom=True, top=True, left=True, right=True, which="both")
            ax.set_xticks(bins)

    dialog.popup_figure(window.get_gui_experiment(), draw_graph, size_cm=(8,9))


def _get_spindle_angles_list(experiment: Experiment, limit_duration: bool = False) -> List[_Line]:
    links = experiment.links
    connections = experiment.connections
    minutes_per_time_point = experiment.images.resolution().time_point_interval_m
    angle_lists = []
    for spindle in plugin_spindle_markers.find_all_spindles(links, connections):
        angle_list = _create_angles_list(spindle, minutes_per_time_point)
        angle_lists.append(angle_list)

    if limit_duration:
        angle_lists = [angle_list for angle_list in angle_lists if angle_list.get_duration() <= 25]
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


def _draw_horline(ax: Axes):
    ax.plot([min(ax.get_xlim()), max(ax.get_xlim())], [_DIVIDER, _DIVIDER], color="lightgray")


def _show_figure(figure: Figure, angle_lists: List[_Line]):
    highest_time = _get_highest_time(angle_lists)
    rotating_list = [angle_list for angle_list in angle_lists if angle_list.is_rotating()]
    non_rotating_list = [angle_list for angle_list in angle_lists if not angle_list.is_rotating()]

    angles_of_list = [l.angles for l in rotating_list]
    angles_of_non_list = [l.angles for l in non_rotating_list]
    average_of_list = numpy.mean([angle_list.get_duration() for angle_list in rotating_list])
    average_of_non_list = numpy.mean([angle_list.get_duration() for angle_list in non_rotating_list])

    # Average lines of all the lines
    avg_t_list, avg_min_list, avg_avg_list, avg_max_list = _get_x_min_avg_max(rotating_list)
    avg_non_t_list, avg_non_min_list, avg_non_avg_list, avg_non_max_list = _get_x_min_avg_max(non_rotating_list)

    figure.suptitle("Rotation of spindle over time")
    axes: Tuple[Tuple[Axes, Axes], Tuple[Axes, Axes]] = figure.subplots(nrows=2, ncols=2, sharex=True, sharey=True)

    for row in axes:
        for ax in row:
            ax.tick_params(direction="in", bottom=True, top=True, left=True, right=True, which="both")
            ax.set_yticks([0, 45, 90], minor=False)
            ax.set_yticks([15, 30, 60, 75], minor=True)
            ax.set_ylim(-5, 95)

    axes[0][0].set_xlim(highest_time + 3, 0)
    _draw_horline(axes[0][0])
    axes[0][0].add_collection(LineCollection(angles_of_list, colors=SANDER_APPROVED_COLORS))
    axes[0][0].set_title(f"More than 45 degrees", fontdict={"fontsize": "medium"})
    #axes[0][0].text(highest_time - 3, 1, f"Avg. duration: {average_of_list:.1f} min")
    _draw_horline(axes[1][0])
    axes[1][0].add_collection(LineCollection(angles_of_non_list, colors=SANDER_APPROVED_COLORS))
    axes[1][0].set_xlabel("Time until spindle disappears (minutes)")
    axes[1][0].set_title(f"Less than 45 degrees", fontdict={"fontsize": "medium"})
    #axes[1][0].text(highest_time - 3, 80, f"Avg. duration: {average_of_non_list:.1f} min")
    _draw_horline(axes[0][1])
    axes[0][1].plot(avg_t_list, avg_avg_list, color="#0984e3")
    axes[0][1].fill_between(avg_t_list, avg_min_list, avg_max_list, facecolor="#74b9ff")
    _draw_horline(axes[1][1])
    axes[1][1].plot(avg_non_t_list, avg_non_avg_list, color="#0984e3")
    axes[1][1].fill_between(avg_non_t_list, avg_non_min_list, avg_non_max_list, facecolor="#74b9ff")
    figure.text(0.01, 0.5, 'Rotation of spindle (degrees)', va='center', rotation='vertical')
