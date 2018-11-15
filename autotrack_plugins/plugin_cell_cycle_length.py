from typing import Dict, Any, Tuple, Optional

import numpy
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from networkx import Graph
from numpy import ndarray

from autotrack.core import UserError
from autotrack.core.experiment import Experiment
from autotrack.core.particles import Particle
from autotrack.core.score import Family
from autotrack.gui import Window, dialog
from autotrack.linking import mother_finder, cell_cycle
from autotrack.linking_analysis import cell_fates
from autotrack.linking_analysis.cell_fates import CellFate


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph/Cell cycle-Cell cycle lengths of two generations...": lambda: _view_cell_cycle_length(window)
    }


def _view_cell_cycle_length(window: Window):
    experiment = window.get_experiment()
    links = experiment.links.graph
    if links is None:
        raise UserError("No links specified", "No links were loaded. Cannot plot anything.")

    third_variable_getter = _ThirdVar()
    try:
        time_point_duration_h = experiment.image_resolution().time_point_interval_m / 60
    except ValueError:
        raise UserError("No resolution set", "The resolution of the images was not set. Cannot plot anything.")

    dialog.popup_figure(experiment.name, lambda fig: _draw_cell_cycle_length(fig, links, time_point_duration_h,
                                                                             third_variable_getter))


class _ThirdVar:
    def get_number(self, daughter: Particle, next_division: Family):
        return 1

    def show_average(self) -> bool:
        return False

    def get_cmap(self) -> str:
        return "Greys"

    def get_colobar_label(self) -> Optional[str]:
        """If this is None, then no color or color bar is used: the the Third Variable will essentiaally be ignored."""
        return None


class _CellCryptPosVar(_ThirdVar):

    experiment: Experiment
    links: Graph

    def __init__(self, experiment: Experiment, links: Graph):
        self.experiment = experiment
        self.links = links

    def get_number(self, daughter: Particle, next_division: Family) -> float:
        path = self.experiment.paths.of_time_point(daughter.time_point())
        if path is None:
            return 0
        return path.get_path_position_2d(daughter)

    def show_average(self) -> bool:
        return False

    def get_colobar_label(self) -> Optional[str]:
        return "Crypt axis position of division (px)"


class _CellFateVar(_ThirdVar):
    experiment: Experiment
    links: Graph

    def __init__(self, experiment: Experiment, links: Graph):
        self.experiment = experiment
        self.links = links

    def get_number(self, daughter: Particle, next_division: Family) -> float:
        combined_fate = None
        for next in next_division.daughters:
            cell_fate = cell_fates.get_fate(self.experiment, self.links, next)
            if combined_fate is None:
                combined_fate = cell_fate
                continue
            if combined_fate == cell_fate:
                continue
            combined_fate = CellFate.UNKNOWN

        if combined_fate == CellFate.WILL_DIVIDE:
            return 1
        elif combined_fate == CellFate.NON_DIVIDING:
            return 0
        else:
            return 0.5

    def show_average(self) -> bool:
        return False

    def get_colobar_label(self) -> str:
        return "Cell fate after next division (1 = divides)"


def _calculate_moving_average(x_values: ndarray, y_values: ndarray, window_size: int = 21
                              ) -> Tuple[ndarray, ndarray, ndarray]:
    """Simply moving average calculating for the given x and y values. Returns x, mean y and standard deviation y
    values."""
    extend = window_size / 2

    x_min = x_values.min()
    x_max = x_values.max()

    x_moving_average = list()
    y_moving_average = list()
    y_moving_average_stdev = list()
    for x in range(int(x_min), int(x_max) + 1):
        # Construct a boolean area on which x values to use
        used_y_values = y_values[(x_values >= x - extend) & (x_values <= x + extend)]

        if len(used_y_values) < 2:
            continue
        x_moving_average.append(x)
        y_moving_average.append(used_y_values.mean())
        y_moving_average_stdev.append(numpy.std(used_y_values, ddof=1))

    return numpy.array(x_moving_average, dtype=numpy.float32),\
           numpy.array(y_moving_average, dtype=numpy.float32),\
           numpy.array(y_moving_average_stdev, dtype=numpy.float32)


def _draw_cell_cycle_length(figure: Figure, links: Graph, time_point_duration_h: float,
                            third_variable_getter: _ThirdVar):
    previous_cycle_durations = list()
    cycle_durations = list()
    third_variables = list()  # Used for color, can be z position

    # Find all families and their next division
    for family in mother_finder.find_families(links):
        previous_cycle_duration = cell_cycle.get_age(links, family.mother)
        if previous_cycle_duration is None:
            continue

        division_time = family.mother.time_point_number()
        for daughter in family.daughters:
            next_division = cell_cycle.get_next_division(links, daughter)
            if next_division is None:
                continue
            cycle_duration = next_division.mother.time_point_number() - division_time
            previous_cycle_durations.append(previous_cycle_duration)
            cycle_durations.append(cycle_duration)
            third_variables.append(third_variable_getter.get_number(daughter, next_division))

    if len(cycle_durations) == 0:
        raise UserError("No cell cycles found", "The linking data contains no sequences of two complete cell cycles. "
                                                "Therefore, we cannot plot anything.")

    # Convert to numpy, get statistics
    previous_cycle_durations = numpy.array(previous_cycle_durations, dtype=numpy.int32) * time_point_duration_h
    cycle_durations = numpy.array(cycle_durations, dtype=numpy.int32) * time_point_duration_h
    third_variables = numpy.array(third_variables, dtype=numpy.float32)
    plot_start = min(cycle_durations.min(), previous_cycle_durations.min()) / 2
    plot_limit = max(previous_cycle_durations.max(), cycle_durations.max()) * 1.1

    window_size = 11
    x_moving_average, y_moving_average, y_moving_average_stdev \
        = _calculate_moving_average(previous_cycle_durations, cycle_durations, window_size=window_size)

    axes = figure.gca()
    axes.plot(numpy.arange(plot_start, plot_limit), numpy.arange(plot_start, plot_limit), color="orange",
              label="Equal durations line")
    if third_variable_getter.show_average():
        axes.plot(x_moving_average, y_moving_average, color="blue", linewidth=2,
                  label=f"Moving average ({window_size} time points)")
        axes.fill_between(x_moving_average, y_moving_average - y_moving_average_stdev,
                          y_moving_average + y_moving_average_stdev, color="blue", alpha=0.2)

    if third_variable_getter.get_colobar_label() is not None:
        scatterplot = axes.scatter(x=previous_cycle_durations, y=cycle_durations, c=third_variables, s=25, lw=1,
                                   cmap=third_variable_getter.get_cmap(), edgecolors="black")
        divider = make_axes_locatable(axes)
        axes_on_right = divider.append_axes("right", size="5%", pad=0.1)
        figure.colorbar(scatterplot, cax=axes_on_right).set_label(third_variable_getter.get_colobar_label())
    else:
        axes.scatter(x=previous_cycle_durations, y=cycle_durations, c="black", s=9, lw=0, alpha=0.7)
    axes.set_xlim(plot_start, plot_limit)
    axes.set_ylim(plot_start, plot_limit)
    axes.set_aspect('equal', adjustable='box')
    axes.set_xlabel("Duration of cell cycle of mother (h)")
    axes.set_ylabel("Duration of cell cycle of daughter (h)")
    axes.legend(loc="lower right")
