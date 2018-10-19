from typing import Dict, Any, Tuple, List

import numpy
from matplotlib.figure import Figure
from networkx import Graph
from numpy import ndarray

from autotrack.core import UserError
from autotrack.gui import Window, dialog
from autotrack.linking import mother_finder, cell_cycle


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph/Cell cycle-Lengths of cell cycles...": lambda: _view_cell_cycle_length(window)
    }


def _view_cell_cycle_length(window: Window):
    experiment = window.get_experiment()
    links = experiment.links.get_baseline_else_scratch()
    if links is None:
        raise UserError("No links specified", "No links were loaded. Cannot plot anything.")

    dialog.popup_figure(experiment.name, lambda fig: _draw_cell_cycle_length(fig, links))


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


def _draw_cell_cycle_length(figure: Figure, links: Graph):
    previous_cycle_durations = list()
    cycle_durations = list()

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

    if len(cycle_durations) == 0:
        raise UserError("No cell cycles found", "The linking data contains no sequences of two complete cell cycles. "
                                                "Therefore, we cannot plot anything.")

    # Convert to numpy, get statistics
    previous_cycle_durations = numpy.array(previous_cycle_durations, dtype=numpy.int32)
    cycle_durations = numpy.array(cycle_durations, dtype=numpy.int32)
    plot_limit = cycle_durations.max() * 1.1

    window_size = 11
    x_moving_average, y_moving_average, y_moving_average_stdev \
        = _calculate_moving_average(previous_cycle_durations, cycle_durations, window_size=window_size)

    axes = figure.gca()
    axes.plot(numpy.arange(plot_limit), color="orange", label="$T_{mother} = T_{daughter}$ line")
    axes.plot(x_moving_average, y_moving_average, color="blue", linewidth=2,
              label=f"Moving average ({window_size} time points)")
    axes.fill_between(x_moving_average, y_moving_average - y_moving_average_stdev,
                      y_moving_average + y_moving_average_stdev, color="blue", alpha=0.2)
    axes.scatter(x=previous_cycle_durations, y=cycle_durations, color="black", alpha=0.4, s=25, lw=0)
    axes.set_xlim(0, plot_limit)
    axes.set_ylim(0, plot_limit)
    axes.set_title("Length of mother cell cycle versus length of daughter cell cycle")
    axes.set_aspect('equal', adjustable='box')
    axes.set_xlabel("$T_{mother}$ (time points)")
    axes.set_ylabel("$T_{daughter}$ (time points)")
    axes.legend(loc="lower right")
