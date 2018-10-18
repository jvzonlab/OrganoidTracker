from typing import Dict, Any, Tuple

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


def _calculate_moving_average(x_values: ndarray, y_values: ndarray, window_size: int = 21) -> Tuple[ndarray, ndarray]:
    """Simply moving average calculating for the given x and y values."""
    extend = window_size / 2

    x_min = x_values.min() - 1
    x_max = x_values.max() + 2

    x_moving_average = numpy.arange(x_min, x_max + 1)
    y_moving_average = numpy.empty_like(x_moving_average)
    for i in range(len(x_moving_average)):
        # Construct a boolean area on which x values to use
        x = x_moving_average[i]
        used_y_values = y_values[(x_values >= x - extend) & (x_values <= x + extend)]
        y_moving_average[i] = used_y_values.mean()
    return x_moving_average, y_moving_average


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

    # Convert to numpy, get statistics
    previous_cycle_durations = numpy.array(previous_cycle_durations, dtype=numpy.int32)
    cycle_durations = numpy.array(cycle_durations, dtype=numpy.int32)
    plot_limit = cycle_durations.max() * 1.1

    window_size = 21
    x_moving_average, y_moving_average = _calculate_moving_average(previous_cycle_durations, cycle_durations,
                                                                   window_size=window_size)

    axes = figure.gca()
    axes.plot(numpy.arange(plot_limit), color="orange", label="$T_{mother} = T_{daughter}$ line")
    axes.plot(x_moving_average, y_moving_average, color="blue", linewidth=2,
              label=f"Moving average ({window_size} time points)")
    axes.scatter(x=previous_cycle_durations, y=cycle_durations, color="black", alpha=0.4, s=25, lw=0)
    axes.set_xlim(0, plot_limit)
    axes.set_ylim(0, plot_limit)
    axes.set_title("Length of mother cell cycle versus length of daughter cell cycle")
    axes.set_aspect('equal', adjustable='box')
    axes.set_xlabel("$T_{mother}$ (time points)")
    axes.set_ylabel("$T_{daughter}$ (time points)")
    axes.legend(loc="lower right")
