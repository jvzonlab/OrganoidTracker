from typing import Dict, Any

import numpy
from matplotlib.figure import Figure
from networkx import Graph
from scipy.stats import linregress

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

    slope, intercept, r_value, p_value, std_err = linregress(x=previous_cycle_durations, y=cycle_durations)
    r_squared = r_value ** 2

    axes = figure.gca()
    axes.plot([0, plot_limit], [intercept, intercept + slope * plot_limit], color="gray")  # Regression line
    axes.scatter(x=previous_cycle_durations, y=cycle_durations, color="black")  # The data
    axes.text(plot_limit * 0.05, intercept - 5,
              f'$T_{{daughter}} = {slope:.3} \cdot T_{{mother}} + {intercept:.3}$\n$(R^2 = {r_squared:.4})$',
              va="top")
    axes.set_xlim(0, plot_limit)
    axes.set_ylim(0, plot_limit)
    axes.set_title("Length of mother cell cycle versus length of daughter cell cycle")
    axes.set_aspect('equal', adjustable='box')
    axes.set_xlabel("$T_{mother}$")
    axes.set_ylabel("$T_{daughter}$")
