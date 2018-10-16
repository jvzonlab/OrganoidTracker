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
        "Graph/Cell cycle-View cell cycle lengths...": lambda: _view_cell_cycle_length(window)
    }


def _view_cell_cycle_length(window: Window):
    experiment = window.get_experiment()
    links = experiment.particle_links() if experiment.particle_links() is not None else experiment.particle_links_scratch()
    if links is None:
        raise UserError("No links specified", "No links were loaded. Cannot plot anything.")

    dialog.popup_figure(experiment.name, lambda fig: _draw_cell_cycle_length(fig, links))


def _draw_cell_cycle_length(figure: Figure, links: Graph):
    division_times = list()
    next_division_times = list()

    # Find all families and their next division
    for family in mother_finder.find_families(links):
        division_time = family.mother.time_point_number()
        for daughter in family.daughters:
            next_division = cell_cycle.get_next_division(links, daughter)
            if next_division is None:
                continue
            division_times.append(division_time)
            next_division_times.append(next_division.mother.time_point_number())

    # Convert to numpy, get statistics
    division_times = numpy.array(division_times, dtype=numpy.int32)
    next_division_times = numpy.array(next_division_times, dtype=numpy.int32)
    plot_limit = next_division_times.max() * 1.1

    slope, intercept, r_value, p_value, std_err = linregress(x=division_times, y=next_division_times)
    r_squared = r_value ** 2

    axes = figure.gca()
    axes.plot([0, plot_limit], [intercept, intercept + slope * plot_limit], color="gray")  # Regression line
    axes.scatter(x=division_times, y=next_division_times, color="black")  # The data
    axes.text(plot_limit * 0.05, intercept - 5,
              f'$T_{{daughter}} = {slope:.3} \cdot T_{{mother}} + {intercept:.3}$\n$(R^2 = {r_squared:.4})$',
              va="top")
    axes.set_xlim(0, plot_limit)
    axes.set_ylim(0, plot_limit)
    axes.set_title("Time of division versus time of next division")
    axes.set_aspect('equal', adjustable='box')
    axes.set_xlabel("$T_{mother}$")
    axes.set_ylabel("$T_{daughter}$")
