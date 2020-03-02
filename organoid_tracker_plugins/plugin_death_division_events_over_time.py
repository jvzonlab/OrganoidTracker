from typing import Dict, Any, List

import numpy
from numpy import ndarray
from matplotlib.figure import Figure

from organoid_tracker.core import UserError
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.linking import cell_division_finder
from organoid_tracker.linking_analysis import linking_markers
from organoid_tracker.util import mpl_helper


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Cell cycle-Death and division events//Graph-Amounts over time...":
            lambda: _view_noncumulative_births_and_deaths(window),
        "Graph//Cell cycle-Death and division events//Graph-Cumulative amounts over time...":
            lambda: _view_cumulative_births_and_deaths(window)
    }


def _view_noncumulative_births_and_deaths(window: Window):
    experiment = window.get_experiment()
    if not experiment.links.has_links():
        raise UserError("Linking data is missing", "The linking data is missing, so we cannot plot cell births and deaths.")
    resolution = experiment.images.resolution()

    births = list()  # Initialize array
    for cell_division in cell_division_finder.find_mothers(experiment.links):  # Iterate over all cell divisions
        births.append(cell_division.time_point_number() * resolution.time_point_interval_h)

    deaths = list()
    for cell_death in linking_markers.find_death_and_shed_positions(experiment.links):  # Iterate over all cell deaths
        deaths.append(cell_death.time_point_number() * resolution.time_point_interval_h)

    experiment_hours = experiment.positions.last_time_point_number() * resolution.time_point_interval_h
    bins = numpy.arange(0, experiment_hours, 1)

    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_noncumulative_figure(figure, births, deaths, bins))


def _draw_noncumulative_figure(figure: Figure, division_time_points: List[int], deaths_time_points: List[int],
                               bins: ndarray):
    """Draws a histogram. The lists contain the time point numbers at which divisions/deaths are occuring.
    For example, `division_time_points=[3,4,4,10]` means two divisions at t=4, one at t=3 and one at t=10."""
    axes = figure.gca()
    axes.set_title("Number of cell deaths and divisions over time")
    axes.hist(division_time_points, bins, label="Divisions", color=mpl_helper.HISTOGRAM_RED)
    axes.hist(deaths_time_points, bins, label="Deaths", color=mpl_helper.HISTOGRAM_BLUE)
    axes.set_xlabel("Time (h))")
    axes.set_ylabel("Amount per hour")
    axes.legend()


def _view_cumulative_births_and_deaths(window: Window):
    experiment = window.get_experiment()
    if not experiment.links.has_links():
        raise UserError("Linking data is missing", "The linking data is missing, so we cannot plot cell births and deaths.")

    cumulative_births = numpy.zeros(experiment.last_time_point_number(), dtype=numpy.int32)  # Initialize array
    for cell_division in cell_division_finder.find_mothers(experiment.links):  # Iterate over all cell divisions
        cumulative_births[cell_division.time_point_number():] += 1  # Add 1 to all time points after the cell division

    cumulative_deaths = numpy.zeros(experiment.last_time_point_number(), dtype=numpy.int32)  # Initialize array
    for cell_death in linking_markers.find_death_and_shed_positions(experiment.links):  # Iterate over all cell deaths
        cumulative_deaths[cell_death.time_point_number():] += 1  # Add 1 to all time points after the cell death

    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_cumulative_figure(
        figure, cumulative_births, cumulative_deaths))


def _draw_cumulative_figure(figure: Figure, divisions_by_time_point: ndarray, deaths_by_time_point: ndarray):
    axes = figure.gca()
    axes.set_title("Number of cell deaths and divisions over time")
    axes.plot(deaths_by_time_point, label="Deaths")
    axes.plot(divisions_by_time_point, label="Divisions")
    axes.set_xlabel("Time point")
    axes.set_ylabel("Amount (cumulative over time)")
    axes.legend()
