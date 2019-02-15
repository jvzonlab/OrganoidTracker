from typing import Dict, Any

import numpy
from numpy import ndarray
from matplotlib.figure import Figure

from autotrack.core import UserError
from autotrack.gui import dialog
from autotrack.gui.window import Window
from autotrack.linking import cell_division_finder
from autotrack.linking_analysis import linking_markers


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Cell cycle-Cell births and deaths over time...": lambda: _view_births_and_deaths(window)
    }


def _view_births_and_deaths(window: Window):
    experiment = window.get_experiment()
    if not experiment.links.has_links():
        raise UserError("Linking data is missing", "The linking data is missing, so we cannot plot cell births and deaths.")

    cumulative_births = numpy.zeros(experiment.last_time_point_number(), dtype=numpy.int32)  # Initialize array
    for cell_division in cell_division_finder.find_mothers(experiment.links):  # Iterate over all cell divisions
        cumulative_births[cell_division.time_point_number():] += 1  # Add 1 to all time points after the cell division

    cumulative_deaths = numpy.zeros(experiment.last_time_point_number(), dtype=numpy.int32)  # Initialize array
    for cell_death in linking_markers.find_death_and_shed_positions(experiment.links):  # Iterate over all cell deaths
        cumulative_deaths[cell_death.time_point_number():] += 1  # Add 1 to all time points after the cell death

    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_figure(figure, cumulative_births, cumulative_deaths))


def _draw_figure(figure: Figure, cumulative_births: ndarray, cumulative_deaths: ndarray):
    axes = figure.gca()
    axes.set_title("Number of cell deaths and divisions over time")
    axes.plot(cumulative_deaths, label="Deaths")
    axes.plot(cumulative_births, label="Divisions")
    axes.set_xlabel("Time point")
    axes.set_ylabel("Amount (cumulative over time)")
    axes.legend()
