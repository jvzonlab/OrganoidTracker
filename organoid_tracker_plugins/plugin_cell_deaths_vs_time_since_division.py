from typing import Dict, Any, List, Tuple

import numpy
from matplotlib.figure import Figure

from organoid_tracker.core import TimePoint, UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.linking import cell_division_finder
from organoid_tracker.linking_analysis import cell_nearby_death_counter


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Cell cycle-Cell deaths//Graph-Number of neighbor deaths versus time since division...":
            lambda: _show_number_of_neighbor_deaths_vs_time_since_division(window)
    }


def _show_number_of_neighbor_deaths_vs_time_since_division(window: Window):
    experiment = window.get_experiment()
    positions = experiment.positions

    last_time_point_number = positions.last_time_point_number()
    if last_time_point_number is None:
        raise UserError("No positions loaded", "No positions are loaded - cannot plot anything.")
    last_time_point = TimePoint(last_time_point_number)
    result = _get_hours_since_division_vs_neighbor_deaths(
            experiment, last_time_point)
    hours_since_last_division_list = result[0]
    number_of_neighbor_deaths_list = result[1]
    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _show_correlation_plot(figure,
            hours_since_last_division_list, number_of_neighbor_deaths_list))


def _get_hours_since_division_vs_neighbor_deaths(experiment: Experiment, time_point: TimePoint) -> Tuple[List[float], List[float]]:
    links = experiment.links
    position_data = experiment.position_data
    resolution = experiment.images.resolution()
    time_point_number = time_point.time_point_number()

    deaths_nearby_tracks = cell_nearby_death_counter.NearbyDeaths(links, position_data, resolution)

    hours_since_last_division_list = []
    number_of_neighbor_deaths_list = []

    for position in experiment.positions.of_time_point(time_point):

        # Calculate the number of hours since the last division
        previous_division = cell_division_finder.get_previous_division(links, position)
        if previous_division is None:
            continue
        hours_since_last_division = (time_point_number - previous_division.mother.time_point_number()) \
                               * resolution.time_point_interval_h

        # Count the number of neighbor deaths
        number_of_neighbor_deaths = deaths_nearby_tracks.count_nearby_deaths_in_past(links, position)

        hours_since_last_division_list.append(hours_since_last_division)
        number_of_neighbor_deaths_list.append(number_of_neighbor_deaths)

    return hours_since_last_division_list, number_of_neighbor_deaths_list


def _show_correlation_plot(figure: Figure, hours_since_last_division_list: List[float], number_of_neighbor_deaths_list: List[float]):
    axes = figure.gca()
    x_edges = numpy.arange(0, max(hours_since_last_division_list) + 5, 5)
    y_edges = numpy.arange(max(number_of_neighbor_deaths_list) + 2)
    _, _, _, image = axes.hist2d(hours_since_last_division_list, number_of_neighbor_deaths_list,
                                        bins=[x_edges, y_edges - 0.5], cmap="gist_stern")
    axes.set_xlabel("Time since last division (h)")
    axes.set_ylabel("Number of deaths seen in neighborhood")
    axes.set_xticks(x_edges)
    axes.set_yticks(y_edges)
    axes.set_facecolor("black")
    figure.colorbar(image).set_label("Number of cells")
