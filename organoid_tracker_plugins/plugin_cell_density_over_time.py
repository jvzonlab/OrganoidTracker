from typing import Tuple, Dict, Any

import numpy
from matplotlib.figure import Figure
from numpy import ndarray

from organoid_tracker.core import TimePoint, UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import cell_density_calculator


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
         "Graph//Cell cycle-Cell density//Average cell density over time...": lambda: _show_cell_density(window),
    }


def _show_cell_density(window: Window):
    experiment = window.get_experiment()

    times_h, densities_um1, densities_stdev_um1 = _get_all_average_densities(experiment)
    if len(times_h) == 0:
        raise UserError("Cell density", "Found no cell positions - cannot plot anything.")
    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_cell_density(figure, times_h, densities_um1,
                                                                                       densities_stdev_um1))


def _draw_cell_density(figure: Figure, times_h: ndarray, densities_um1: ndarray, densities_stdev_um1: ndarray):
    axes = figure.gca()
    axes.plot(times_h, densities_um1)
    axes.fill_between(times_h, densities_um1 - densities_stdev_um1, densities_um1 + densities_stdev_um1,
                      color="lightblue")
    axes.set_xlabel("Time (h)")
    axes.set_ylabel("Density (mm$^{-1}$)")
    axes.set_title("Average cell density over time")
    axes.set_ylim(0, max(densities_um1) * 1.2)


def _get_all_average_densities(experiment: Experiment) -> Tuple[ndarray, ndarray, ndarray]:
    """Returns three lists: time (hours) vs average density (mm^-1) and its standard deviation."""
    resolution = experiment.images.resolution()
    positions = experiment.positions

    times_h = []
    densities_mm1 = []
    densities_stdev_mm1 = []
    for time_point in experiment.time_points():
        density_avg_mm1, density_stdev_mm1 = _get_average_density_mm1(positions, time_point, resolution)
        if density_avg_mm1 > 0:
            times_h.append(time_point.time_point_number() * resolution.time_point_interval_h)
            densities_mm1.append(density_avg_mm1)
            densities_stdev_mm1.append(density_stdev_mm1)

    return numpy.array(times_h), numpy.array(densities_mm1), numpy.array(densities_stdev_mm1)


def _get_average_density_mm1(positions: PositionCollection, time_point: TimePoint, resolution: ImageResolution
                             ) -> Tuple[float, float]:
    """Gets the average density and the std dev for the whole organoid at the given time point."""
    densities = list()
    for position in positions.of_time_point(time_point):
        density = cell_density_calculator.get_density_mm1(positions.of_time_point(time_point), position, resolution)

        densities.append(density)

    if len(densities) == 0:
        return 0, 0
    return float(numpy.mean(densities)), float(numpy.std(densities, ddof=1))
