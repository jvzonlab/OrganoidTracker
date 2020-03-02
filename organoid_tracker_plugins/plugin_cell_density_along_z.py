from typing import Dict, Any, Tuple

import numpy
from numpy import ndarray
from matplotlib.figure import Figure

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.linking_analysis import linking_markers
from organoid_tracker.position_analysis import cell_density_calculator


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
         "Graph//Cell cycle-Cell density//Average cell density over z...": lambda: _show_cell_density(window),
    }


def _show_cell_density(window: Window):
    densities = _get_average_cell_densities(window.get_experiment())
    z_positions_deaths = _get_z_positions_of_deaths(window.get_experiment())
    if len(z_positions_deaths) == 0:
        raise UserError("Cell density", "Found no positions - cannot plot anything.")
    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_cell_densities(figure, densities, z_positions_deaths))


def _get_z_positions_of_deaths(experiment: Experiment) -> Dict[int, int]:
    death_counts_by_z = dict()
    for position in linking_markers.find_death_and_shed_positions(experiment.links):
        z = round(position.z)
        if z in death_counts_by_z:
            death_counts_by_z[z] += 1
        else:
            death_counts_by_z[z] = 1
    return death_counts_by_z


def _get_average_cell_densities(experiment: Experiment) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """Gets the average cell density at each z layer. Returned as z (px), count, density (mm-1), stdev (mm-1)"""
    densities_by_z = dict()
    positions = experiment.positions
    resolution = experiment.images.resolution()

    for time_point in experiment.time_points():
        positions_of_time_point = positions.of_time_point(time_point)
        for position in positions_of_time_point:
            density = cell_density_calculator.get_density_mm1(positions_of_time_point, around=position, resolution=resolution)
            z = round(position.z)
            densities_at_z = densities_by_z.get(z)
            if densities_at_z is None:
                densities_at_z = list()
                densities_by_z[z] = densities_at_z
            densities_at_z.append(density)

    z_positions, counts, densities_mean_mm1, densities_stdev_mm1 = list(), list(), list(), list()
    for z, densities in densities_by_z.items():
        z_positions.append(z)
        counts.append(len(densities))
        densities_mean_mm1.append(float(numpy.mean(densities)))
        densities_stdev_mm1.append(float(numpy.std(densities, ddof=1)))
    return numpy.array(z_positions), numpy.array(counts, dtype=numpy.uint32), numpy.array(densities_mean_mm1), numpy.array(densities_stdev_mm1)


def _draw_cell_densities(figure: Figure, densities_by_z: Tuple[ndarray, ndarray, ndarray, ndarray],
                         z_positions_of_deaths: Dict[int, int]):
    z_positions_um, counts, densities_mean_mm1, densities_stdev_mm1 = densities_by_z

    axes = figure.gca()
    axes.set_title("Densities and standard deviation versus depth in the organoid")
    axes.hist(z_positions_of_deaths)
    twin_axes = axes.twinx()
    twin_axes.plot(z_positions_um, densities_mean_mm1)
    twin_axes.fill_between(z_positions_um, densities_mean_mm1 - densities_stdev_mm1, densities_mean_mm1 + densities_stdev_mm1, alpha=0.5)

    axes.set_xlabel("Z (μm)")
    axes.set_ylabel("Cell deaths")
    twin_axes.set_ylabel("Density (mm$^{-1}$)")
