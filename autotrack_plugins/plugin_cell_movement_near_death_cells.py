
from typing import Any, Dict

import numpy
from matplotlib.figure import Figure
from numpy import ndarray

from autotrack.core.experiment import Experiment
from autotrack.core.position_collection import PositionCollection
from autotrack.core.position import Position
from autotrack.core.resolution import ImageResolution
from autotrack.gui import dialog
from autotrack.gui.window import Window
from autotrack.linking import nearby_position_finder
from autotrack.linking_analysis import linking_markers, position_connection_finder

_STEPS_BACK = 15


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
         "Graph//Cell cycle-Cell deaths//Distance to nearby cells...": lambda: _nearby_cell_movement(window),
    }


def _nearby_cell_movement(window: Window):
    experiment = window.get_experiment()
    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_figure(experiment, figure))


def _draw_figure(experiment: Experiment, figure: Figure):
    links = experiment.links
    positions = experiment.positions
    resolution = experiment.images.resolution()
    axes = figure.gca()
    axes.set_xlim(_STEPS_BACK * resolution.time_point_interval_m, resolution.time_point_interval_m)
    axes.set_xlabel("Minutes before death")
    axes.set_ylabel("Average distance to two nearest cells (μm)")

    dead_cells = list(linking_markers.find_death_and_shed_positions(links))
    previous_times = numpy.array(range(_STEPS_BACK + 1)) * resolution.time_point_interval_m
    all_distances = numpy.full((len(dead_cells), len(previous_times)), fill_value=numpy.nan, dtype=numpy.float32)

    for i, dead_cell in enumerate(dead_cells):
        previous_positions = position_connection_finder.find_previous_positions(dead_cell, links, steps_back=_STEPS_BACK)
        if previous_positions is None:
            continue

        previous_distances = [_get_average_distance_to_nearest_two_cells(positions, pos, resolution)
                              for pos in previous_positions]
        all_distances[i] = previous_distances
        axes.plot(previous_times, previous_distances, color="black", alpha=0.3)

    if len(dead_cells) > 0:
        mean = numpy.nanmean(all_distances, 0)
        stdev = numpy.nanstd(all_distances, 0, ddof=1)
        axes.plot(previous_times, mean, color="blue", linewidth=3, label="Average")
        axes.fill_between(previous_times, mean - stdev, mean + stdev, color="blue", alpha=0.2)
        axes.legend()
    else:
        axes.text(0.5, 0.5, f"No cells were found with both a death marker and {_STEPS_BACK} time points of history.",
                  horizontalalignment='center', verticalalignment = 'center', transform = axes.transAxes)


def _get_average_distance_to_nearest_two_cells(all_positions: PositionCollection, around: Position, resolution: ImageResolution) -> float:
    positions = all_positions.of_time_point(around.time_point())
    closest_positions = nearby_position_finder.find_closest_n_positions(positions, around=around, max_amount=2,
                                                                        resolution=resolution)
    distance1 = closest_positions.pop().distance_um(around, resolution)
    distance2 = closest_positions.pop().distance_um(around, resolution)
    return (distance1 + distance2) / 2
