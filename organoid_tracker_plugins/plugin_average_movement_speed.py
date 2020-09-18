from typing import Tuple

import numpy
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import dialog
from organoid_tracker.gui.dialog import DefaultOption
from organoid_tracker.gui.window import Window


def get_menu_items(window: Window):
    return {
        "Graph//Over space and time-Cell movement between time points...":
            lambda: _prompt_average_movement_speed(window)
    }



def _prompt_average_movement_speed(window: Window):
    # Collect all distances, switch to 2D
    distances_um = []
    for experiment in window.get_active_experiments():
        distances_um.append(_get_average_movement_speed(experiment))
    distances_um = numpy.array(distances_um).flatten()

    mean = numpy.mean(distances_um)
    stdev = numpy.std(distances_um)
    result = dialog.prompt_options("Average movement speed",
                                   f"The cells move {mean:.2f} ± {stdev:.2f} µm on average between time points.",
                                   option_1="Show full distribution",
                                   option_default=DefaultOption.OK)
    if result == 1:
        def draw_figure(figure: Figure):
            ax: Axes = figure.gca()
            ax.hist(distances_um, bins=20)
            ax.set_ylabel("Frequency")
            ax.set_xlabel("Distance (µm)")

        dialog.popup_figure(window.get_gui_experiment(), draw_figure)


def _get_average_movement_speed(experiment: Experiment) -> ndarray:
    resolution = experiment.images.resolution()
    distances_um = list()
    for source, target in experiment.links.find_all_links():
        distance_um = source.distance_um(target, resolution)
        distances_um.append(distance_um)

    return numpy.array(distances_um, dtype=numpy.float64)