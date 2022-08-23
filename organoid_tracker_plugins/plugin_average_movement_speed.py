from typing import Tuple

import numpy
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import norm, lognorm

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
        resolution = experiment.images.resolution()
        for source, target in experiment.links.find_all_links():
            distances_um.append(source.distance_um(target, resolution))

    distances_um = numpy.array(distances_um)

    mean = numpy.mean(distances_um)
    stdev = numpy.std(distances_um)
    result = dialog.prompt_options("Average movement speed",
                                   f"The cells move {mean:.2f} ± {stdev:.2f} µm on average between time points.",
                                   option_1="Show full distribution",
                                   option_default=DefaultOption.OK)
    if result == 1:
        def draw_figure(figure: Figure):
            ax: Axes = figure.gca()
            ax.hist(distances_um, bins="scott", density=True)

            a, loc, scale = lognorm.fit(distances_um, loc=-1, scale=2)
            ax.text(0, 0, f"a={a:.2f}, loc={loc:.2f}, scale={scale:.2f}")

            x = numpy.linspace(0, 30, 1000)
            ax.plot(x, lognorm(a, loc=loc, scale=scale).pdf(x))

            ax.set_ylabel("Fraction")
            ax.set_xlabel("Distance (µm)")

        dialog.popup_figure(window.get_gui_experiment(), draw_figure)
