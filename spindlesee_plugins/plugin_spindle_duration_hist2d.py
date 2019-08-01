from typing import Dict, Any

from matplotlib.figure import Figure

from ai_track.gui import dialog
from ai_track.gui.window import Window
from . import plugin_spindle_markers


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Spindle-Histogram of angle changes...": lambda: _show_histogram(window),
    }


def _show_histogram(window: Window):
    experiment = window.get_experiment()
    time_resolution = experiment.images.resolution().time_point_interval_m

    x_values = list()
    y_values = list()
    for spindle in plugin_spindle_markers.find_all_spindles(experiment.links, experiment.connections):
        x_values.append((len(spindle.positions1) - 1) * time_resolution)
        y_values.append(spindle.get_orientation_change())

    def show_figure(figure: Figure):
        axes = figure.gca()
        _, _, _, histogram = axes.hist2d(x_values, y_values)
        figure.colorbar(histogram).set_label("Amount of spindles", rotation=270)

        axes.set_xlabel("Duration of spindle appearance (minutes)")
        axes.set_ylabel("Orientation change (degrees)")

    dialog.popup_figure(window.get_gui_experiment(), show_figure, size_cm=(7, 7))
