from typing import Dict, Any, List, Optional, Iterable

import numpy
from matplotlib.figure import Figure

from autotrack.core.connections import Connections
from autotrack.core.links import LinkingTrack, Links
from autotrack.core.position import Position
from autotrack.gui import dialog
from autotrack.gui.window import Window
from . import plugin_spindle_markers


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Spindle-Print spindles": lambda: _print_spindles(window)
    }


def _show_midbody_spindle_lenghts(window: Window):
    experiment = window.get_experiment()
    lengths_no_midbody = []
    lengths_midbody = []
    t_res = experiment.images.resolution().time_point_interval_m
    print("---")
    for spindle in plugin_spindle_markers.find_all_spindles(experiment.links, experiment.connections):
        if len(spindle.midbody) > 0:
            lengths_midbody.append(len(spindle.positions1) * t_res)
            midbody_first = spindle.midbody[0]
            print(
                f"/goto {midbody_first.x:.1f} {midbody_first.y:.1f} {midbody_first.z:.1f} {midbody_first.time_point_number()}")
        else:
            lengths_no_midbody.append(len(spindle.positions2) * t_res)
    print(numpy.mean(lengths_no_midbody), numpy.std(lengths_no_midbody))
    print("0")
    print(numpy.mean(lengths_midbody), numpy.std(lengths_midbody))


def _print_spindles(window: Window):
    experiment = window.get_experiment()
    for spindle in plugin_spindle_markers.find_all_spindles(experiment.links, experiment.connections):
        pos = spindle.positions1[0]
        print(f"/goto {pos.x:.2f} {pos.y:.2f} {pos.z:.2f} {pos.time_point_number()}")


def _show_distance_graph(window: Window):
    experiment = window.get_experiment()
    spindles = list(plugin_spindle_markers.find_all_spindles(experiment.links, experiment.connections))
    resolution = experiment.images.resolution()

    def draw_function(figure: Figure):
        axes = figure.gca()
        for spindle in spindles:
            distances = [val[0].distance_um(val[1], resolution) for val in zip(spindle.midbody, spindle.midbody_edge)]
            times = numpy.arange(len(distances)) * resolution.time_point_interval_m
            axes.plot(times, distances)
            axes.set_xlabel("Time since division (minutes)")
            axes.set_ylabel("Distance to edge of organoid (μm)")

    dialog.popup_figure(window.get_gui_experiment(), draw_function)
