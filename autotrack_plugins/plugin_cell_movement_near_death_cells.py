
from typing import Any, Dict

import numpy
from matplotlib.figure import Figure
from numpy import ndarray

from autotrack.core.experiment import Experiment
from autotrack.core.particles import ParticleCollection, Particle
from autotrack.core.resolution import ImageResolution
from autotrack.gui import dialog
from autotrack.gui.window import Window
from autotrack.linking import nearby_particle_finder
from autotrack.linking_analysis import linking_markers, particle_connection_finder

_STEPS_BACK = 15


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
         "Graph//Cell deaths-Cells nearby cell deaths...": lambda: _nearby_cell_movement(window),
    }


def _nearby_cell_movement(window: Window):
    experiment = window.get_experiment()
    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_figure(experiment, figure))


def _draw_figure(experiment: Experiment, figure: Figure):
    links = experiment.links
    particles = experiment.particles
    resolution = experiment.image_resolution()
    axes = figure.gca()
    axes.set_xlim(_STEPS_BACK * resolution.time_point_interval_m, resolution.time_point_interval_m)
    axes.set_xlabel("Minutes before death")
    axes.set_ylabel("Average distance to two nearest cells (μm)")

    dead_cells = list(linking_markers.find_dead_particles(links))
    previous_times = numpy.array(range(_STEPS_BACK + 1)) * resolution.time_point_interval_m
    all_distances = numpy.full((len(dead_cells), len(previous_times)), fill_value=numpy.nan, dtype=numpy.float32)

    for i, dead_cell in enumerate(dead_cells):
        previous_positions = particle_connection_finder.find_previous_positions(dead_cell, links, steps_back=_STEPS_BACK)
        if previous_positions is None:
            continue

        previous_distances = [_get_average_distance_to_nearest_two_cells(particles, pos, resolution)
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


def _get_average_distance_to_nearest_two_cells(all_particles: ParticleCollection, around: Particle, resolution: ImageResolution) -> float:
    particles = all_particles.of_time_point(around.time_point())
    closest_particles = nearby_particle_finder.find_closest_n_particles(particles, around, max_amount=2)
    distance1 = closest_particles.pop().distance_um(around, resolution)
    distance2 = closest_particles.pop().distance_um(around, resolution)
    return (distance1 + distance2) / 2
