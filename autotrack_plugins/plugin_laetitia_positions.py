import os
import re
from os import path
from typing import Dict, Any, AbstractSet

import numpy

from autotrack.core import UserError, TimePoint
from autotrack.core.experiment import Experiment
from autotrack.core.particles import Particle, ParticleCollection
from autotrack.gui import Window, dialog

Z_OVERSCALED = 6.0
EXPECTED_Z_LAYERS = 32
TIME_POINT_FROM_FILE_NAME = re.compile("t(\d+)")


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "File/Import-Import positions in Laetitia's format...": lambda: _import_laetitia_positions(window),
        "File/Export-Export positions in Laetitia's format...": lambda: _export_laetitia_positions(window)
    }


def _import_laetitia_positions(window: Window):
    z_offset = _get_z_offset(window.get_experiment())

    if not dialog.popup_message_cancellable("Instructions", "Choose the directory containing the *.npy or *.txt files."):
        return
    directory = dialog.prompt_directory("Choose directory with positions files...")
    if directory is None:
        return

    window.get_experiment().remove_all_particles()
    for file_name in os.listdir(directory):
        _import_file(window.get_experiment(), directory, file_name, z_offset)

    window.refresh()


def _export_laetitia_positions(window: Window):
    experiment = window.get_experiment()
    z_offset = _get_z_offset(experiment)
    directory = dialog.prompt_directory("Choose a directory to export the positions files to...")
    if directory is None:
        return
    overwrite = False

    for time_point in experiment.time_points():
        file_name = experiment.name.get_save_name() + "t" + "{:03}".format(time_point.time_point_number()) + ".npy"
        file_path = path.join(directory, file_name)
        if path.exists(file_path) and not overwrite:
            if dialog.prompt_confirmation("Directory not empty", "We're about to overwrite files, including "
                                                                 + file_name + ". Is that OK?"
                                          "\n\nExperiment: " + str(experiment.name)
                                                                 + "\nDirectory: " + directory):
                overwrite = True
            else:
                return
        _export_file(experiment.particles.of_time_point(time_point), file_path, z_offset)


def _get_z_offset(experiment: Experiment) -> int:
    """Laetitia adds black xy planes until the image has the expected number of z layers. We don't do that, so we need
    to correct the particle z for this.
    """
    return int((_get_image_z_size(experiment) - EXPECTED_Z_LAYERS) / 2)


def _get_image_z_size(experiment: Experiment) -> int:
    """Gets the number of Z layers in the images."""
    try:
        first_time_point_number = experiment.first_time_point_number()
        image_stack = experiment.get_image_stack(experiment.get_time_point(first_time_point_number))
        if image_stack is not None:
            return len(image_stack)
    except ValueError:
        pass

    raise UserError("No images have been loaded", "Please load some images first. The size of the images needs to"
                                                  " be known in order to correctly handle Laetitia's file format.")


def _import_file(experiment: Experiment, directory: str, file_name: str, z_offset: int):
    match = re.search(TIME_POINT_FROM_FILE_NAME, file_name)
    if match is None:
        return

    time_point_number = int(match.group(1))  # Safe, as this group contains only numbers

    if file_name.endswith(".txt"):
        coords = numpy.loadtxt(path.join(directory, file_name))
    else:
        coords = numpy.load(path.join(directory, file_name))

    # Add new cells to the time point
    for row in range(len(coords)):
        particle = Particle(coords[row, 2], coords[row, 1], (coords[row, 0] / Z_OVERSCALED) + z_offset)

        experiment.particles.add(particle.with_time_point_number(time_point_number))


def _export_file(particles: AbstractSet[Particle], file_path: str, z_offset: int):
    if len(particles) == 0:
        return
    array = numpy.empty((len(particles), 3), dtype=numpy.int64)

    row = 0
    for particle in particles:
        array[row, 2] = round(particle.x)
        array[row, 1] = round(particle.y)
        array[row, 0] = round((particle.z - z_offset) * Z_OVERSCALED)
        row += 1

    numpy.save(file_path, array)
