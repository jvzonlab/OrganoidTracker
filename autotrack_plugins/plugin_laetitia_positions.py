import re
from typing import Dict, List, Any

import numpy

from core import Particle, Experiment, UserError
from gui import Window, dialog
from os import path
import os

Z_OVERSCALED = 6.0
EXPECTED_Z_LAYERS = 32
TIME_POINT_FROM_FILE_NAME = re.compile("t(\d+)")


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "File/Import-Import Laetitia's positions...": lambda: _import_laetitia_positions(window)
    }


def _import_laetitia_positions(window: Window):
    # If the images have fewer layers than expected, then Laetitia's adds black layers to the image. We need to correct
    # the particle z position for those added layers.
    z_offset = int((_get_image_z_size(window.get_experiment()) - EXPECTED_Z_LAYERS) / 2)

    if not dialog.popup_message_cancellable("Instructions", "Choose the directory containing the *.npy or *.txt files."):
        return
    directory = dialog.prompt_directory("Choose directory with positions files...")
    if directory is None:
        return

    for file_name in os.listdir(directory):
        _import_file(window.get_experiment(), directory, file_name, z_offset)

    window.refresh()


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
                                                  " be known in order to correctly read Laetitia's file format.")


def _import_file(experiment: Experiment, directory: str, file_name: str, z_offset: int):
    match = re.search(TIME_POINT_FROM_FILE_NAME, file_name)
    if match is None:
        return

    time_point_number = int(match.group(1))  # Safe, as this group contains only numbers
    time_point = experiment.get_or_add_time_point(time_point_number)
    experiment.remove_particles(time_point)  # Remove existing cells from this time point

    if file_name.endswith(".txt"):
        coords = numpy.loadtxt(path.join(directory, file_name))
    else:
        coords = numpy.load(path.join(directory, file_name))

    # Add new cells to the time point
    for row in range(len(coords)):
        particle = Particle(coords[row, 2], coords[row, 1], (coords[row, 0] / Z_OVERSCALED) + z_offset)

        time_point.add_particle(particle)
