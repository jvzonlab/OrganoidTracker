import os
import re
from os import path
from typing import Dict, Any, AbstractSet

import numpy

from autotrack.core import UserError, TimePoint
from autotrack.core.experiment import Experiment
from autotrack.core.positions import Position, PositionCollection
from autotrack.gui import dialog
from autotrack.gui.window import Window


Z_OVERSCALED = 6.0
EXPECTED_Z_LAYERS = 32
TIME_POINT_FROM_FILE_NAME = re.compile("t(\d+)")


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "File//SaveLoad-Load positions in Laetitia's format...": lambda: _load_laetitia_positions(window),
        "File//Export-Export positions in Laetitia's format...": lambda: _export_laetitia_positions(window)
    }


def _load_laetitia_positions(window: Window):
    new_experiment = Experiment()
    new_experiment.image_loader(window.get_experiment().image_loader())  # Copy over image loader

    z_offset = _get_z_offset(window.get_experiment())

    if not dialog.popup_message_cancellable("Instructions", "Choose the directory containing the *.npy or *.txt files."):
        return
    directory = dialog.prompt_directory("Choose directory with positions files...")
    if directory is None:
        return

    for file_name in os.listdir(directory):
        _import_file(new_experiment, directory, file_name, z_offset)

    window.get_gui_experiment().set_experiment(new_experiment)
    window.redraw_data()


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
        _export_file(experiment.positions.of_time_point(time_point), file_path, z_offset)


def _get_z_offset(experiment: Experiment) -> int:
    """Laetitia adds black xy planes until the image has the expected number of z layers. We don't do that, so we need
    to correct the position z for this.
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
        position = Position(coords[row, 2], coords[row, 1], (coords[row, 0] / Z_OVERSCALED) + z_offset,
                            time_point_number=time_point_number)

        experiment.positions.add(position)


def _export_file(positions: AbstractSet[Position], file_path: str, z_offset: int):
    if len(positions) == 0:
        return
    array = numpy.empty((len(positions), 3), dtype=numpy.int64)

    row = 0
    for position in positions:
        array[row, 2] = round(position.x)
        array[row, 1] = round(position.y)
        array[row, 0] = round((position.z - z_offset) * Z_OVERSCALED)
        row += 1

    numpy.save(file_path, array)
