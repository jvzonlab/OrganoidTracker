import re
from typing import Dict, List, Any

import numpy

from core import Particle, Experiment
from gui import Window, dialog
from os import path
import os

Z_OVERSCALED = 6
TIME_POINT_FROM_FILE_NAME = re.compile("t(\d+)")


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "File/Import-Import Laetitia's positions...": lambda: _import_laetitia_positions(window)
    }


def _import_laetitia_positions(window: Window):
    if not dialog.popup_message_cancellable("Instructions", "Choose the directory containing *.npy or *.txt files."):
        return
    directory = dialog.prompt_directory("Choose file...")
    if directory is None:
        return

    for file_name in os.listdir(directory):
        _import_file(window.get_experiment(), directory, file_name)

    window.refresh()


def _import_file(experiment: Experiment, directory: str, file_name: str):
    match = re.search(TIME_POINT_FROM_FILE_NAME, file_name)
    if match is None:
        return

    time_point_number = int(match.group(1))  # Safe, as this group contains only numbers
    time_point = experiment.get_or_add_time_point(time_point_number)
    experiment.remove_particles(time_point)

    if file_name.endswith(".txt"):
        coords = numpy.loadtxt(path.join(directory, file_name))
    else:
        coords = numpy.load(path.join(directory, file_name))
    for row in range(len(coords)):
        particle = Particle(coords[row, 2], coords[row, 1], coords[row, 0] / Z_OVERSCALED)
        time_point.add_particle(particle)
