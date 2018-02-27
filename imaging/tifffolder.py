from os import path
from typing import Optional

import tifffile
from numpy import ndarray

from imaging import Experiment


def load_images_from_folder(experiment: Experiment, folder: str, file_name_format: str,
                            min_time_point: Optional[int] = None, max_time_point: Optional[int] = None):
    if min_time_point is None:
        min_time_point = 1
    if max_time_point is None:
        max_time_point = 5000

    time_point = min_time_point
    while time_point <= max_time_point:
        file_name = path.join(folder, file_name_format % time_point)

        if not path.isfile(file_name):
            break

        experiment.add_image_loader(time_point, _create_image_loader(file_name))
        time_point += 1


def _create_image_loader(file_name: str):
    def image_loader() -> ndarray:
        with tifffile.TiffFile(file_name) as f:
            # noinspection PyTypeChecker
            return f.asarray(maxworkers=None)  # maxworkers=None makes image loader work on half of all cores
    return image_loader
