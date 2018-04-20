from os import path
from typing import Optional

import tifffile
from numpy import ndarray

from core import Experiment, ImageLoader, TimePoint


def load_images_from_folder(experiment: Experiment, folder: str, file_name_format: str,
                            min_time_point: Optional[int] = None, max_time_point: Optional[int] = None):
    if min_time_point is None:
        min_time_point = 1
    if max_time_point is None:
        max_time_point = 5000

    experiment.set_image_loader(TiffImageLoader(folder, file_name_format))

    # Create time points for all discovered image files
    time_point_number = max(1, min_time_point)
    while time_point_number <= max_time_point:
        file_name = path.join(folder, file_name_format % time_point_number)

        if not path.isfile(file_name):
            break

        experiment.get_or_add_time_point(time_point_number)
        time_point_number += 1


class TiffImageLoader(ImageLoader):

    _folder: str
    _file_name_format: str

    def __init__(self, folder: str, file_name_format: str):
        """Creates a loader for multi-page TIFF files. file_name_format is a format string (so containing something
        like %03d), accepting one parameter representing the time point number."""
        self._folder = folder
        self._file_name_format = file_name_format

    def load_3d_image(self, time_point: TimePoint) -> Optional[ndarray]:
        file_name = path.join(self._folder, self._file_name_format % time_point.time_point_number())
        if not path.exists(file_name):
            return None
        with tifffile.TiffFile(file_name, movie=True) as f:
            # noinspection PyTypeChecker
            return f.asarray(maxworkers=None)  # maxworkers=None makes image loader work on half of all cores

