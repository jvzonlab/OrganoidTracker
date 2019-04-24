from os import path
from typing import Optional, Tuple, List

import tifffile
from numpy import ndarray
import numpy

from autotrack.core import TimePoint
from autotrack.core.image_loader import ImageLoader, ImageChannel
from autotrack.core.experiment import Experiment

class _OnlyChannel(ImageChannel):
    pass


_CHANNELS = [_OnlyChannel()]


def load_images_from_folder(experiment: Experiment, folder: str, file_name_format: str,
                            min_time_point: Optional[int] = None, max_time_point: Optional[int] = None):
    if min_time_point is None:
        min_time_point = 0
    if max_time_point is None:
        max_time_point = 5000

    min_time_point = max(0, min_time_point)

    # Create time points for all discovered image files
    time_point_number = min_time_point
    while time_point_number <= max_time_point:
        file_name = path.join(folder, file_name_format % time_point_number)

        if not path.isfile(file_name):
            if time_point_number == 0:
                # Not a fatal error if time point number 0 doesn't exist
                time_point_number += 1
                min_time_point += 1
                continue
            break

        time_point_number += 1
    max_time_point = time_point_number - 1  # Last actual image is attempted number - 1

    if not experiment.name.has_name():
        experiment.name.set_name(path.basename(folder).replace("-stacks", ""))
    experiment.images.image_loader(TiffImageLoader(folder, file_name_format, min_time_point, max_time_point))


class TiffImageLoader(ImageLoader):

    _folder: str
    _file_name_format: str
    _min_time_point: int
    _max_time_point: int
    _image_size_zyx: Optional[Tuple[int, int, int]]

    def __init__(self, folder: str, file_name_format: str, min_time_point: int, max_time_point: int):
        """Creates a loader for multi-page TIFF files. file_name_format is a format string (so containing something
        like %03d), accepting one parameter representing the time point number."""
        self._folder = folder
        self._file_name_format = file_name_format
        self._min_time_point = min_time_point
        self._max_time_point = max_time_point
        self._image_size_zyx = None

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        """Just get the size of the image at the first time point, and cache it."""
        if self._image_size_zyx is None:
            first_image_stack = self.get_image_array(TimePoint(self._min_time_point), _CHANNELS[0])
            if first_image_stack is not None:
                self._image_size_zyx = first_image_stack.shape
        return self._image_size_zyx

    def get_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        if time_point.time_point_number() < self._min_time_point or\
                time_point.time_point_number() > self._max_time_point:
            return None
        if image_channel != _CHANNELS[0]:
            return None  # Asking for an image channel that doesn't exist

        file_name = path.join(self._folder, self._file_name_format % time_point.time_point_number())
        if not path.exists(file_name):
            return None
        with tifffile.TiffFile(file_name, movie=True) as f:
            # noinspection PyTypeChecker
            array = numpy.squeeze(f.asarray(maxworkers=None))  # maxworkers=None makes image loader work on half of all cores
            if array.shape[-1] == 3 or array.shape[-1] == 4:
                # Convert RGB to grayscale
                array = numpy.dot(array[...,:3], [0.299, 0.587, 0.114])
            if len(array.shape) == 3:
                return array
            if len(array.shape) == 2:  # Support for 2d images
                outer = numpy.array((array,))
                return outer
            return None

    def get_channels(self) -> List[ImageChannel]:
        return _CHANNELS

    def first_time_point_number(self) -> Optional[int]:
        return self._min_time_point

    def last_time_point_number(self) -> Optional[int]:
        return self._max_time_point

    def copy(self) -> ImageLoader:
        return TiffImageLoader(self._folder, self._file_name_format, self._min_time_point, self._max_time_point)
