"""Image loader for a folder with images with timestamps in the file names.

Right now, the only format supported is "XdYhZm", where X is the number of days, Y is the number of hours
and Z is the number of minutes. For example, "01d02h30m" means 1 day, 2 hours and 30 minutes. The time points are
sorted by the timestamps in the file names."""
import os
import re
from typing import Tuple, Optional

import numpy
from numpy import ndarray

from organoid_tracker.core import TimePoint, UserError, max_none, min_none
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageLoader, ImageChannel
from organoid_tracker.core.resolution import ImageTimings
from organoid_tracker.image_loading._simple_image_file_io import read_image_cyx, read_image_czyx, write_image_czyx

_DATE_PATTERN = r"\d+d\d\dh\d\dm"
_DATE_PATTERN_WRITTEN = "{day:02d}d{hour:02d}h{minute:02d}m"


def is_timestamped_pattern(file_name_format: str) -> bool:
    """Returns True if the file name format contains a timestamp pattern, False otherwise."""
    return _DATE_PATTERN_WRITTEN in file_name_format


def load_images_from_folder(experiment: Experiment, container: str, pattern: str,
            min_time_point: Optional[int] = None, max_time_point: Optional[int] = None):
    """Loads images from a folder with timestamped file names. The pattern must contain the timestamp placeholder _DATE_PATTERN_WRITTEN."""
    files = _find_files_in_folder(container, pattern)
    if len(files) == 0:
        return

    image_loader = _TimestampedImageLoader(files, pattern)
    image_loader.set_min_max_time_point_numbers(min_time_point, max_time_point)
    experiment.images.image_loader(image_loader)
    experiment.name.provide_automatic_name(image_loader.guess_automatic_name())
    experiment.images.set_timings(image_loader.create_timings())


def _find_files_in_folder(folder: str, file_name_format: str) -> list[str]:
    """Returns a list of file paths in the folder that match the file name format, which must include _DATE_PATTERN_WRITTEN."""
    # Escape the file name format to make it a valid regex pattern, and replace the timestamp pattern with a regex group
    file_name_format = re.escape(file_name_format)
    file_name_format = file_name_format.replace(re.escape(_DATE_PATTERN_WRITTEN), "(" + _DATE_PATTERN + ")")
    pattern = re.compile(file_name_format)

    file_paths = list()
    for file_name in os.listdir(folder):
        match = pattern.fullmatch(file_name)
        if match is not None:
            file_paths.append(os.path.join(folder, file_name))
    return sorted(file_paths)


def _select_channel(image_czyx_or_cyx: Optional[ndarray], image_channel: ImageChannel) -> Optional[ndarray]:
    """Returns the selected channel. If the image has multiple channels, a copy of the image is returned, to avoid
     a reference to the big original image."""
    if image_czyx_or_cyx is None:
        return None

    if image_czyx_or_cyx.shape[0] == 1:
        return image_czyx_or_cyx[0, ...]  # If we have only one channel, return it

    # Return a copy, to avoid a reference to the big original image
    return image_czyx_or_cyx[image_channel.index_zero, ...].copy()


class _TimestampedImageLoader(ImageLoader):
    _file_paths: list[str]
    _pattern: str
    _size_czyx: Tuple[int, int, int, int]

    _min_time_point_number: int
    _max_time_point_number: int

    def __init__(self, file_paths: list[str], pattern: str):
        self._file_paths = file_paths
        self._pattern = pattern

        if len(file_paths) == 0:
            raise ValueError("No image files found in the selected folder")

        # Load one example image to get the size
        example_image = read_image_czyx(file_paths[0])
        self._size_czyx = example_image.shape

        self._min_time_point_number = 0
        self._max_time_point_number = len(file_paths) - 1

    def set_min_max_time_point_numbers(self, min_time_point_number: Optional[int], max_time_point_number: Optional[int]):
        """Updates the minimum and maximum time point numbers. This is useful if the user wants to load only a subset
        of the images. The time point numbers are clamped to the range of available images."""
        self._min_time_point_number = max_none(0, min_time_point_number)
        self._max_time_point_number = min_none(self._max_time_point_number, max_time_point_number)

    def create_timings(self) -> ImageTimings | None:
        pattern = re.compile(_DATE_PATTERN)

        timings_array_m = numpy.full(len(self._file_paths), -1, dtype=numpy.float64)
        for i in range(len(self._file_paths)):
            file_name = os.path.basename(self._file_paths[i])
            file_name = os.path.splitext(file_name)[0]  # Remove the file extension
            match = pattern.search(file_name)
            if match is not None:
                # Extract the day, hour, and minute from the matched string
                file_name_part = match.group(0)
                day = int(file_name_part[0:file_name_part.index("d")])
                hour = int(file_name_part[file_name_part.index("d") + 1:file_name_part.index("h")])
                minute = int(file_name_part[file_name_part.index("h") + 1:file_name_part.index("m")])
                timings_array_m[i] = day * 24 * 60 + hour * 60 + minute

        if numpy.any(timings_array_m == -1):
            # If any timing is missing, we can't use the timings
            return None

        return ImageTimings(0, timings_array_m)

    def get_3d_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        if image_channel.index_zero < 0 or image_channel.index_zero >= self._size_czyx[0]:
            return None  # Invalid image channel
        if time_point.time_point_number() < 0 or time_point.time_point_number() >= len(self._file_paths):
            return None  # Invalid time point number
        return _select_channel(read_image_czyx(self._file_paths[time_point.time_point_number()]), image_channel)

    def get_2d_image_array(self, time_point: TimePoint, image_channel: ImageChannel, image_z: int) -> Optional[ndarray]:
        if image_channel.index_zero < 0 or image_channel.index_zero >= self._size_czyx[0]:
            return None  # Invalid image channel
        if time_point.time_point_number() < 0 or time_point.time_point_number() >= len(self._file_paths):
            return None  # Invalid time point number
        if image_z < 0 or image_z >= self._size_czyx[1]:
            return None  # Invalid z-plane
        return _select_channel(read_image_cyx(self._file_paths[time_point.time_point_number()], image_z), image_channel)

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        return self._size_czyx[1], self._size_czyx[2], self._size_czyx[3]

    def first_time_point_number(self) -> Optional[int]:
        return self._min_time_point_number

    def last_time_point_number(self) -> Optional[int]:
        return self._max_time_point_number

    def get_channel_count(self) -> int:
        return self._size_czyx[0]

    def serialize_to_config(self) -> Tuple[str, str]:
        return os.path.dirname(self._file_paths[0]), self._pattern

    def copy(self) -> "ImageLoader":
        copy = _TimestampedImageLoader(self._file_paths, self._pattern)
        copy.set_min_max_time_point_numbers(self._min_time_point_number, self._max_time_point_number)
        return copy

    def can_save_images(self, image_channel: ImageChannel) -> bool:
        return True

    def save_3d_image_array(self, time_point: TimePoint, image_channel: ImageChannel, image: ndarray):
        if len(image.shape) != 3:
            raise ValueError("Image must be 3D")

        # To save to the image file, we must collect all channels
        channel_images = list()
        for channel_index in range(self._size_czyx[0]):
            channel_images.append(image if image_channel.index_zero == channel_index
                                  else self.get_3d_image_array(time_point, ImageChannel(index_zero=channel_index)))
        image = numpy.stack(channel_images, axis=0)

        if time_point.time_point_number() < self._min_time_point_number or time_point.time_point_number() > self._max_time_point_number:
            raise UserError("Invalid time point number",
                            f"Time point number {time_point.time_point_number()} is invalid. It must be from {self._min_time_point_number} to {self._max_time_point_number}.")
        file_name = self._file_paths[time_point.time_point_number()]
        write_image_czyx(file_name, image)

    def guess_automatic_name(self) -> str:
        # Check if there's something in the file name format, after removing the pattern
        # (If file names are like "4-8_C6_1_00d00h00m", then we want to extract "4-8_C6_1" as the name)
        name_left = self._pattern.replace(_DATE_PATTERN_WRITTEN, "")
        if "." in name_left:
            name_left = name_left[:name_left.rindex(".")]
        name_left = name_left.strip("_- ")
        if len(name_left) >= 2:
            return name_left

        # Otherwise, we can just use the folder name as the automatic name
        return os.path.basename(os.path.dirname(self._file_paths[0]))

