from os import path
from typing import Optional, Tuple, List, Any

import numpy
from numpy import ndarray

from organoid_tracker.core import TimePoint
from organoid_tracker.core.image_loader import ImageLoader, ImageChannel
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.image_loading._simple_image_file_io import read_image_czyx, read_image_cyx, write_image_czyx


def _discover_min_time_point_and_channel(folder: str, file_name_format: str, guess_time_point: int) -> Tuple[Optional[int], Optional[int]]:
    for test_time_point in [0, 1, guess_time_point]:
        for test_channel in [0, 1]:
            file_name = path.join(folder, file_name_format.format(time=test_time_point, channel=test_channel))
            if path.isfile(file_name):
                return test_time_point, test_channel
    return None, None


def load_images_from_folder(experiment: Experiment, folder: str, file_name_format: str,
                            min_time_point: Optional[int] = None, max_time_point: Optional[int] = None):
    min_channel = 0
    if min_time_point is None:
        min_time_point = 0
    if max_time_point is None:
        max_time_point = 1000000

    # Discover all time points and the min channel
    max_found_time_point = min_time_point
    testing_file_name = path.join(folder, file_name_format.format(time=min_time_point - 1, channel=min_channel))
    while max_found_time_point <= max_time_point:
        file_name = path.join(folder, file_name_format.format(time=max_found_time_point, channel=min_channel))
        if not path.isfile(file_name):
            if min_channel == 0:
                # Not a fatal error if channel 0 doesn't exist, but channel 1 does exist
                next_channel_file_name = path.join(folder, file_name_format.format(time=max_found_time_point,
                                                                                   channel=min_channel + 1))
                if path.isfile(next_channel_file_name):
                    min_channel += 1
                    continue
            if max_found_time_point == 0:
                # Not a fatal error if time point number 0 doesn't exist
                max_found_time_point += 1
                min_time_point += 1
                continue
            break

        max_found_time_point += 1

        if testing_file_name == file_name:
            # No time parameter
            break
    max_time_point = max_found_time_point - 1  # Last tested time point doesn't exist, so subtract one

    # Discover max channel
    max_found_channel = min_channel
    testing_file_name = path.join(folder, file_name_format.format(time=min_time_point, channel=min_channel))
    while True:
        file_name = path.join(folder, file_name_format.format(time=min_time_point, channel=max_found_channel + 1))
        if testing_file_name == file_name:
            break  # Channel is not included in file name, so assume there's only one channel
        if not path.isfile(file_name):
            break  # Channel doesn't exist
        max_found_channel += 1

    experiment.name.provide_automatic_name(path.basename(folder).replace("-stacks", ""))
    experiment.images.image_loader(FolderImageLoader(folder, file_name_format, min_time_point, max_time_point,
                                                     min_channel, max_found_channel))


class FolderImageLoader(ImageLoader):

    _folder: str
    _file_name_format: str
    _min_time_point: int
    _max_time_point: int
    _channel_offset: int
    _channel_count: int
    _image_size_czyx: Optional[Tuple[int, int, int, int]]

    def __init__(self, folder: str, file_name_format: str, min_time_point: int, max_time_point: int, min_file_name_channel: int,
                 max_file_name_channel: int):
        """Creates a loader for multi-page TIFF files. file_name_format is a format string (so containing something
        like t{time:03}_c{channel}), accepting one parameter representing the time point number and another the
        channel. It is also possible to have the channel as a part of the image itself, this method will check for
        that in case the file name does not specify multiple channels."""
        self._folder = folder
        self._file_name_format = file_name_format
        self._min_time_point = min_time_point
        self._max_time_point = max_time_point
        self._image_size_czyx = None
        self._channel_offset = min_file_name_channel
        self._channel_count = max_file_name_channel - min_file_name_channel + 1

        if self._channel_count == 1:
            # If we only have one channel, check if maybe the images themselves have multiple channels.
            # If so, we will use those instead of the file-name based channels
            self._channel_count = self._get_image_size_czyx()[0]
            self._channel_offset = 0

    def _get_image_size_czyx(self) -> Tuple[int, int, int, int]:
        """Get the size of the image at the first time point, and cache it."""
        if self._image_size_czyx is None:
            file_name = path.join(self._folder, self._file_name_format.format(time=self._min_time_point, channel=self._channel_offset))
            image_czyx = read_image_czyx(file_name)
            self._image_size_czyx = image_czyx.shape
        return self._image_size_czyx

    def _channel_is_in_image(self) -> bool:
        """We can store multiple channels across multiple files using the "c{channel}" format in the file name.
        Alternatively, we can store multiple channels in a single file. This function returns True if the latter
        is the case."""
        return self._get_image_size_czyx()[0] > 1

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        """Just get the size of the image at the first time point, and cache it."""
        image_size_czyx = self._get_image_size_czyx()
        return image_size_czyx[1], image_size_czyx[2], image_size_czyx[3]

    def get_3d_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        if time_point.time_point_number() < self._min_time_point or\
                time_point.time_point_number() > self._max_time_point:
            return None
        if image_channel.index_zero >= self._channel_count:
            return None  # Asking for an image channel that doesn't exist

        file_name = path.join(self._folder, self._file_name_format.format(
                time=time_point.time_point_number(),
                channel=image_channel.index_zero + self._channel_offset))

        return self._select_channel(read_image_czyx(file_name), image_channel)

    def get_2d_image_array(self, time_point: TimePoint, image_channel: ImageChannel, image_z: int) -> Optional[ndarray]:
        if time_point.time_point_number() < self._min_time_point or\
                time_point.time_point_number() > self._max_time_point:
            return None
        if image_channel.index_zero >= self._channel_count:
            return None  # Asking for an image channel that doesn't exist

        file_name = path.join(self._folder, self._file_name_format.format(
            time=time_point.time_point_number(),
            channel=image_channel.index_zero + self._channel_offset))
        return self._select_channel(read_image_cyx(file_name, image_z), image_channel)

    def _select_channel(self, image_czyx_or_cyx: Optional[ndarray], image_channel: ImageChannel) -> Optional[ndarray]:
        if image_czyx_or_cyx is None:
            return None

        has_multiple_channels = image_czyx_or_cyx.shape[0] > 1
        c_index = image_channel.index_zero + self._channel_offset if self._channel_is_in_image() else 0
        if has_multiple_channels:
            return image_czyx_or_cyx[c_index].copy()  # Copy to avoid keeping a reference to the big original image
        else:
            return image_czyx_or_cyx[c_index]

    def get_channel_count(self) -> int:
        return self._channel_count

    def first_time_point_number(self) -> Optional[int]:
        return self._min_time_point

    def last_time_point_number(self) -> Optional[int]:
        return self._max_time_point

    def copy(self) -> ImageLoader:
        return FolderImageLoader(self._folder, self._file_name_format, self._min_time_point, self._max_time_point,
                                 self._channel_offset, self._channel_offset + self._channel_count - 1)

    def serialize_to_config(self) -> Tuple[str, str]:
        return self._folder, self._file_name_format

    def can_save_images(self, image_channel: ImageChannel) -> bool:
        return True  # Yes we can!

    def save_3d_image_array(self, time_point: TimePoint, image_channel: ImageChannel, image: ndarray):
        if len(image.shape) != 3:
            raise ValueError("Image must be 3D")
        if self._channel_is_in_image():
            # To save to the image file, we must collect all channels
            channel_images = list()
            for channel_index in range(self._channel_count):
                channel_images.append(image if image_channel.index_zero == channel_index
                                      else self.get_3d_image_array(time_point, ImageChannel(index_zero=channel_index)))
            image = numpy.stack(channel_images, axis=0)
        else:
            image = image[numpy.newaxis, ...]  # Add a channel axis

        file_name = path.join(self._folder, self._file_name_format.format(
            time=time_point.time_point_number(),
            channel=image_channel.index_zero + self._channel_offset))
        write_image_czyx(file_name, image)
