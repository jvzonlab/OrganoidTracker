from os import path
from typing import Optional, Tuple, List, Any

from numpy import ndarray

from organoid_tracker.core import TimePoint
from organoid_tracker.core.image_loader import ImageLoader, ImageChannel
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.image_loading._simple_image_file_io import read_image_3d, read_image_2d, write_image_3d


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
    _image_size_zyx: Optional[Tuple[int, int, int]]

    def __init__(self, folder: str, file_name_format: str, min_time_point: int, max_time_point: int, min_channel: int,
                 max_channel: int):
        """Creates a loader for multi-page TIFF files. file_name_format is a format string (so containing something
        like {time:03}), accepting one parameter representing the time point number."""
        self._folder = folder
        self._file_name_format = file_name_format
        self._min_time_point = min_time_point
        self._max_time_point = max_time_point
        self._image_size_zyx = None
        self._channel_offset = min_channel
        self._channel_count = max_channel - min_channel + 1

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        """Just get the size of the image at the first time point, and cache it."""
        if self._image_size_zyx is None:
            first_image_stack = self.get_3d_image_array(TimePoint(self._min_time_point), ImageChannel(index_zero=0))
            if first_image_stack is not None:
                self._image_size_zyx = first_image_stack.shape
        return self._image_size_zyx

    def get_3d_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        if time_point.time_point_number() < self._min_time_point or\
                time_point.time_point_number() > self._max_time_point:
            return None
        if image_channel.index_zero >= self._channel_count:
            return None  # Asking for an image channel that doesn't exist

        file_name = path.join(self._folder, self._file_name_format.format(
                time=time_point.time_point_number(),
                channel=image_channel.index_zero + self._channel_offset))

        return read_image_3d(file_name)

    def get_2d_image_array(self, time_point: TimePoint, image_channel: ImageChannel, image_z: int) -> Optional[ndarray]:
        if time_point.time_point_number() < self._min_time_point or\
                time_point.time_point_number() > self._max_time_point:
            return None
        if image_channel.index_zero >= self._channel_count:
            return None  # Asking for an image channel that doesn't exist

        file_name = path.join(self._folder, self._file_name_format.format(
            time=time_point.time_point_number(),
            channel=image_channel.index_zero + self._channel_offset))
        return read_image_2d(file_name, image_z)

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

        file_name = path.join(self._folder, self._file_name_format.format(
            time=time_point.time_point_number(),
            channel=image_channel.index_zero + self._channel_offset))
        write_image_3d(file_name, image)
