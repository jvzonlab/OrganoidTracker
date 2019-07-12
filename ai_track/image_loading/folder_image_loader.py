from os import path
from typing import Optional, Tuple, List

from numpy import ndarray

from ai_track.core import TimePoint
from ai_track.core.image_loader import ImageLoader, ImageChannel
from ai_track.core.experiment import Experiment
from ai_track.image_loading._simple_image_file_reader import read_image_3d


class _IndexedChannel(ImageChannel):
    """A simple channel system with numbers."""
    _index: int

    def __init__(self, index: int):
        self._index = index

    @property
    def index(self) -> int:
        return self._index

    def __repr__(self) -> str:
        return f"_IndexedChannel({self._index})"


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
                # Not a fatal error if channel 0 doesn't exist
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

    if not experiment.name.has_name():
        experiment.name.set_name(path.basename(folder).replace("-stacks", ""))
    experiment.images.image_loader(FolderImageLoader(folder, file_name_format, min_time_point, max_time_point,
                                                     min_channel, max_found_channel))


class FolderImageLoader(ImageLoader):

    _folder: str
    _file_name_format: str
    _min_time_point: int
    _max_time_point: int
    _channels: List[_IndexedChannel]
    _image_size_zyx: Optional[Tuple[int, int, int]]

    def __init__(self, folder: str, file_name_format: str, min_time_point: int, max_time_point: int, min_channel: int,
                 max_channel: int):
        """Creates a loader for multi-page TIFF files. file_name_format is a format string (so containing something
        like %03d), accepting one parameter representing the time point number."""
        self._folder = folder
        self._file_name_format = file_name_format
        self._min_time_point = min_time_point
        self._max_time_point = max_time_point
        self._image_size_zyx = None
        self._channels = [_IndexedChannel(i) for i in range(min_channel, max_channel + 1)]

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        """Just get the size of the image at the first time point, and cache it."""
        if self._image_size_zyx is None:
            first_image_stack = self.get_image_array(TimePoint(self._min_time_point), self._channels[0])
            if first_image_stack is not None:
                self._image_size_zyx = first_image_stack.shape
        return self._image_size_zyx

    def get_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        if time_point.time_point_number() < self._min_time_point or\
                time_point.time_point_number() > self._max_time_point:
            return None
        if not isinstance(image_channel, _IndexedChannel):
            return None  # Asking for an image channel that doesn't exist

        file_name = path.join(self._folder, self._file_name_format.format(
                time=time_point.time_point_number(),
                channel=image_channel.index))

        return read_image_3d(file_name)

    def get_channels(self) -> List[ImageChannel]:
        return self._channels

    def first_time_point_number(self) -> Optional[int]:
        return self._min_time_point

    def last_time_point_number(self) -> Optional[int]:
        return self._max_time_point

    def copy(self) -> ImageLoader:
        channel_indices = [channel.index for channel in self._channels]

        return FolderImageLoader(self._folder, self._file_name_format, self._min_time_point, self._max_time_point,
                                 min(channel_indices), max(channel_indices))

    def serialize_to_config(self) -> Tuple[str, str]:
        return self._folder, self._file_name_format
