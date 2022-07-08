import os
from typing import Tuple, List, Optional, Any

import numpy
from nd2reader.parser import Parser
from numpy.core.multiarray import ndarray

from organoid_tracker.core import TimePoint, max_none, min_none
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageLoader, ImageChannel
from organoid_tracker.core.images import Images


class Nd2File:
    """A ND2 file with searchable image series. Use load_image_series to load the actual images."""
    _nd2_parser: Parser
    _file_name: str

    def __init__(self, file_name: str):
        if not os.path.exists(file_name):
            raise ValueError("File does not exist: " + str(file_name))
        self._file_name = file_name
        handle = open(file_name, "rb")
        self._nd2_parser = Parser(handle)

    def get_location_counts(self) -> int:
        """Gets all available loctions, from 1 to the returned count."""
        return len(self._nd2_parser.metadata["fields_of_view"])


def load_image_series(experiment: Experiment, file: Nd2File, field_of_view: int, min_time_point: Optional[int] = None,
                      max_time_point: Optional[int] = None):
    """Gets the image loader for the given series inside the given file. Raises ValueError if that series doesn't
    exist."""
    image_loader = _Nd2ImageLoader(file._file_name, file._nd2_parser, field_of_view, min_time_point, max_time_point)
    experiment.images.image_loader(image_loader)

    # Generate an automatic name for the experiment
    file_name = os.path.basename(file._file_name)
    if file_name.lower().endswith(".nd2"):
        file_name = file_name[:-4]
    experiment.name.provide_automatic_name(file_name + f"xy{field_of_view:02}")


def load_image_series_from_config(experiment: Experiment, file_name: str, pattern: str, min_time_point: int, max_time_point: int):
    """Loads the image seriers into the images object using the file_name and pattern settings."""
    field_of_view = int(pattern)
    load_image_series(experiment, Nd2File(file_name), field_of_view, min_time_point, max_time_point)



class _Nd2ImageLoader(ImageLoader):
    _file_name: str
    _nd2_parser: Parser
    _channels: List[ImageChannel]
    _min_time_point: int
    _max_time_point: int
    _location: int

    def __init__(self, file_name: str, nd2_parser: Parser, location: int, min_time_point: Optional[int],
                 max_time_point: Optional[int]):
        max_field_of_view = len(nd2_parser.metadata["fields_of_view"])
        if location < 1 or location > max_field_of_view:
            raise ValueError(f"Unknown field_of_view: {location}. Available: 0 to {max_field_of_view}")

        self._file_name = file_name
        self._nd2_parser = nd2_parser
        self._channels = [ImageChannel(index_zero=i) for i, name in enumerate(self._nd2_parser.metadata["channels"])]
        time_points = self._nd2_parser.metadata["frames"]
        self._min_time_point = max_none(min(time_points), min_time_point)
        self._max_time_point = min_none(max(time_points), max_time_point)
        self._location = location

    def get_3d_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        if image_channel.index_zero >= len(self._channels):
            return None
        if time_point.time_point_number() < self._min_time_point\
                or time_point.time_point_number() > self._max_time_point:
            return None

        frame_number = time_point.time_point_number()
        depth, height, width = self.get_image_size_zyx()
        image = None
        for z in self._nd2_parser.metadata["z_levels"]:
            # Using "location - 1": Nikon NIS-Elements GUI is one-indexed, but save format is zero-indexed
            frame = self._nd2_parser.get_image_by_attributes(frame_number, self._location - 1, image_channel.index_zero, z, height, width)
            if image is None:
                image = numpy.zeros((depth, height, width), dtype=frame.dtype)
            if len(frame) > 0:
                image[z] = frame
        return image

    def get_2d_image_array(self, time_point: TimePoint, image_channel: ImageChannel, image_z: int) -> Optional[ndarray]:
        if image_channel.index_zero >= len(self._channels):
            return None
        if time_point.time_point_number() < self._min_time_point\
                or time_point.time_point_number() > self._max_time_point:
            return None

        depth, height, width = self.get_image_size_zyx()
        if image_z < 0 or image_z >= depth:
            return None

        frame_number = time_point.time_point_number()
        channel_index = self._channels.index(image_channel)
        # Using "location - 1": Nikon NIS-Elements GUI is one-indexed, but save format is zero-indexed
        image = self._nd2_parser.get_image_by_attributes(frame_number, self._location - 1, channel_index, image_z, height, width)
        if len(image.shape) != 2:
            return None
        return image

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        height = self._nd2_parser.metadata["height"]
        width = self._nd2_parser.metadata["width"]
        depth = len(self._nd2_parser.metadata["z_levels"])
        return depth, height, width

    def first_time_point_number(self) -> Optional[int]:
        return self._min_time_point

    def last_time_point_number(self) -> Optional[int]:
        return self._max_time_point

    def get_channel_count(self) -> int:
        return len(self._channels)

    def serialize_to_config(self) -> Tuple[str, str]:
        return self._file_name, str(self._location)

    def copy(self) -> "ImageLoader":
        return _Nd2ImageLoader(self._file_name, Nd2File(self._file_name)._nd2_parser, self._location,
                               self._min_time_point, self._max_time_point)
