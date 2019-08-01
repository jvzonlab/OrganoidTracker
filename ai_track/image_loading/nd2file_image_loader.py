import os
from typing import Tuple, List, Optional

import numpy
from nd2reader.parser import Parser
from numpy.core.multiarray import ndarray

from ai_track.core import TimePoint, max_none, min_none
from ai_track.core.image_loader import ImageLoader, ImageChannel
from ai_track.core.images import Images


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


def load_image_series(file: Nd2File, field_of_view: int, min_time_point: Optional[int] = None,
                      max_time_point: Optional[int] = None) -> ImageLoader:
    """Gets the image loader for the given series inside the given file. Raises ValueError if that series doesn't
    exist."""
    return _Nd2ImageLoader(file._file_name, file._nd2_parser, field_of_view, min_time_point, max_time_point)


def load_image_series_from_config(images: Images, file_name: str, pattern: str, min_time_point: int, max_time_point: int):
    """Loads the image seriers into the images object using the file_name and pattern settings."""
    field_of_view = int(pattern)
    image_loader = load_image_series(Nd2File(file_name), field_of_view, min_time_point, max_time_point)
    images.image_loader(image_loader)


class _NamedImageChannel(ImageChannel):
    name: str

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"_NamedImageChannel({self.name})"



class _Nd2ImageLoader(ImageLoader):
    _file_name: str
    _nd2_parser: Parser
    _channels: List[_NamedImageChannel]
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
        self._channels = [_NamedImageChannel(name) for name in self._nd2_parser.metadata["channels"]]
        time_points = self._nd2_parser.metadata["frames"]
        self._min_time_point = max_none(min(time_points), min_time_point)
        self._max_time_point = min_none(max(time_points), max_time_point)
        self._location = location

    def get_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        print(f"Getting image for {time_point} {image_channel}")
        if not isinstance(image_channel, _NamedImageChannel) or image_channel not in self._channels:
            return None
        if time_point.time_point_number() < self._min_time_point\
                or time_point.time_point_number() > self._max_time_point:
            return None

        frame_number = time_point.time_point_number()
        channel_name = image_channel.name
        depth, height, width = self.get_image_size_zyx()
        image = None
        for z in self._nd2_parser.metadata["z_levels"]:
            # Using location - 1: Nikon NIS-Elements GUI is one-indexed, but save format is zero-indexed
            frame = self._nd2_parser.get_image_by_attributes(frame_number, self._location - 1, channel_name, z, height, width)
            if image is None:
                image = numpy.zeros((depth, height, width), dtype=frame.dtype)
            if len(frame) > 0:
                image[z] = frame
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

    def get_channels(self) -> List[ImageChannel]:
        return self._channels

    def serialize_to_config(self) -> Tuple[str, str]:
        return self._file_name, str(self._location)

    def copy(self) -> "ImageLoader":
        return load_image_series(Nd2File(self._file_name), self._location, self._min_time_point,
                                 self._max_time_point)
