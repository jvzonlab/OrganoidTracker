from typing import Tuple, List, Optional

import numpy
from numpy.core.multiarray import ndarray

from ai_track.core import TimePoint
from ai_track.core.image_loader import ImageLoader, ImageChannel
from ai_track.imaging import bits
from ai_track.position_detection import thresholding


class NoiseSuppressingFilter(ImageLoader):
    _internal: ImageLoader

    def __init__(self, image_loader: ImageLoader):
        self._internal = image_loader

    def get_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        array = self._internal.get_image_array(time_point, image_channel)
        if array is None:
            return None

        # Remove all pixels outside the threshold
        array = bits.image_to_8bit(array)
        array[array < 40] = 0
        return array

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        return self._internal.get_image_size_zyx()

    def first_time_point_number(self) -> Optional[int]:
        return self._internal.first_time_point_number()

    def last_time_point_number(self) -> Optional[int]:
        return self._internal.last_time_point_number()

    def get_channels(self) -> List[ImageChannel]:
        return self._internal.get_channels()

    def serialize_to_config(self) -> Tuple[str, str]:
        return self._internal.serialize_to_config()

    def copy(self) -> "ImageLoader":
        return NoiseSuppressingFilter(self._internal.copy())

    def uncached(self) -> "ImageLoader":
        return NoiseSuppressingFilter(self._internal.uncached())
