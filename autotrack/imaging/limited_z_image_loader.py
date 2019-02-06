from typing import Optional, Tuple, List

from numpy import ndarray

from autotrack.core import TimePoint
from autotrack.core.image_loader import ImageLoader, ImageChannel


class LimitedZImageLoader(ImageLoader):
    """Image loader that cuts off all images after a certain z position."""
    _original: ImageLoader
    _max_layers: int

    def __init__(self, image_loader: ImageLoader, max_layers: int):
        self._original = image_loader
        self._max_layers = max_layers

    def copy(self) -> "ImageLoader":
        return LimitedZImageLoader(self._original.copy(), self._max_layers)

    def get_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        array = self._original.get_image_array(time_point, image_channel)
        if array is None:
            return None
        if array.shape[0] > self._max_layers:
            return array[0:self._max_layers]
        return array

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        size = self._original.get_image_size_zyx()
        if size is None:
            return None
        if size[0] > self._max_layers:
            return (self._max_layers, size[1], size[2])
        return size

    def uncached(self) -> "ImageLoader":
        return LimitedZImageLoader(self._original.uncached(), self._max_layers)

    def first_time_point_number(self) -> Optional[int]:
        return self._original.first_time_point_number()

    def last_time_point_number(self) -> Optional[int]:
        return self._original.last_time_point_number()

    def get_channels(self) -> List[ImageChannel]:
        return self._original.get_channels()
