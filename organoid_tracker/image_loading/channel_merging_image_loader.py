"""
For summing multiple image channels. Example usage:

>>> old_image_loader: ImageLoader = ...
>>> old_channels = old_image_loader.get_channels()
>>> merged_image_loader = ChannelMergingImageLoader(old_image_loader, [
>>>    [old_channels[0], old_channels[1], old_channels[2]],
>>>    [old_channels[3]]
>>> ])

"""

from typing import Tuple, List, Optional, Dict, Any, Iterable, Collection

import numpy
from numpy import ndarray

from organoid_tracker.core import TimePoint
from organoid_tracker.core.image_loader import ImageLoader, ImageChannel
from organoid_tracker.util import bits


class _MergedImageChannel(ImageChannel):
    _channels: List[ImageChannel]

    def __init__(self, channels: Collection[ImageChannel]):
        self._channels = list(channels)

    def __repr__(self) -> str:
        return "_MergedImageChannel(" + repr(self._channels) + ")"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _MergedImageChannel):
            return False
        return other._channels == self._channels

    def __hash__(self) -> int:
        return hash(self._channels)


def _add_and_return_8bit(a: ndarray, b: ndarray) -> ndarray:
    """Adds the two arrays. First, the arrays are scaled to 8bit if they aren't already, and then they are added without
    overflow issues: 240 + 80 is capped at 255."""
    a = bits.ensure_8bit(a)
    b = bits.ensure_8bit(b)

    # https://stackoverflow.com/questions/29611185/avoid-overflow-when-adding-numpy-arrays
    b = 255 - b  # old b is gone shortly after new array is created
    numpy.putmask(a, b < a, b)  # a temp bool array here, then it's gone
    a += 255 - b  # a temp array here, then it's gone
    return a


class ChannelMergingImageLoader(ImageLoader):
    """Sums multiple channels, which is useful to enhance an image."""
    _image_loader: ImageLoader
    _channels: List[_MergedImageChannel]

    def __init__(self, original: ImageLoader, channels: Iterable[Collection[ImageChannel]]):
        self._image_loader = original
        self._channels = list()
        for channel_group in channels:
            self._channels.append(_MergedImageChannel(channel_group))

    def get_3d_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        if not isinstance(image_channel, _MergedImageChannel):
            return None  # Don't know this channel

        image = None
        for original_channel in image_channel._channels:
            returned_image = self._image_loader.get_3d_image_array(time_point, original_channel)
            if returned_image is None:
                # Easy, no image to merge
                continue
            if image is None:
                # Easy, just replace the image
                image = returned_image
                continue
            # Need to merge two images
            image = _add_and_return_8bit(image, returned_image)
        return image

    def get_2d_image_array(self, time_point: TimePoint, image_channel: ImageChannel, image_z: int) -> Optional[ndarray]:
        if not isinstance(image_channel, _MergedImageChannel):
            return None  # Don't know this channel

        image = None
        for original_channel in image_channel._channels:
            returned_image = self._image_loader.get_2d_image_array(time_point, original_channel, image_z)
            if returned_image is None:
                # Easy, no image to merge
                continue
            if image is None:
                # Easy, just replace the image
                image = returned_image
                continue
            # Need to merge two images
            image = _add_and_return_8bit(image, returned_image)
        return image

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        return self._image_loader.get_image_size_zyx()

    def first_time_point_number(self) -> Optional[int]:
        return self._image_loader.first_time_point_number()

    def last_time_point_number(self) -> Optional[int]:
        return self._image_loader.last_time_point_number()

    def get_channels(self) -> List[ImageChannel]:
        return self._channels

    def serialize_to_config(self) -> Tuple[str, str]:
        return self._image_loader.serialize_to_config()

    def copy(self) -> "ImageLoader":
        return ChannelMergingImageLoader(self._image_loader.copy(),
                                         (channel._channels for channel in self._channels))