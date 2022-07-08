"""
For summing multiple image channels. Example usage:

>>> old_image_loader: ImageLoader = ...
>>> old_channels = old_image_loader.get_channels()
>>> merged_image_loader = ChannelMergingImageLoader(old_image_loader, [
>>>    [old_channels[0], old_channels[1], old_channels[2]],
>>>    [old_channels[3]]
>>> ])

"""

from typing import Tuple, List, Optional, Iterable, Collection

from numpy import ndarray

from organoid_tracker.core import TimePoint
from organoid_tracker.core.image_loader import ImageLoader, ImageChannel
from organoid_tracker.util import bits


class ChannelMergingImageLoader(ImageLoader):
    """Sums multiple channels, which is useful to enhance an image."""
    _image_loader: ImageLoader
    _channels: List[List[ImageChannel]]

    def __init__(self, original: ImageLoader, channels: Iterable[Collection[ImageChannel]]):
        self._image_loader = original
        self._channels = list()
        for channel_group in channels:
            self._channels.append(list(channel_group))

    def get_3d_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        if image_channel.index_zero >= len(self._channels):
            return None  # Don't know this channel

        image = None
        original_channels = self._channels[image_channel.index_zero]
        for original_channel in original_channels:
            returned_image = self._image_loader.get_3d_image_array(time_point, original_channel)
            if returned_image is None:
                # Easy, no image to merge
                continue
            if image is None:
                # Easy, just replace the image
                image = returned_image
                continue
            # Need to merge two images
            image = bits.add_and_return_8bit(image, returned_image)
        return image

    def get_2d_image_array(self, time_point: TimePoint, image_channel: ImageChannel, image_z: int) -> Optional[ndarray]:
        if image_channel.index_zero >= len(self._channels):
            return None  # Don't know this channel

        image = None
        original_channels = self._channels[image_channel.index_zero]
        for original_channel in original_channels:
            returned_image = self._image_loader.get_2d_image_array(time_point, original_channel, image_z)
            if returned_image is None:
                # Easy, no image to merge
                continue
            if image is None:
                # Easy, just replace the image
                image = returned_image
                continue
            # Need to merge two images
            image = bits.add_and_return_8bit(image, returned_image)
        return image

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        return self._image_loader.get_image_size_zyx()

    def first_time_point_number(self) -> Optional[int]:
        return self._image_loader.first_time_point_number()

    def last_time_point_number(self) -> Optional[int]:
        return self._image_loader.last_time_point_number()

    def get_channel_count(self) -> int:
        return len(self._channels)

    def serialize_to_config(self) -> Tuple[str, str]:
        return self._image_loader.serialize_to_config()

    def get_unmerged_image_loader(self) -> ImageLoader:
        return self._image_loader

    def copy(self) -> "ImageLoader":
        return ChannelMergingImageLoader(self._image_loader.copy(), self._channels.copy())
