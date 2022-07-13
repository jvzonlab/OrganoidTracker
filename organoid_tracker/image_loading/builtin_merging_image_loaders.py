"""Contains image loaders that merge several other image loaders. Examples:

- An image loader that appends one set of images after another.
  >>> ChannelSummingImageLoader
- An image loader that takes channel 1 and 2 from one set of images, and channel 3 of another.
  >>> ChannelAppendingImageLoader
- An image loader that appends multiple time lapses after each other
  >>> TimeAppendingImageLoader
"""

from typing import Tuple, List, Optional, Iterable, Collection, Dict, Any

from numpy import ndarray

from organoid_tracker.core import TimePoint, min_none, max_none
from organoid_tracker.core.image_loader import ImageLoader, ImageChannel
from organoid_tracker.util import bits


class ChannelSummingImageLoader(ImageLoader):
    """
    For summing multiple image channels. Example usage:

    >>> old_image_loader: ImageLoader = ...
    >>> old_channels = old_image_loader.get_channels()
    >>> merged_image_loader = ChannelSummingImageLoader(old_image_loader, [
    >>>    [old_channels[0], old_channels[1], old_channels[2]],
    >>>    [old_channels[3]]
    >>> ])

    """
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

    def serialize_to_dictionary(self) -> Dict[str, Any]:
        channel_indices = list()
        for channels in self._channels:
            channel_indices.append([channel.index_one for channel in channels])

        return {
            "images_original": self._image_loader.serialize_to_dictionary(),
            "images_channel_summing": channel_indices
        }

    def get_unmerged_image_loader(self) -> ImageLoader:
        return self._image_loader

    def copy(self) -> "ImageLoader":
        return ChannelSummingImageLoader(self._image_loader.copy(), self._channels.copy())


class ChannelAppendingImageLoader(ImageLoader):
    """Combines multiple image loaders, showing their channels after each other."""

    # A list of all image loaders, where every loader appears once in the list
    _unique_loaders: List[ImageLoader]

    def __init__(self, image_loaders: List[ImageLoader]):
        self._unique_loaders = list()

        for image_loader in image_loaders:
            if not image_loader.has_images():
                continue

            if isinstance(image_loader, ChannelAppendingImageLoader):
                # Avoid double-wrapping, could hurt performance
                self._unique_loaders += image_loader._unique_loaders
            else:
                self._unique_loaders.append(image_loader)

    def get_3d_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        # Iterate over all image loaders
        image_channel_index = image_channel.index_zero
        for image_loader in self._unique_loaders:
            if image_channel_index < image_loader.get_channel_count():
                return image_loader.get_3d_image_array(time_point, ImageChannel(index_zero=image_channel_index))
            image_channel_index -= image_loader.get_channel_count()

        return None

    def get_2d_image_array(self, time_point: TimePoint, image_channel: ImageChannel, image_z: int) -> Optional[ndarray]:
        # Iterate over all image loaders
        image_channel_index = image_channel.index_zero
        for image_loader in self._unique_loaders:
            if image_channel_index < image_loader.get_channel_count():
                return image_loader.get_2d_image_array(time_point, ImageChannel(index_zero=image_channel_index), image_z)
            image_channel_index -= image_loader.get_channel_count()

        return None

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        # Returns the size only if all image loaders have the same image size
        the_size = None
        for image_loader in self._unique_loaders:
            new_size = image_loader.get_image_size_zyx()
            if new_size is None:
                return None  # Not all sizes are known, so return None
            if the_size is None:
                the_size = new_size  # Set the first size
            else:
                if the_size != new_size:
                    return None  # Different channels have different sizes, so return None
        return the_size

    def first_time_point_number(self) -> Optional[int]:
        # Return the first time point number for which we have at least one channel
        first_time_point_numbers = [image_loader.first_time_point_number() for image_loader in self._unique_loaders]
        return min_none(first_time_point_numbers)

    def last_time_point_number(self) -> Optional[int]:
        # Return the last time point number for which we have at least one channel
        last_time_point_numbers = [image_loader.last_time_point_number() for image_loader in self._unique_loaders]
        return max_none(last_time_point_numbers)

    def get_channel_count(self) -> int:
        count = 0
        for image_loader in self._unique_loaders:
            count += image_loader.get_channel_count()
        return count

    def serialize_to_config(self) -> Tuple[str, str]:
        # This data format makes it impossible to store all information (at least not without horrible hacks),
        # so we just serialize the first one
        # People should just use self.serialize_to_dictionary() if possible
        for image_loader in self._unique_loaders:
            return image_loader.serialize_to_config()
        return "", ""

    def serialize_to_dictionary(self) -> Dict[str, Any]:
        return {
            "images_channel_appending": [
                image_loader.serialize_to_dictionary() for image_loader in self._unique_loaders
            ]
        }

    def copy(self) -> "ImageLoader":
        new_internal = list()
        for internal in self._unique_loaders:
            new_internal.append(internal.copy())
        return ChannelAppendingImageLoader(new_internal)

    def uncached(self) -> "ImageLoader":
        new_internal = list()
        for internal in self._unique_loaders:
            new_internal.append(internal.uncached())
        return ChannelAppendingImageLoader(new_internal)


class TimeAppendingImageLoader(ImageLoader):
    """Combines to image loaders, showing images after each other."""
    _internal: List[ImageLoader]

    def __init__(self, image_loaders: List[ImageLoader]):
        self._internal = list()
        for image_loader in image_loaders:
            if not image_loader.has_images():
                continue

            if isinstance(image_loader, TimeAppendingImageLoader):
                # Avoid double-wrapping, could hurt performance
                self._internal += image_loader._internal
            else:
                self._internal.append(image_loader)

    def get_3d_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        if len(self._internal) == 0:
            return None

        time_point_number = time_point.time_point_number()

        # List of channels is different for different image loaders, so use the channel index instead
        try:
            channel_index = self.get_channels().index(image_channel)
        except ValueError:
            return None  # Channel not available

        image_loader_index = 0
        while True:
            # Check channels
            channels = self._internal[image_loader_index].get_channels()
            if channel_index >= len(channels):
                return None  # Not that many channels available for this ImageLoader

            if time_point_number <= self._internal[image_loader_index].last_time_point_number():
                return self._internal[image_loader_index].get_3d_image_array(
                    TimePoint(time_point_number),
                    channels[channel_index])

            # Out of bounds for this time lapse, on to the next
            time_point_number -= self._internal[image_loader_index].last_time_point_number() + 1
            image_loader_index += 1
            if image_loader_index >= len(self._internal):
                return None  # Out of images
            time_point_number += self._internal[image_loader_index].first_time_point_number()

    def get_2d_image_array(self, time_point: TimePoint, image_channel: ImageChannel, image_z: int) -> Optional[ndarray]:
        if len(self._internal) == 0:
            return None

        time_point_number = time_point.time_point_number()

        # List of channels is different for different image loaders, so use the channel index instead
        try:
            channel_index = self.get_channels().index(image_channel)
        except ValueError:
            return None  # Channel not available

        image_loader_index = 0
        while True:
            if time_point_number <= self._internal[image_loader_index].last_time_point_number():
                # Check channels
                channels = self._internal[image_loader_index].get_channels()
                if channel_index >= len(channels):
                    return None  # Not that many channels available for this ImageLoader

                return self._internal[image_loader_index].get_2d_image_array(
                    TimePoint(time_point_number),
                    channels[channel_index], image_z)

            # Out of bounds for this time lapse, on to the next
            time_point_number -= self._internal[image_loader_index].last_time_point_number() + 1
            image_loader_index += 1
            if image_loader_index >= len(self._internal):
                return None  # Out of images
            time_point_number += self._internal[image_loader_index].first_time_point_number()

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        if len(self._internal) == 0:
            return None
        return self._internal[0].get_image_size_zyx()

    def first_time_point_number(self) -> Optional[int]:
        if len(self._internal) == 0:
            return None
        return self._internal[0].first_time_point_number()

    def last_time_point_number(self) -> Optional[int]:
        if len(self._internal) == 0:
            return None

        image_count = 0
        for internal in self._internal:
            new_last = internal.last_time_point_number()
            new_first = internal.first_time_point_number()
            if new_last is None or new_first is None:
                return None
            image_count += new_last - new_first + 1
        return image_count + self._internal[0].first_time_point_number() - 1

    def get_channel_count(self) -> int:
        # Return the highest count for selecting channels, in case multiple time lapses have different numbers of
        # channels
        highest_number = 0
        for internal in self._internal:
            highest_number = max(internal.get_channel_count(), highest_number)
        return highest_number

    def serialize_to_config(self) -> Tuple[str, str]:
        # This data format makes it impossible to store all information (at least not without horrible hacks),
        # so we just serialize the first one
        # People should just use self.serialize_to_dictionary() if possible
        for image_loader in self._internal:
            return image_loader.serialize_to_config()
        return "", ""

    def serialize_to_dictionary(self) -> Dict[str, Any]:
        return {
            "images_time_appending": [
                image_loader.serialize_to_dictionary() for image_loader in self._internal
            ]
        }

    def copy(self) -> "ImageLoader":
        new_internal = list()
        for internal in self._internal:
            new_internal.append(internal.copy())
        return TimeAppendingImageLoader(new_internal)

    def uncached(self) -> "ImageLoader":
        new_internal = list()
        for internal in self._internal:
            new_internal.append(internal.uncached())
        return TimeAppendingImageLoader(new_internal)


