from abc import ABC
from typing import Optional, Tuple, List
from numpy import ndarray

from autotrack.core import TimePoint


class ImageChannel:
    """Represents an image channel - for example, the bright field channel, the red channel, etc."""


class ImageLoader:
    """Responsible for loading all images in an experiment."""

    def get_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        """Loads an image, usually from disk. Returns None if there is no image for this time point or channel.

        Using image_channel you can ask for a specific channel. Use None to just use the default channel."""
        return None

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        """Gets the image size. Returns None if there are no images, or if the image size is not constant."""
        return None

    def uncached(self) -> "ImageLoader":
        """If this loader is a caching wrapper around another loader, this method returns one loader below. Otherwise,
        it returns self.
        """
        return self

    def first_time_point_number(self) -> Optional[int]:
        """Gets the first time point for which images are available."""
        return None

    def last_time_point_number(self) -> Optional[int]:
        """Gets the last time point (inclusive) for which images are available."""
        return None

    def get_channels(self) -> List[ImageChannel]:
        """Gets a list of all available image channels."""
        return []

    def copy(self) -> "ImageLoader":
        """Copies the image loader, so that you can use it on another thread."""
        return self
