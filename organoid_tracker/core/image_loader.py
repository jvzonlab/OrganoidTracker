from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Any
from numpy import ndarray

from organoid_tracker.core import TimePoint


class ImageChannel(ABC):
    """Represents an image channel - for example, the bright field channel, the red channel, etc."""

    @abstractmethod
    def __repr__(self) -> str:
        """Returns some way to recreate this image channel, used for debugging."""
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass


class ImageLoader(ABC):
    """Responsible for loading all images in an experiment."""

    @abstractmethod
    def get_3d_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        """Loads an image, usually from disk. Returns None if there is no image for this time point or channel."""
        pass

    @abstractmethod
    def get_2d_image_array(self, time_point: TimePoint, image_channel: ImageChannel, image_z: int) -> Optional[ndarray]:
        """Loads one single 2d slice of an image. Returns None if there is no image for this z, time point or channel.
        Note: the image z always goes from 0 to image_size_z - 1.
        """

    @abstractmethod
    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        """Gets the image size. Returns None if there are no images."""
        pass

    @abstractmethod
    def first_time_point_number(self) -> Optional[int]:
        """Gets the first time point for which images are available."""
        pass

    @abstractmethod
    def last_time_point_number(self) -> Optional[int]:
        """Gets the last time point (inclusive) for which images are available."""
        pass

    @abstractmethod
    def get_channels(self) -> List[ImageChannel]:
        """Gets a list of all available image channels."""
        pass

    @abstractmethod
    def serialize_to_config(self) -> Tuple[str, str]:
        """Serializes this image loader into a path and a file/series name. This can be stored in configuration files."""
        pass

    @abstractmethod
    def copy(self) -> "ImageLoader":
        """Copies the image loader, so that you can use it on another thread."""
        pass

    def uncached(self) -> "ImageLoader":
        """If this loader is a caching wrapper around another loader, this method returns one loader below. Otherwise,
        it returns self.
        """
        return self

    def has_images(self) -> bool:
        """Returns True if there are any images loaded, False otherwise.

        The default implementation just checks if get_image_size_zyx() returns something. If yes, then it is assumed
        that there are images stored.
        """
        return self.get_image_size_zyx() is not None


class NullImageLoader(ImageLoader):

    def copy(self) -> "ImageLoader":
        return self  # No need to copy, object holds no data

    def get_3d_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        return None

    def get_2d_image_array(self, time_point: TimePoint, image_channel: ImageChannel, image_z: int) -> Optional[ndarray]:
        return None

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        return None

    def first_time_point_number(self) -> Optional[int]:
        return None

    def last_time_point_number(self) -> Optional[int]:
        return None

    def get_channels(self) -> List[ImageChannel]:
        return []

    def serialize_to_config(self) -> Tuple[str, str]:
        return "", ""


class ImageFilter:
    """Filter for images, for example to enhance the contrast."""

    @abstractmethod
    def filter(self, image_8bit: ndarray):
        """Filters the given input array, which is a grayscale array with values from 0 to 255 of 2 or 3 dimensions.
        The input array will be modified."""
        raise NotImplementedError()

    @abstractmethod
    def copy(self):
        """Copies the filter, such that changes to this filter have no effect on the copy, and vice versa."""
        raise NotImplementedError()

    @abstractmethod
    def get_name(self) -> str:
        """Returns a user-friendly name, like "Enhance contrast" or "Suppress noise"."""
        raise NotImplementedError()
