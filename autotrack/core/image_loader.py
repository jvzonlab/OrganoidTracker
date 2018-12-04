from typing import Optional
from numpy import ndarray

from autotrack.core import TimePoint


class ImageLoader:
    """Responsible for loading all images in an experiment."""

    def get_image_stack(self, time_point: TimePoint) -> Optional[ndarray]:
        """Loads an image, usually from disk. Returns None if there is no image for this time point."""
        return None

    def uncached(self) -> "ImageLoader":
        """If this loader is a caching wrapper around another loader, this method returns one loader below. Otherwise,
        it returns self.
        """
        return self

    def get_first_time_point(self) -> Optional[int]:
        """Gets the first time point for which images are available."""
        return None

    def get_last_time_point(self) -> Optional[int]:
        """Gets the last time point (inclusive) for which images are available."""
        return None

    def copy(self) -> "ImageLoader":
        """Copies the image loader, so that you can use it on another thread."""
        raise NotImplementedError()
