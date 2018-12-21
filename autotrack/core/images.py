from typing import Dict, Optional, List, Tuple

import numpy
from numpy import ndarray

from autotrack.core import TimePoint, UserError
from autotrack.core.image_loader import ImageLoader
from autotrack.core.positions import Position
from autotrack.core.resolution import ImageResolution

_ZERO = Position(0, 0, 0)


class _CachedImageLoader(ImageLoader):
    """Wrapper that caches the last few loaded images."""

    _internal: ImageLoader
    _image_cache: List[Tuple[int, ndarray]]

    def __init__(self, wrapped: ImageLoader):
        self._image_cache = []
        self._internal = wrapped

    def _add_to_cache(self, time_point_number: int, image: ndarray):
        if len(self._image_cache) > 5:
            self._image_cache.pop(0)
        self._image_cache.append((time_point_number, image))

    def get_image_stack(self, time_point: TimePoint) -> Optional[ndarray]:
        time_point_number = time_point.time_point_number()
        for entry in self._image_cache:
            if entry[0] == time_point_number:
                return entry[1]

        # Cache miss
        image = self._internal.get_image_stack(time_point)
        self._add_to_cache(time_point_number, image)
        return image

    def uncached(self) -> ImageLoader:
        return self._internal

    def first_time_point_number(self) -> Optional[int]:
        return self._internal.first_time_point_number()

    def last_time_point_number(self) -> Optional[int]:
        return self._internal.last_time_point_number()

    def copy(self) -> ImageLoader:
        return _CachedImageLoader(self._internal.copy())


class Images:
    """Records the images (3D + time), their resolution and their offset."""

    _image_loader: ImageLoader
    _offset: Dict[int, Position]
    _resolution: Optional[ImageResolution] = None

    def __init__(self):
        self._image_loader = ImageLoader()
        self._offset = numpy.zeros((0, 3), dtype=numpy.int32)

    def set_offset(self, dx: int, dy: int, dz: int, min_time_point: int, max_time_point: int):
        """Sets the offset for all of the given time point range (inclusive)."""
        offset = Position(dx, dy, dz)
        for time_point_number in range(min_time_point, max_time_point + 1):
            self._offset[time_point_number] = offset

    def get_offset(self, time_point: TimePoint) -> Position:
        """Gets the offset of the image in the given time point."""
        return self._offset.get(time_point.time_point_number(), _ZERO)

    def resolution(self):
        """Gets the image resolution. Raises UserError if you try to get the resolution when none has been set."""
        if self._resolution is None:
            raise UserError("No image resolution set", "No image resolution was set. Please set a resolution first.")
        return self._resolution

    def image_loader(self, image_loader: Optional[ImageLoader] = None) -> ImageLoader:
        """Gets/sets the image loader."""
        if image_loader is not None:
            self._image_loader = _CachedImageLoader(image_loader.uncached())
            return image_loader
        return self._image_loader

    def get_image_stack(self, time_point: TimePoint) -> Optional[ndarray]:
        """Loads an image using the current image loader. Returns None if there is no image for this time point."""
        return self._image_loader.get_image_stack(time_point)

    def set_resolution(self, resolution: Optional[ImageResolution]):
        """Sets the image resolution."""
        self._resolution = resolution
