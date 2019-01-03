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


class ImageOffsets:
    _offset: Dict[int, Position]

    def __init__(self, offsets: List[Position] = None):
        self._offset = dict()

        if offsets is not None:
            for offset in offsets:
                self._offset[offset.time_point_number()] = Position(offset.x, offset.y, offset.z)

    def __eq__(self, other):
        if other is self:
            return True
        if not isinstance(other, ImageOffsets):
            return False
        return other._offset == self._offset

    def of_time_point(self, time_point: TimePoint) -> Position:
        """Gets the pixel offset of the image in the given time point."""
        return self._offset.get(time_point.time_point_number(), _ZERO)

    def update_offset(self, dx: int, dy: int, dz: int, min_time_point: int, max_time_point: int):
        """Sets the offset for all of the given time point range (inclusive). The offset is added to the current offset.
        """
        offset = Position(dx, dy, dz)
        for time_point_number in range(min_time_point, max_time_point + 1):
            current_offset = self._offset.get(time_point_number, _ZERO)
            self._offset[time_point_number] = current_offset.add_pos(offset)

    def to_list(self) -> List[Position]:
        """Exports this offset list as a list of offsets with time points specified."""
        offset_list = []
        for time_point_number, position in self._offset.items():
            offset_list.append(Position(position.x, position.y, position.z, time_point_number=time_point_number))
        return offset_list

    def copy(self) -> "ImageOffsets":
        """Returns a copy of this object. Any changes to the copy won't have an effect on this object, and vice versa.
        """
        copy = ImageOffsets()
        copy._offset = self._offset.copy()  # Positions are immutable, so no need for a deep copy here
        return copy


class Images:
    """Records the images (3D + time), their resolution and their offset."""

    _image_loader: ImageLoader
    _offsets: ImageOffsets
    _resolution: Optional[ImageResolution] = None

    def __init__(self):
        self._image_loader = ImageLoader()
        self._offsets = ImageOffsets()

    @property
    def offsets(self):
        """Gets the image offsets - used to keep the position of the object of interest constant, while the images move.
        """
        return self._offsets

    @offsets.setter
    def offsets(self, offsets: ImageOffsets):
        """Sets the image offsets. We're using a @property here to make sure that an object of the correct type is
        inserted."""
        if not isinstance(offsets, ImageOffsets):
            raise TypeError()
        self._offsets = offsets

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
