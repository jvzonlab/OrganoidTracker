from typing import Dict, Optional, List, Tuple, Iterable

from numpy import ndarray

from ai_track.core import TimePoint, UserError
from ai_track.core.image_loader import ImageLoader, ImageChannel, NullImageLoader, ImageFilter
from ai_track.core.position import Position
from ai_track.core.resolution import ImageResolution
from ai_track.util import bits

_ZERO = Position(0, 0, 0)


class _CachedImageLoader(ImageLoader):
    """Wrapper that caches the last few loaded images."""

    _internal: ImageLoader
    _image_cache: List[Tuple[int, ImageChannel, ndarray]]

    def __init__(self, wrapped: ImageLoader):
        self._image_cache = []
        self._internal = wrapped

    def _add_to_cache(self, time_point_number: int, image_channel: ImageChannel, image: ndarray):
        if len(self._image_cache) > 5:
            self._image_cache.pop(0)
        self._image_cache.append((time_point_number, image_channel, image))

    def get_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        time_point_number = time_point.time_point_number()
        for entry in self._image_cache:
            if entry[0] == time_point_number and entry[1] == image_channel:
                return entry[2]

        # Cache miss
        image = self._internal.get_image_array(time_point, image_channel)
        self._add_to_cache(time_point_number, image_channel, image)
        return image

    def get_channels(self) -> List[ImageChannel]:
        return self._internal.get_channels()

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        return self._internal.get_image_size_zyx()

    def uncached(self) -> ImageLoader:
        return self._internal.uncached()

    def first_time_point_number(self) -> Optional[int]:
        return self._internal.first_time_point_number()

    def last_time_point_number(self) -> Optional[int]:
        return self._internal.last_time_point_number()

    def copy(self) -> ImageLoader:
        return _CachedImageLoader(self._internal.copy())

    def serialize_to_config(self) -> Tuple[str, str]:
        return self._internal.serialize_to_config()


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
            self._offset[time_point_number] = current_offset + offset

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


class Image:
    """Represents a single 3D image"""

    _offset: Position
    _array: ndarray

    def __init__(self, array: ndarray, offset: Position = Position(0, 0, 0)):
        self._array = array
        self._offset = offset

    @property
    def array(self):
        """Gets the raw array. Note that the array indices are not translated, so the position self.offset appears at
        index (0,0,0) in the array."""
        return self._array

    @array.setter
    def array(self, value: ndarray):
        """Sets the raw array. Make sure the new array has the same (x,y,z) size as the old array."""
        if value.shape != self._array.shape:
            raise ValueError("Cannot change size of array")
        self._array = value

    @property
    def offset(self) -> Position:
        """Gets the offset of the array. If the offset is (10, 2, 0), then a position at (11, 2, 0) will appear in the
        array at index (1, 2, 0)"""
        return self._offset

    @property
    def min_x(self) -> int:
        """Gets the lowest X coord for which the image has a value."""
        return int(self._offset.x)

    @property
    def min_y(self) -> int:
        """Gets the lowest X coord for which the image has a value."""
        return int(self._offset.y)

    @property
    def min_z(self) -> int:
        """Gets the lowest X coord for which the image has a value."""
        return int(self._offset.z)

    @property
    def limit_x(self) -> int:
        """Gets the limit in the x direction. If the image has an offset of 10 and a size of 20, the limit will be 30.
        """
        return int(self._offset.x) + self._array.shape[2]

    @property
    def limit_y(self) -> int:
        """Gets the limit in the y direction. If the image has an offset of 10 and a size of 20, the limit will be 30.
        """
        return int(self._offset.y) + self._array.shape[1]

    @property
    def limit_z(self) -> int:
        """Gets the limit in the z direction. If the image has an offset of 10 and a size of 20, the limit will be 30.
        """
        return int(self._offset.z) + self._array.shape[0]


class Images:
    """Records the images (3D + time), their resolution and their offset."""

    _image_loader: ImageLoader
    _offsets: ImageOffsets
    _resolution: Optional[ImageResolution] = None
    _filters: List[ImageFilter]

    def __init__(self):
        self._image_loader = NullImageLoader()
        self._offsets = ImageOffsets()
        self._filters = []

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
            raise UserError("No image resolution set", "No image resolution was set. Please set a resolution first."
                                                       " This can be done in the Edit menu of the program.")
        return self._resolution

    def is_inside_image(self, position: Position, *, margin_xy: int = 0, margin_z: int = 0) -> Optional[bool]:
        """Checks if the given position is inside the images. If there are no images loaded, this returns None. Any
        image offsets (see self.offsets) are taken into account.

        If margins are specified, then this method also returns False for positions that are within the given number of
        pixels towards the edge of the image.
        """
        image_size_zyx = self._image_loader.get_image_size_zyx()
        if image_size_zyx is None:
            return None
        image_position = position - self._offsets.of_time_point(position.time_point())
        if image_position.x < margin_xy or image_position.x >= image_size_zyx[2] - margin_xy:
            return False
        if image_position.y < margin_xy or image_position.y >= image_size_zyx[1] - margin_xy:
            return False
        if image_position.z < margin_z or image_position.z >= image_size_zyx[0] - margin_z:
            return False
        return True

    def get_image(self, time_point: TimePoint) -> Optional[Image]:
        """Gets an image along with offset information, or None if there is no image available for that time point."""
        array = self.get_image_stack(time_point)
        if array is None:
            return None
        return Image(array, self._offsets.of_time_point(time_point))

    def image_loader(self, image_loader: Optional[ImageLoader] = None) -> ImageLoader:
        """Gets/sets the image loader."""
        if image_loader is not None:
            self._image_loader = _CachedImageLoader(image_loader.uncached())
            return image_loader
        return self._image_loader

    def get_image_stack(self, time_point: TimePoint, image_channel: Optional[ImageChannel] = None) -> Optional[ndarray]:
        """Loads an image using the current image loader. Returns None if there is no image for this time point."""
        if image_channel is None:
            # Find the default image channel
            channels = self._image_loader.get_channels()
            if len(channels) == 0:
                return None
            image_channel = channels[0]

        array = self._image_loader.get_image_array(time_point, image_channel)
        if len(self._filters) > 0:
            # Apply all filters
            image_8bit = bits.image_to_8bit(array)
            for image_filter in self._filters:
                image_filter.filter(image_8bit)
            array = image_8bit
        return array

    def set_resolution(self, resolution: Optional[ImageResolution]):
        """Sets the image resolution."""
        self._resolution = resolution

    def copy(self) -> "Images":
        """Returns a copy of this images object. Any changes to the copy won't affect this object and vice versa."""
        copy = Images()
        copy._image_loader = self._image_loader.copy()
        copy._resolution = self._resolution  # No copy, as this object is immutable
        copy._offsets = self._offsets.copy()
        copy._filters = [filter.copy() for filter in self._filters]
        return copy

    def time_points(self) -> Iterable[TimePoint]:
        """Gets all time points for which images are available."""
        min_time_point_number = self._image_loader.first_time_point_number()
        max_time_point_number = self._image_loader.last_time_point_number()
        if min_time_point_number is None or max_time_point_number is None:
            return
        for time_point_number in range(min_time_point_number, max_time_point_number + 1):
            yield TimePoint(time_point_number)

    @property
    def filters(self) -> List[ImageFilter]:
        """Gets a mutable list of all filters applied to the images."""
        return self._filters
