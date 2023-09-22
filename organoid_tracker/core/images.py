from typing import Dict, Optional, List, Tuple, Iterable, Union, Any

import numpy
from numpy import ndarray

from organoid_tracker.core import TimePoint, UserError
from organoid_tracker.core.bounding_box import BoundingBox
from organoid_tracker.core.image_filters import ImageFilter, ImageFilters
from organoid_tracker.core.image_loader import ImageLoader, ImageChannel, NullImageLoader
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution, ImageTimings

_ZERO = Position(0, 0, 0)


class _CachedImageLoader(ImageLoader):
    """Wrapper that caches the last few loaded images."""

    _internal: ImageLoader
    _image_cache: List[Tuple[int, int, ImageChannel, ndarray]]
    _CACHE_SIZE: int = 5 * 30

    def __init__(self, wrapped: ImageLoader):
        self._image_cache = []
        self._internal = wrapped

    def _add_to_cache(self, time_point_number: int, image_z: int, image_channel: ImageChannel, image: ndarray):
        if len(self._image_cache) > self._CACHE_SIZE:
            self._image_cache.pop(0)
        self._image_cache.append((time_point_number, image_z, image_channel, image))

    def get_3d_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        time_point_number = time_point.time_point_number()

        z_size = self._internal.get_image_size_zyx()[0]
        image_layers_by_z: List[Optional[ndarray]] = [None] * z_size
        for entry in self._image_cache:
            if entry[0] == time_point_number and entry[2] == image_channel:
                cached_z: int = entry[1]
                try:
                    # Found cache entry for this z
                    image_layers_by_z[cached_z] = entry[3]
                except IndexError:
                    pass  # Ignore, image contains extra z levels
        if self._is_complete(image_layers_by_z):
            # Collected all necessary cache entries
            return numpy.array(image_layers_by_z, dtype=image_layers_by_z[0].dtype)

        # Cache miss
        array = self._internal.get_3d_image_array(time_point, image_channel)
        if array is None:
            return None
        if array.shape[0] * 2 < self._CACHE_SIZE:
            # The 3D image is small enough for cache, so add it
            for image_z in range(array.shape[0]):
                self._add_to_cache(time_point.time_point_number(), image_z, image_channel, array[image_z])
        return array

    def _is_complete(self, arrays: List[Optional[ndarray]]):
        for array in arrays:
            if array is None:
                return False
        return True

    def get_2d_image_array(self, time_point: TimePoint, image_channel: ImageChannel, image_z: int) -> Optional[ndarray]:
        time_point_number = time_point.time_point_number()
        for entry in self._image_cache:
            if entry[0] == time_point_number and entry[1] == image_z and entry[2] == image_channel:
                # Cache hit
                return entry[3]

        # Cache miss
        array = self._internal.get_2d_image_array(time_point, image_channel, image_z)
        self._add_to_cache(time_point.time_point_number(), image_z, image_channel, array)
        return array

    def get_channel_count(self) -> int:
        return self._internal.get_channel_count()

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

    def serialize_to_dictionary(self) -> Dict[str, Any]:
        return self._internal.serialize_to_dictionary()

    def close(self):
        self._internal.close()


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

    @staticmethod
    def zeros_like(image: "Image", *, dtype: Optional = None) -> "Image":
        """Returns a new image consisting of only zeros with the same shape and offset as the existing image."""
        return Image(numpy.zeros_like(image.array, dtype=dtype), offset=image.offset)

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

    @property
    def shape_x(self) -> Tuple[int, int, int]:
        """Gets the size in the x direction in pixels."""
        return self._array.shape[2]

    @property
    def shape_y(self) -> int:
        """Gets the size in the y direction in pixels."""
        return self._array.shape[1]

    @property
    def shape_z(self) -> int:
        """Gets the size in the z direction in pixels."""
        return self._array.shape[0]

    def bounding_box(self) -> BoundingBox:
        """Gets a bounding box that encompasses the entire image."""
        return BoundingBox(self.min_x, self.min_y, self.min_z, self.limit_x, self.limit_y, self.limit_z)

    def value_at(self, position: Position) -> Optional[int]:
        """Gets the value at the given position. Takes the offset of this image into account.
        Returns None if the position is outside the image."""
        position -= self._offset
        if position.x < 0 or position.x >= self._array.shape[2]:
            return None
        if position.y < 0 or position.y >= self._array.shape[1]:
            return None
        if position.z < 0 or position.z >= self._array.shape[0]:
            return None
        return self._array[int(position.z), int(position.y), int(position.x)]

    def set_pixel(self, position: Position, value: Union[float, int]):
        """Sets a single pixel, taking the offset of this image into account. Warning: doesn't do bounds checking."""
        self._array[int(position.z - self.offset.z),
        int(position.y - self.offset.y),
        int(position.x - self.offset.x)] = value


class Images:
    """Records the images (3D + time), their resolution and their offset."""

    _image_loader: ImageLoader
    _offsets: ImageOffsets
    _resolution: ImageResolution
    _timings: Optional[ImageTimings] = None
    filters: ImageFilters

    def __init__(self):
        self._image_loader = NullImageLoader()
        self._offsets = ImageOffsets()
        self._resolution = ImageResolution(0, 0, 0, 0)
        self.filters = ImageFilters()

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

    def resolution(self, allow_incomplete: bool = False) -> ImageResolution:
        """Gets the image resolution. Raises UserError no spatial resolution has been set. Also raises UserError if
        the data has multiple time points, but no time resolution has been set.
        """
        require_time_resolution = self.first_time_point_number() != self.last_time_point_number()
        require_z = False
        image_size = self._image_loader.get_image_size_zyx()
        if image_size is not None and image_size[0] > 1:
            require_z = True  # We have 3D images, require a Z resolution
        if not allow_incomplete and self._resolution.is_incomplete(require_time_resolution=require_time_resolution,
                                                                   require_z=require_z):
            raise UserError("No image resolution set", "No image resolution was set. Please set a resolution first."
                                                       " This can be done in the Edit menu of the program.")
        return self._resolution

    def has_timings(self) -> bool:
        """Checks if the experiment has timing information for the time points, i.e. how long each time point is.

        If explicit is True, then we only return True if the timings information was provided explicitly, instead of
        just using the time resolution.
        """
        return self._timings is not None or self._resolution.time_point_interval_m > 0

    def timings(self) -> ImageTimings:
        """Gets the timings of all time points of the experiment. If they aren't found, but a time resolution is
        specified, then constant timing is assumed. If no time resolution, and no explicit timings are found, a
        UserError is raised."""
        if self._timings is None:
            if self._resolution.time_point_interval_m > 0:
                # No timings, but a time resolution was provided. Assume constant timing.
                self._timings = ImageTimings.contant_timing(self._resolution.time_point_interval_m)
                return self._timings
            raise UserError("No time resolution set", "No time resolution was set. Please set a resolution first."
                                                      " This can be done in the Edit menu of the program.")
        return self._timings

    def set_timings(self, timings: Optional[ImageTimings]):
        """Sets explicit timings for all time points in the experiment. Useful if not all time points have the same
        time resolution.

        If you set the timings to None, only the time resolution in ImageResolution will be used.

        Note: in ImageResolution, the time interval is updated to match t(1) - t(0). If you later set a different
        time resolution, then that will delete the timings information.
        """
        if timings is None:
            self._timings = None
            return
        if not isinstance(timings, ImageTimings):
            raise ValueError("Not an ImageTimings instance: " + repr(timings))
        self._timings = timings

        # Also keep time resolution in sync
        self._resolution = ImageResolution(self._resolution.pixel_size_x_um, self._resolution.pixel_size_y_um,
                                           self._resolution.pixel_size_z_um,
                                           timings.get_time_m_since_previous(TimePoint(1)))

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

    def get_image(self, time_point: TimePoint, image_channel: ImageChannel = ImageChannel(index_zero=0)) -> Optional[
        Image]:
        """Gets an image along with offset information, or None if there is no image available for that time point."""
        array = self.get_image_stack(time_point, image_channel)
        if array is None:
            return None
        return Image(array, self._offsets.of_time_point(time_point))

    def image_loader(self, image_loader: Optional[ImageLoader] = None) -> ImageLoader:
        """Gets/sets the image loader. Note: images loaded directly from this image loader will be uncached.

        Warning: consider whether you need to close the old image loader first.
        """
        if image_loader is not None:
            self._image_loader = _CachedImageLoader(image_loader)
            return image_loader
        return self._image_loader.uncached()

    def use_image_loader_from(self, images: "Images"):
        """Transfers the image loader from another Images instance, sharing the image cache."""
        self._image_loader = images._image_loader

    def get_image_stack(self, time_point: TimePoint, image_channel: ImageChannel = ImageChannel(index_zero=0)) -> \
    Optional[ndarray]:
        """Loads an image using the current image loader. Returns None if there is no image for this time point."""
        array = self._image_loader.get_3d_image_array(time_point, image_channel)
        return self.filters.filter(time_point, image_channel, None, array)

    def get_image_slice_2d(self, time_point: TimePoint, image_channel: ImageChannel, z: int) -> Optional[ndarray]:
        """Gets a 2D grayscale image for the given time point, image channel and z."""
        offset_z = self._offsets.of_time_point(time_point).z
        image_z = int(z - offset_z)
        array = self._image_loader.get_2d_image_array(time_point, image_channel, image_z)
        return self.filters.filter(time_point, image_channel, image_z, array)

    def set_resolution(self, resolution: Optional[ImageResolution]):
        """Sets the image resolution."""
        if resolution is None:
            resolution = ImageResolution(0, 0, 0, 0)
        self._resolution = resolution

        # Keep timings in sync
        time_resolution_m = self._resolution.time_point_interval_m
        if self._timings is not None and self._timings.get_time_m_since_previous(TimePoint(1)) != time_resolution_m:
            self._timings = None  # Delete timing information, as it's not in sync with the resolution anymore

    def copy(self) -> "Images":
        """Returns a copy of this images object. Any changes to the copy won't affect this object and vice versa."""
        copy = Images()
        copy._image_loader = self._image_loader.copy()
        copy._resolution = self._resolution  # No copy, as this object is immutable
        copy._offsets = self._offsets.copy()
        copy.filters = self.filters.copy()
        return copy

    def first_time_point_number(self) -> Optional[int]:
        """Gets the first time point with an image, if any."""
        return self._image_loader.first_time_point_number()

    def last_time_point_number(self) -> Optional[int]:
        """Gets the last time point with an image, if any."""
        return self._image_loader.last_time_point_number()

    def time_points(self) -> Iterable[TimePoint]:
        """Gets all time points for which images are available."""
        min_time_point_number = self._image_loader.first_time_point_number()
        max_time_point_number = self._image_loader.last_time_point_number()
        if min_time_point_number is None or max_time_point_number is None:
            return
        for time_point_number in range(min_time_point_number, max_time_point_number + 1):
            yield TimePoint(time_point_number)

    def get_channels(self) -> List[ImageChannel]:
        """Gets all available image channels. These are determined by the image_loader."""
        return self._image_loader.get_channels()

    def close_image_loader(self):
        """Closes the image loader, releasing file system handles, and replacing it with a dummy one that contains no
        images."""
        self._image_loader.close()
        self._image_loader = NullImageLoader()
