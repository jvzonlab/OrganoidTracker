from abc import abstractmethod, ABC
from typing import Optional, Dict, List, Iterable, Tuple

from numpy import ndarray

from organoid_tracker.core import TimePoint
from organoid_tracker.core.image_loader import ImageChannel


class ImageFilter(ABC):
    """Filter for images, for example to enhance the contrast."""

    @abstractmethod
    def filter(self, time_point: TimePoint, image_z: Optional[int], image: ndarray):
        """Filters the given input array, which is a grayscale array of 2 or 3 dimensions. If it is three dimensions,
        then image_z is None. Note that the image_z does not include any image offsets, so z=0 will always be the
        lowest image plane.
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


class ImageFilters:
    _filters: Dict[ImageChannel, List[ImageFilter]]

    def __init__(self):
        self._filters = dict()

    def filter(self, time_point: TimePoint, image_channel: ImageChannel, image_z: Optional[int], array: Optional[ndarray]):
        """Applies the filters to the given array. For 2D arrays, supply an image_z. For 3D arrays, you use None
        instead."""
        if array is None:
            return None

        if image_channel in self._filters:
            # Apply all filters (we need to make a copy of the array, otherwise we modify cached arrays)
            copied_array = array.copy()
            for image_filter in self._filters[image_channel]:
                image_filter.filter(time_point, image_z, copied_array)
            return copied_array

        return array

    def clear_channel(self, channel: ImageChannel):
        """Removes all filters for the given channel."""
        if channel in self._filters:
            del self._filters[channel]

    def add_filter(self, channel: ImageChannel, filter: ImageFilter):
        """Adds a new filter for the given channel."""
        if channel not in self._filters:
            self._filters[channel] = [filter]
        else:
            self._filters[channel].append(filter)

    def of_channel(self, channel: ImageChannel) -> Iterable[ImageFilter]:
        """Gets all filters for the given channel."""
        if channel in self._filters:
            yield from self._filters[channel]

    def copy(self) -> "ImageFilters":
        """Makes a copy of this object."""
        copy = ImageFilters()
        for channel, filters in self._filters.items():
            for filter in filters:
                copy.add_filter(channel, filter.copy())
        return copy

    def items(self) -> Iterable[Tuple[ImageChannel, List[ImageFilter]]]:
        """Gets all channels and filters."""
        for channel, filters in self._filters.items():
            yield channel, list(filters)

    def has_filters(self) -> bool:
        """Returns True if there are any filters stored here, False otherwise."""

        # Note: this method assumes that if a channel has no filters, it won't have any entry in self._filters, not even
        # an entry like `channel -> empty list`.
        return len(self._filters) > 0
