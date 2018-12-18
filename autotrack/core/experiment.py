from typing import Optional, List, Tuple, Iterable

from numpy import ndarray

from autotrack.core import TimePoint, Name, UserError, min_none, max_none
from autotrack.core.image_loader import ImageLoader
from autotrack.core.links import PositionLinks
from autotrack.core.positions import Position, PositionCollection
from autotrack.core.data_axis import DataAxisCollection
from autotrack.core.resolution import ImageResolution
from autotrack.core.score import ScoreCollection


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


class Experiment:
    """A complete experiment, with many stacks of images collected over time. This class ultimately collects all
    details of the experiment."""

    # Note: none of the fields may be None after __init__ is called
    _positions: PositionCollection
    scores: ScoreCollection
    _links: PositionLinks
    _image_loader: ImageLoader = ImageLoader()
    _name: Name
    data_axes: DataAxisCollection
    _image_resolution: Optional[ImageResolution] = None

    def __init__(self):
        self._name = Name()
        self._positions = PositionCollection()
        self.scores = ScoreCollection()
        self.data_axes = DataAxisCollection()
        self._links = PositionLinks()

    def remove_position(self, position: Position):
        """Removes both a position and its links from the experiment."""
        self._positions.detach_position(position)
        self._links.remove_links_of_position(position)

    def move_position(self, old_position: Position, position_new: Position) -> bool:
        """Moves the position of a position, preserving any links. (So it's different from remove-and-readd.) The shape
        of a position is not preserved, though. Throws ValueError when the position is moved to another time point. If
        the new position has not time point specified, it is set to the time point o the existing position."""
        position_new.with_time_point_number(old_position.time_point_number())  # Make sure both have the same time point

        # Replace in linking graphs
        self._links.replace_position(old_position, position_new)

        # Replace in position collection
        self._positions.detach_position(old_position)
        self._positions.add(position_new)
        return True

    def remove_positions(self, time_point: TimePoint):
        """Removes the positions and links of a given time point."""
        for position in self._positions.of_time_point(time_point):
            self._links.remove_links_of_position(position)
        self._positions.detach_all_for_time_point(time_point)

    def get_time_point(self, time_point_number: int) -> TimePoint:
        """Gets the time point with the given number. Throws ValueError if no such time point exists. This method is
        essentially an alternative for `TimePoint(time_point_number)`, but with added bound checks."""
        first = self.first_time_point_number()
        last = self.last_time_point_number()
        if first is None or last is None:
            raise ValueError("No time points have been loaded yet")
        if time_point_number < first or time_point_number > last:
            raise ValueError(f"Time point out of bounds (was: {time_point_number}, first: {first}, last: {last})")
        return TimePoint(time_point_number)

    def first_time_point_number(self) -> Optional[int]:
        """Gets the first time point of the experiment where there is data (images and/or positions)."""
        return min_none(self._image_loader.first_time_point_number(),
                        self._positions.first_time_point_number(),
                        self.data_axes.first_time_point_number())

    def last_time_point_number(self) -> Optional[int]:
        """Gets the last time point (inclusive) of the experiment where there is data (images and/or positions)."""
        return max_none(self._image_loader.last_time_point_number(),
                        self._positions.last_time_point_number(),
                        self.data_axes.last_time_point_number())

    def get_previous_time_point(self, time_point: TimePoint) -> TimePoint:
        """Gets the time point directly before the given time point. Throws KeyError if the given time point is the last
        time point."""
        return self.get_time_point(time_point.time_point_number() - 1)

    def get_next_time_point(self, time_point: TimePoint) -> TimePoint:
        """Gets the time point directly after the given time point. Throws KeyError if the given time point is the last
         time point."""
        return self.get_time_point(time_point.time_point_number() + 1)

    def time_points(self) -> Iterable[TimePoint]:
        first_number = self.first_time_point_number()
        last_number = self.last_time_point_number()
        if first_number is None or last_number is None:
            return []

        current_number = first_number
        while current_number <= last_number:
            yield self.get_time_point(current_number)
            current_number += 1

    def image_loader(self, image_loader: Optional[ImageLoader] = None) -> ImageLoader:
        """Gets/sets the image loader."""
        if image_loader is not None:
            self._image_loader = _CachedImageLoader(image_loader.uncached())
            return image_loader
        return self._image_loader

    def get_image_stack(self, time_point: TimePoint) -> Optional[ndarray]:
        """Gets a stack of all images for a time point, one for every z layer. Returns None if there is no image."""
        return self._image_loader.get_image_stack(time_point)

    def image_resolution(self, *args: Optional[ImageResolution]):
        """Gets or sets the image resolution. Throws UserError if you try to get the resolution when none has been set.

        Set the image resolution:
        >>> self.image_resolution(ImageResolution(0.32, 0.32, 0.32, 12))

        Get the image resolution:
        >>> self.image_resolution()

        Delete the image resolution:
        >>> self.image_resolution(None)
        """
        if len(args) == 1:
            self._image_resolution = args[0]
        if len(args) > 1:
            raise ValueError(f"Too many args: {args}")
        if self._image_resolution is None:
            raise UserError("No image resolution set", "No image resolution was set. Please set a resolution first.")
        return self._image_resolution

    @property
    def positions(self) -> PositionCollection:
        """Gets all positions of all time points."""
        return self._positions

    @property
    def name(self) -> Name:
        # Don't allow to replace the Name object
        return self._name

    @property
    def links(self) -> PositionLinks:
        """Gets all links between the positions of different time points."""
        # Using a property to prevent someone from setting links to None
        return self._links

    @links.setter
    def links(self, links: PositionLinks):
        """Sets the links to the given value. May not be None."""
        if not isinstance(links, PositionLinks):
            raise ValueError("links must be a PositionLinks object, was " + repr(links))
        self._links = links

    @property
    def division_lookahead_time_points(self):
        """Where there no divisions found because a cell really didn't divide, or did the experiment simply end before
        the cell divided? If the experiment continues for at least this many time points, then we can safely assume that
         the cell did not divide."""
        return 80
