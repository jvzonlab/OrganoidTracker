from typing import Optional, List, Tuple, Iterable

from numpy import ndarray

from autotrack.core import TimePoint, Name, UserError, min_none, max_none
from autotrack.core.connections import Connections
from autotrack.core.image_loader import ImageLoader
from autotrack.core.images import Images
from autotrack.core.links import Links
from autotrack.core.position_collection import PositionCollection
from autotrack.core.position import Position
from autotrack.core.data_axis import DataAxisCollection
from autotrack.core.resolution import ImageResolution
from autotrack.core.score import ScoreCollection



class Experiment:
    """A complete experiment, with many stacks of images collected over time. This class ultimately collects all
    details of the experiment."""

    # Note: none of the fields may be None after __init__ is called
    _positions: PositionCollection
    scores: ScoreCollection
    _links: Links
    _images: Images
    _connections: Connections
    _name: Name
    data_axes: DataAxisCollection

    def __init__(self):
        self._name = Name()
        self._positions = PositionCollection()
        self.scores = ScoreCollection()
        self.data_axes = DataAxisCollection()
        self._links = Links()
        self._images = Images()
        self._connections = Connections()

    def remove_position(self, position: Position):
        """Removes a position and its links and other data from the experiment."""
        self.remove_positions([position])

    def remove_positions(self, positions: Iterable[Position]):
        """Removes multiple positions and their links and other data from the experiment. If you have multiple positions
        to delete, it is more efficient to call this method than to call remove_position many times."""
        affected_time_points = set()
        for position in positions:
            self._positions.detach_position(position)
            self._links.remove_links_of_position(position)
            self._connections.remove_connections_of_position(position)

            affected_time_points.add(position.time_point())

        # Update the data axes origins for all affected time points
        for time_point in affected_time_points:
            self.data_axes.update_for_changed_positions(time_point, self._positions.of_time_point(time_point))

    def move_position(self, position_old: Position, position_new: Position) -> bool:
        """Moves the position of a position, preserving any links. (So it's different from remove-and-readd.) The shape
        of a position is not preserved, though. Throws ValueError when the position is moved to another time point. If
        the new position has not time point specified, it is set to the time point o the existing position."""
        position_new.check_time_point(position_old.time_point())  # Make sure both have the same time point

        # Replace in all collections
        self._links.replace_position(position_old, position_new)
        self._connections.replace_position(position_old, position_new)
        self._positions.move_position(position_old, position_new)

        time_point = position_new.time_point()
        self.data_axes.update_for_changed_positions(time_point, self._positions.of_time_point(time_point))
        return True

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
        return min_none(self._images.image_loader().first_time_point_number(),
                        self._positions.first_time_point_number(),
                        self.data_axes.first_time_point_number())

    def last_time_point_number(self) -> Optional[int]:
        """Gets the last time point (inclusive) of the experiment where there is data (images and/or positions)."""
        return max_none(self._images.image_loader().last_time_point_number(),
                        self._positions.last_time_point_number(),
                        self.data_axes.last_time_point_number())

    def get_previous_time_point(self, time_point: TimePoint) -> TimePoint:
        """Gets the time point directly before the given time point. Throws ValueError if the given time point is the
        first time point of the experiment."""
        return self.get_time_point(time_point.time_point_number() - 1)

    def get_next_time_point(self, time_point: TimePoint) -> TimePoint:
        """Gets the time point directly after the given time point. Throws ValueError if the given time point is the
        last time point of the experiment."""
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

    def get_image_stack(self, time_point: TimePoint) -> Optional[ndarray]:
        """Gets a stack of all images for a time point, one for every z layer. Returns None if there is no image."""
        return self._images.get_image_stack(time_point)

    @property
    def positions(self) -> PositionCollection:
        """Gets all positions of all time points."""
        return self._positions

    @positions.setter
    def positions(self, positions: PositionCollection):
        if not isinstance(positions, PositionCollection):
            raise TypeError(f"positions must be a {PositionCollection.__name__} object, was " + repr(positions))
        self._positions = positions

    @property
    def name(self) -> Name:
        # Don't allow to replace the Name object
        return self._name

    @property
    def links(self) -> Links:
        """Gets all links between the positions of different time points."""
        # Using a property to prevent someone from setting links to None
        return self._links

    @links.setter
    def links(self, links: Links):
        """Sets the links to the given value. May not be None."""
        if not isinstance(links, Links):
            raise TypeError(f"links must be a {Links.__name__} object, was " + repr(links))
        self._links = links

    @property
    def images(self) -> Images:
        """Gets all images stored in the experiment."""
        return self._images

    @images.setter
    def images(self, images: Images):
        """Sets the images to the given value. May not be None."""
        if not isinstance(images, Images):
            raise TypeError(f"images mut be an {Images.__name__} object, was " + repr(images))
        self._images = images

    @property
    def connections(self) -> Connections:
        """Gets the connections, which are used to group positions at the same time point."""
        return self._connections

    @connections.setter
    def connections(self, connections: Connections):
        """Sets the connections, which are used to group positions at the same time point."""
        if not isinstance(connections, Connections):
            raise TypeError(f"connections mut be a {Connections.__name__} object, was " + repr(connections))
        self._connections = connections

    @property
    def division_lookahead_time_points(self):
        """Where there no divisions found because a cell really didn't divide, or did the experiment simply end before
        the cell divided? If the experiment continues for at least this many time points, then we can safely assume that
         the cell did not divide."""
        return 80

    def merge(self, other: "Experiment"):
        """Merges the position, linking and connections data of two experiments. Images, resolution and scores are not
        yet merged."""
        self.positions.add_positions_and_shapes(other.positions)
        self.links.add_links(other.links)
        self.connections.add_connections(other.connections)
