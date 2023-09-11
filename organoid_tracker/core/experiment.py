from typing import Optional, Iterable

from numpy import ndarray

from organoid_tracker.core import TimePoint, Name, UserError, min_none, max_none, Color
from organoid_tracker.core.beacon_collection import BeaconCollection
from organoid_tracker.core.connections import Connections
from organoid_tracker.core.global_data import GlobalData
from organoid_tracker.core.images import Images
from organoid_tracker.core.link_data import LinkData
from organoid_tracker.core.links import Links
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.spline import SplineCollection
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.core.warning_limits import WarningLimits


class Experiment:
    """A complete experiment, with many stacks of images collected over time. This class ultimately collects all
    details of the experiment.

    An experiment contains :class:`~organoid_tracker.core.position.Position` objects. They represent a cell detection
    at a particular time point. The object itself just stores an x, y, z and time point. The objects can be found in
    :code:`experiment.positions`. For example, to get a list of all positions at time point 7, use
    :code:`list(experiment.positions.of_time_point(TimePoint(7))`.

    Cell detections from different time points can be marked as belonging to the same biological cell. These links over
    time are stored in :code:`experiment.links`. Using for example :code:`experiment.links.find_futures(position)`,
    you'll know what  position in the next time point represents locates same cell. This will usually be one position,
    but in the case of a division, it will be two positions. In the case of a cell death, it will be 0 positions.

    Cells can also have neighbors, and those are defined in :code:`experiment.connections`. Metadata of cell positions,
    like the cell type or fluorescent intensity, is stored in :code:`experiment.position_data`.
    """

    # Note: none of the fields may be None after __init__ is called
    _positions: PositionCollection  # Used to mark cell positions
    _beacons: BeaconCollection  # Used to mark some abstract position that the cells move to or move around.
    _links: Links  # Used to link cells together accross multiple time points
    _position_data: PositionData  # Used for metadata of cells
    _link_data: LinkData  # Used for metadata of links
    _images: Images  # Used for microscopy images
    _connections: Connections  # Used to create connections in a single time point, for example for related cells
    _name: Name  # Name of the experiment
    splines: SplineCollection  # Splines, can be used to track progress of a cell across a trajectory.
    _warning_limits: WarningLimits
    _color: Color
    last_save_file: Optional[str] = None  # Location where the "Save" button will save again.
    _global_data: GlobalData

    def __init__(self):
        self._name = Name()
        self._positions = PositionCollection()
        self._beacons = BeaconCollection()
        self._position_data = PositionData()
        self.splines = SplineCollection()
        self._links = Links()
        self._link_data = LinkData()
        self._images = Images()
        self._connections = Connections()
        self._warning_limits = WarningLimits()
        self._color = Color.black()
        self._global_data = GlobalData()

    def remove_position(self, position: Position, *, update_splines: bool = True):
        """Removes a position and its links and other data from the experiment.

        When update_splines is set to False, the zero point of all splines (if any) will not be updated. This makes
        moving the positions a lot faster. However, you should call splines.update_for_changed_positions yourself after
        moving all positions."""
        self.remove_positions([position], update_splines=update_splines)

    def remove_positions(self, positions: Iterable[Position], *, update_splines: bool = True):
        """Removes multiple positions and their links and other data from the experiment. If you have multiple positions
        to delete, it is more efficient to call this method than to call remove_position many times.

        When update_splines is set to False, the zero point of all splines (if any) will not be updated. This makes
        moving the positions a lot faster. However, you should call splines.update_for_changed_positions yourself after
        moving all positions."""
        affected_time_points = set()
        for position in positions:
            old_links = self._links.find_links_of(position)
            for old_link in old_links:
                self._link_data.remove_link(position, old_link)

            self._positions.detach_position(position)
            self._links.remove_links_of_position(position)
            self._connections.remove_connections_of_position(position)
            self._position_data.remove_position(position)

            affected_time_points.add(position.time_point())

        # Update the data axes origins for all affected time points
        if update_splines:
            for time_point in affected_time_points:
                self.splines.update_for_changed_positions(time_point, self._positions.of_time_point(time_point))

    def move_position(self, position_old: Position, position_new: Position, update_splines: bool = True):
        """Moves the position of a position, preserving any links. (So it's different from remove-and-readd.) The shape
        of a position is not preserved, though. Throws ValueError when the position is moved to another time point. If
        the new position has not time point specified, it is set to the time point o the existing position.

        When update_splines is set to False, the zero point of all splines (if any) will not be updated. This makes
        moving the positions a lot faster. However, you should call splines.update_for_changed_positions yourself after
        moving all positions."""
        position_new.check_time_point(position_old.time_point())  # Make sure both have the same time point

        # Replace in all collections
        for linked_position in self._links.find_links_of(position_old):
            self._link_data.replace_link(position_old, linked_position, position_new, linked_position)
        self._links.replace_position(position_old, position_new)
        self._connections.replace_position(position_old, position_new)
        self._positions.move_position(position_old, position_new)
        self._position_data.replace_position(position_old, position_new)

        if update_splines:
            time_point = position_new.time_point()
            self.splines.update_for_changed_positions(time_point, self._positions.of_time_point(time_point))

    def _scale_to_resolution(self, new_resolution: ImageResolution):
        """Scales this experiment so that it has a different resolution."""
        try:
            old_resolution = self._images.resolution()
        except UserError:
            return  # No resolution was set, do nothing
        else:
            x_factor = new_resolution.pixel_size_x_um / old_resolution.pixel_size_x_um
            z_factor = new_resolution.pixel_size_z_um / old_resolution.pixel_size_z_um
            t_factor = 1 if (new_resolution.time_point_interval_m == 0 or old_resolution.time_point_interval_m == 0) \
                else new_resolution.time_point_interval_m / old_resolution.time_point_interval_m
            if t_factor < 0.9 or t_factor > 1.1:
                # We cannot scale in time, unfortunately. Links must go from one time point to the next time point.
                # So we throw an error if the scale changes too much
                raise ValueError(f"Cannot change time scale; existing scale {old_resolution.time_point_interval_m},"
                                 f" new scale {new_resolution.time_point_interval_m}")
            if abs(x_factor - 1) < 0.0001 and abs(z_factor - 1) < 0.0001:
                return  # Nothing to scale
            scale_factor = Position(x_factor, x_factor, z_factor)
            print(f"Scaling to {scale_factor}")
            for time_point in self.time_points():
                positions = list(self.positions.of_time_point(time_point))
                for position in positions:
                    self.move_position(position, position * scale_factor)
        self.images.set_resolution(new_resolution)

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
                        self.splines.first_time_point_number())

    def last_time_point_number(self) -> Optional[int]:
        """Gets the last time point (inclusive) of the experiment where there is data (images and/or positions)."""
        return max_none(self._images.image_loader().last_time_point_number(),
                        self._positions.last_time_point_number(),
                        self.splines.last_time_point_number())

    def get_previous_time_point(self, time_point: TimePoint) -> TimePoint:
        """Gets the time point directly before the given time point. Throws ValueError if the given time point is the
        first time point of the experiment."""
        return self.get_time_point(time_point.time_point_number() - 1)

    def get_next_time_point(self, time_point: TimePoint) -> TimePoint:
        """Gets the time point directly after the given time point. Throws ValueError if the given time point is the
        last time point of the experiment."""
        return self.get_time_point(time_point.time_point_number() + 1)

    def time_points(self) -> Iterable[TimePoint]:
        """Returns an iterable over all time points, so that you can do `for time_point in experiment.time_points():`.
        """
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
    def color(self) -> Color:
        """Returns the color of this experiment. This color is used in various plots where multiple experiments are
        summarized."""
        return self._color

    @color.setter
    def color(self, color: Color):
        """Sets the color of this experiment. This color is used in various plots where multiple experiments are
        summarized."""
        if not isinstance(color, Color):
            raise TypeError(f"color must be a {Color.__name__} object, was " + repr(color))
        self._color = color

    @property
    def positions(self) -> PositionCollection:
        """Gets all cell positions of all time points."""
        return self._positions

    @positions.setter
    def positions(self, positions: PositionCollection):
        """Replaced all the cell positions by a new set."""
        if not isinstance(positions, PositionCollection):
            raise TypeError(f"positions must be a {PositionCollection.__name__} object, was " + repr(positions))
        self._positions = positions

    @property
    def beacons(self) -> BeaconCollection:
        """Gets all beacon positions. Beacons are used to measure the movement of cells towards or around some static
        position."""
        return self._beacons

    @beacons.setter
    def beacons(self, beacons: BeaconCollection):
        """Sets the beacon positions. Beacons are used to measure the movement of cells towards or around some static
        position."""
        if not isinstance(beacons, BeaconCollection):
            raise TypeError(f"beacons must be a {BeaconCollection.__name__} object, was " + repr(beacons))
        self._beacons = beacons

    @property
    def position_data(self) -> PositionData:
        """Gets the metadata associated with cell positions, like cell type, intensity, etc."""
        return self._position_data

    @position_data.setter
    def position_data(self, position_data: PositionData):
        """Replaces the metadata associated with cell positions."""
        if not isinstance(position_data, PositionData):
            raise TypeError(f"position_data must be a {PositionData.__name__} object, was " + repr(position_data))
        self._position_data = position_data

    @property
    def name(self) -> Name:
        """Gets the name of the experiment."""
        # Don't allow to replace the Name object
        return self._name

    @name.setter
    def name(self, name: Name):
        if not isinstance(name, Name):
            raise TypeError("name must be an instance of Name, but as " + str(name))
        self._name = name

    @property
    def warning_limits(self) -> WarningLimits:
        """Gets the limits used by the error checker. For example: what movement is so fast that it should raise an
        error?"""
        return self._warning_limits

    @warning_limits.setter
    def warning_limits(self, warning_limits: WarningLimits):
        """Sets the limits used by the error checker. For example: what movement is too fast?"""
        if not isinstance(warning_limits, WarningLimits):
            raise TypeError(f"warnings_limits must be a {WarningLimits.__name__} object, was " + repr(warning_limits))
        self._warning_limits = warning_limits

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
    def link_data(self) -> LinkData:
        """Gets all metadata of links."""
        # Using a property to prevent someone from setting link_data to None
        return self._link_data

    @link_data.setter
    def link_data(self, link_data: LinkData):
        """Sets the links to the given value. May not be None."""
        if not isinstance(link_data, LinkData):
            raise TypeError(f"link_data must be a {LinkData.__name__} object, was " + repr(link_data))
        self._link_data = link_data

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

    @property
    def global_data(self) -> GlobalData:
        return self._global_data

    @global_data.setter
    def global_data(self, global_data: GlobalData):
        """Sets the global data, after checking the object type"""
        if not isinstance(global_data, GlobalData):
            raise ValueError(f"global_data must be a {GlobalData.__name__} object, was " + repr(global_data))
        self._global_data = global_data

    def merge(self, other: "Experiment"):
        """Merges the position, linking and connections data of two experiments. Images, resolution and scores are not
        yet merged."""

        # Scale the other experiment first
        try:
            resolution = self.images.resolution()
        except UserError:
            pass
        else:
            other._scale_to_resolution(resolution)

        self.positions.add_positions(other.positions)
        self.beacons.add_beacons(other.beacons)
        self.links.add_links(other.links)
        self.position_data.merge_data(other.position_data)
        self.link_data.merge_data(other.link_data)
        self.connections.add_connections(other.connections)
        self.global_data.merge_data(other.global_data)

    def copy_selected(self, *, images: bool = False, positions: bool = False, position_data: bool = False,
                      links: bool = False, link_data: bool = False, global_data: bool = False,
                      connections: bool = False, name: bool = False) -> "Experiment":
        """Copies the selected attributes over to a new experiment. Note that position_data and links can only be copied
        if the positions are copied."""
        copy = Experiment()
        if images:
            copy.images = self._images.copy()
        if positions:
            copy.positions = self._positions.copy()
            if position_data:
                copy.position_data = self._position_data.copy()
            if links:
                copy.links = self._links.copy()
                if link_data:
                    copy.link_data = self._link_data.copy()
        if global_data:
            copy.global_data = self._global_data.copy()
        if connections:
            copy.connections = self._connections.copy()
        if name:
            copy.name = self._name.copy()
        return copy

    def close(self):
        """Closes any system resources of this experiment, like file handles for images on disk."""
        # Right now, we only need to close the images
        self.images.close_image_loader()
