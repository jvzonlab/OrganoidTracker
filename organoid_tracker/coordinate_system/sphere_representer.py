from typing import Iterable, List, Optional, Dict

from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from organoid_tracker.coordinate_system.spherical_coordinates import SphericalCoordinate
from organoid_tracker.core import TimePoint
from organoid_tracker.core.beacon_collection import BeaconCollection
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.core.typing import MPLColor
from organoid_tracker.core.vector import Vector3, Vector2
from organoid_tracker.util import mpl_helper_3d


class SphereRepresentation:
    _beacons: BeaconCollection
    _beacon_index: int
    _resolution: ImageResolution

    _tracks: List[List[Position]]
    _track_colors: List[List[MPLColor]]
    _lone_positions: List[Position]
    _lone_positions_colors: List[MPLColor]

    _orientation_tracks: List[List[Position]]
    _orientation_positions: List[Position]
    _orientation_positions_by_time_point: Dict[TimePoint, Position]

    def __init__(self, beacons: BeaconCollection, beacon_index: int, resolution: ImageResolution):
        self._beacons = beacons
        self._beacon_index = beacon_index
        self._resolution = resolution

        self._tracks = list()
        self._track_colors = list()
        self._lone_positions = list()
        self._lone_positions_colors = list()
        self._orientation_tracks = list()
        self._orientation_positions = list()
        self._orientation_positions_by_time_point = dict()

    def add_point(self, position: Position, *, color: MPLColor = "black"):
        """Adds a single point to this sphere representation."""
        self._lone_positions.append(position)
        self._lone_positions_colors.append(color)

    def add_track(self, track: Iterable[Position], *, colors: Optional[Iterable[MPLColor]] = None,
                  highlight_first: bool = True, highlight_last: bool = False):
        """Adds a list of positions to be displayed on the sphere."""
        track_as_list = list(track)
        colors_as_list = list(colors) if colors is not None else ["black"] * len(track_as_list)
        if len(colors_as_list) != len(track_as_list):
            raise ValueError(f"Supplied {len(track_as_list)} positions, but {len(colors_as_list)} colors")
        self._tracks.append(track_as_list)
        self._track_colors.append(colors_as_list)

        # Highlight the first position
        if highlight_first and len(track_as_list) > 0:
            self._lone_positions.append(track_as_list[0])
            self._lone_positions_colors.append(colors_as_list[0])
        # Highlight the last position
        if highlight_last and len(track_as_list) > 0:
            self._lone_positions.append(track_as_list[-1])
            self._lone_positions_colors.append(colors_as_list[-1])

    def add_orientation_track(self, track: Iterable[Position], *, highlight_last: bool = True):
        """Adds a list of positions to be displayed further away from the sphere."""
        track_as_list = list(track)
        self._orientation_tracks.append(track_as_list)
        self._orientation_positions_by_time_point.clear()
        # This is the first orientation track, use it to align everything
        for position in track_as_list:
            self._orientation_positions_by_time_point[position.time_point()] = position

        # Highlight the last position
        if highlight_last and len(track_as_list) > 0:
            self._orientation_positions.append(track_as_list[-1])

    def _get_beacon(self, position: Position) -> Optional[Position]:
        return self._beacons.get_beacon_by_index(position.time_point(), self._beacon_index)

    def draw_3d(self, ax: Axes3D, sphere_origin_px: Vector3):
        # Start with the sphere
        mpl_helper_3d.draw_sphere(ax, color=(1, 0, 0, 0.2), radius=0.9, center=sphere_origin_px)

        self._draw_tracks_3d_color(ax, self._tracks, self._track_colors, center=sphere_origin_px)
        self._draw_positions_3d_color(ax, self._lone_positions, self._lone_positions_colors, marker='o',
                                      center=sphere_origin_px, edgecolors="black")
        self._draw_tracks_3d(ax, self._orientation_tracks, radius=1.8, color="red", center=sphere_origin_px)

        # Plot the final orientation points as lines
        for position in self._orientation_positions:
            # Plot final location
            projected_nearby = self._get_draw_coords_3d(position, radius=0.9)
            projected_further_away = self._get_draw_coords_3d(position, radius=1.8)
            ax.plot([projected_nearby.x + sphere_origin_px.x, projected_further_away.x + sphere_origin_px.x],
                    [projected_nearby.y + sphere_origin_px.y, projected_further_away.y + sphere_origin_px.y],
                    [projected_nearby.z + sphere_origin_px.z, projected_further_away.z + sphere_origin_px.z],
                    linewidth=5, color="red")

    def draw_2d(self, ax: Axes):
        self._draw_tracks_2d_color(ax, self._tracks, self._track_colors)
        self._draw_positions_2d_color(ax, self._lone_positions, self._lone_positions_colors, marker='o', edgecolors="black")
        self._draw_tracks_2d(ax, self._orientation_tracks, color="red")

        # Plot the final orientation points as lines
        for position in self._orientation_positions:
            # Plot final location
            projected_nearby = self._get_draw_coords_2d(position)
            ax.scatter([projected_nearby.x], [projected_nearby.y], marker="o", s=15, color="red")

    def _get_draw_coords(self, position: Position) -> Optional[SphericalCoordinate]:
        """Gets the drawing coordinates of the given position in the spherical coordinate system."""
        # Translate for beacon
        beacon = self._get_beacon(position)
        if beacon is None:
            return None
        relative = position - beacon
        projected = SphericalCoordinate.from_cartesian(relative.to_vector_um(self._resolution))

        # Rotate for orientation_position (optional)
        orientation_position = self._orientation_positions_by_time_point.get(position.time_point())
        if orientation_position is not None:
            relative_orientation_position = orientation_position - beacon
            projected_orientation_position = SphericalCoordinate.from_cartesian(relative_orientation_position
                                                                                .to_vector_um(self._resolution))
            projected = projected.angular_difference(projected_orientation_position)

        return projected

    def _get_draw_coords_3d(self, position: Position, *, radius: float = 1) -> Optional[Vector3]:
        """Returns a Vector3 that is the projection of the position on a sphere with the given radius."""
        projected = self._get_draw_coords(position)
        if projected is not None:
            return projected.to_cartesian(radius=radius)
        return None

    def _get_draw_coords_2d(self, position: Position) -> Optional[Vector2]:
        """Returns a Vector2 with y=latitude and x=longitude."""
        projected = self._get_draw_coords(position)
        if projected is not None:
            return Vector2(projected.longitude_degrees(), projected.latitude_degrees())
        return None

    def _draw_tracks_3d_color(self, ax: Axes3D, tracks: List[List[Position]], colors: List[List[MPLColor]], *,
                              radius: float = 1, center: Vector3 = Vector3(0, 0, 0), **kwargs):
        """Draws lines along the given tracks using multiple colors per track."""
        for track, track_colors in zip(tracks, colors):
            previous_draw_coords = None

            # Draw entire track
            for position, color in zip(track, track_colors):
                draw_coords = self._get_draw_coords_3d(position, radius=radius)
                if draw_coords is not None and previous_draw_coords is not None:
                    ax.plot([previous_draw_coords.x + center.x, draw_coords.x + center.x],
                            [previous_draw_coords.y + center.y, draw_coords.y + center.y],
                            [previous_draw_coords.z + center.z, draw_coords.z + center.z], color=color, **kwargs)
                previous_draw_coords = draw_coords

    def _draw_tracks_2d_color(self, ax: Axes, tracks: List[List[Position]], colors: List[List[MPLColor]], **kwargs):
        """Draws lines along the given tracks using multiple colors per track."""
        for track, track_colors in zip(tracks, colors):
            previous_draw_coords = None

            # Draw entire track
            for position, color in zip(track, track_colors):
                draw_coords = self._get_draw_coords_2d(position)
                if draw_coords is not None and previous_draw_coords is not None:
                    ax.plot([previous_draw_coords.x, draw_coords.x],
                            [previous_draw_coords.y, draw_coords.y], color=color, **kwargs)
                previous_draw_coords = draw_coords

    def _draw_tracks_3d(self, ax: Axes3D, tracks: List[List[Position]], *,
                        radius: float = 1, center: Vector3 = Vector3(0, 0, 0), **kwargs):
        """Draws lines along the given tracks."""
        for track in tracks:
            previous_draw_coords = None

            # Draw entire track
            for position in track:
                draw_coords = self._get_draw_coords_3d(position, radius=radius)
                if draw_coords is not None and previous_draw_coords is not None:
                    ax.plot([previous_draw_coords.x + center.x, draw_coords.x + center.x],
                            [previous_draw_coords.y + center.y, draw_coords.y + center.y],
                            [previous_draw_coords.z + center.z, draw_coords.z + center.z], **kwargs)
                previous_draw_coords = draw_coords

    def _draw_tracks_2d(self, ax: Axes, tracks: List[List[Position]], **kwargs):
        """Draws lines along the given tracks."""
        for track in tracks:
            points_x = []
            points_y = []

            # Draw entire track
            for position in track:
                draw_coords = self._get_draw_coords_2d(position)
                if draw_coords is not None:
                    points_x.append(draw_coords.x)
                    points_y.append(draw_coords.y)
            ax.plot(points_x, points_y, **kwargs)

    def _draw_positions_3d_color(self, ax: Axes3D, positions: List[Position], colors: List[MPLColor], *,
                                 radius: float = 1, center: Vector3 = Vector3(0, 0, 0), **kwargs):
        """Draws the given positions (no lines), correctly projected on the sphere."""
        x_list, y_list, z_list = list(), list(), list()
        for position in positions:
            draw_coords = self._get_draw_coords_3d(position, radius=radius)
            if draw_coords is None:
                continue
            x_list.append(draw_coords.x + center.x)
            y_list.append(draw_coords.y + center.y)
            z_list.append(draw_coords.z + center.z)
        ax.scatter(x_list, y_list, z_list, color=colors, **kwargs)

    def _draw_positions_2d_color(self, ax: Axes, positions: List[Position], colors: List[MPLColor], **kwargs):
        """Draws the given positions (no lines), correctly projected on the world map."""
        x_list, y_list = list(), list()
        for position in positions:
            draw_coords = self._get_draw_coords_2d(position)
            if draw_coords is None:
                continue
            x_list.append(draw_coords.x)
            y_list.append(draw_coords.y)
        ax.scatter(x_list, y_list, color=colors, **kwargs)


def setup_figure_3d(figure: Figure, sphere_representation: SphereRepresentation):
    from mpl_toolkits.mplot3d import Axes3D
    # Right now, this ignores the fact that there may be multiple spheres, and just plots everything on a single sphere

    # noinspection PyTypeChecker
    ax: Axes3D = figure.add_subplot(111, projection='3d')

    sphere_representation.draw_3d(ax, Vector3(0, 0, 0))

    ax.set_xticks([-1, 1])
    ax.set_yticks([-1, 1])
    ax.set_zticks([-1, 1])
    ax.set_xticklabels(["-", "+"])
    ax.set_yticklabels(["-", "+"])
    ax.set_zticklabels(["-", "+"])
    ax.set_xlim([-1, 1])
    ax.set_ylim([1, -1])
    ax.set_zlim([1, -1])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')


def setup_figure_2d(ax: Axes, sphere_representation: SphereRepresentation):
    sphere_representation.draw_2d(ax)
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
